"""Singleton state: loaded data and fitted detectors."""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

from app.config import settings
from app.detection.base import Anomaly
from app.detection.cusum_detector import CUSUMDetector
from app.detection.iqr_detector import IQRDetector
from app.detection.isolation_forest import IsolationForestDetector
from app.detection.zscore_detector import ZScoreDetector

logger = logging.getLogger(__name__)

UNIVARIATE_METRICS = ["revenue", "orders", "avg_check", "unique_customers", "items_sold"]
AVAILABLE_DETECTORS = [
    "zscore", "iqr", "cusum", "prophet", "isolation_forest", "ensemble",
]


@dataclass
class AppState:
    daily: pd.DataFrame | None = None
    zscore: dict[str, ZScoreDetector] = field(default_factory=dict)
    iqr: dict[str, IQRDetector] = field(default_factory=dict)
    cusum: dict[str, CUSUMDetector] = field(default_factory=dict)
    iforest: IsolationForestDetector | None = None
    _prophet_fitted: dict[str, object] = field(default_factory=dict)
    anomaly_cache: list[Anomaly] = field(default_factory=list)


state = AppState()


def load_data() -> pd.DataFrame:
    """Load processed daily metrics. Run preprocessing if file missing."""
    path = Path(settings.processed_data_path)
    if not path.exists():
        logger.info("Processed data not found, running preprocessing pipeline...")
        from app.preprocessing.aggregator import aggregate_daily
        from app.preprocessing.cleaner import load_and_clean

        df = load_and_clean(settings.raw_data_path)
        daily = aggregate_daily(df)
        path.parent.mkdir(parents=True, exist_ok=True)
        daily.to_csv(path)
    else:
        daily = pd.read_csv(path, index_col="date", parse_dates=True)
    return daily


def fit_detectors(daily: pd.DataFrame) -> None:
    """Fit Z-Score, IQR, CUSUM and Isolation Forest detectors."""
    active = daily[~daily["missing_day"].astype(bool)]

    for metric in UNIVARIATE_METRICS:
        series = active[metric]
        series.name = metric

        t0 = time.perf_counter()
        z = ZScoreDetector(
            window=settings.zscore_window,
            warning_threshold=settings.zscore_warning_threshold,
            critical_threshold=settings.zscore_critical_threshold,
        )
        z.fit(series)
        state.zscore[metric] = z
        logger.info("Fitted ZScore/%s in %.0f ms", metric, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        iq = IQRDetector(
            window=settings.iqr_window,
            warning_multiplier=settings.iqr_warning_multiplier,
            critical_multiplier=settings.iqr_critical_multiplier,
        )
        iq.fit(series)
        state.iqr[metric] = iq
        logger.info("Fitted IQR/%s in %.0f ms", metric, (time.perf_counter() - t0) * 1000)

        t0 = time.perf_counter()
        cu = CUSUMDetector(
            drift_factor=settings.cusum_drift_factor,
            warning_factor=settings.cusum_warning_factor,
            critical_factor=settings.cusum_critical_factor,
        )
        cu.fit(series)
        state.cusum[metric] = cu
        logger.info("Fitted CUSUM/%s in %.0f ms", metric, (time.perf_counter() - t0) * 1000)

    # Isolation Forest (multivariate)
    t0 = time.perf_counter()
    ifo = IsolationForestDetector(
        contamination=settings.iforest_contamination,
        warning_score=settings.iforest_warning_score,
        critical_score=settings.iforest_critical_score,
    )
    ifo.fit(active)
    state.iforest = ifo
    logger.info("Fitted IsolationForest in %.0f ms", (time.perf_counter() - t0) * 1000)


def get_prophet_detector(metric: str):
    """Lazy-fit Prophet for a single metric (slow, ~5-10s per metric)."""
    if metric in state._prophet_fitted:
        return state._prophet_fitted[metric]

    from app.detection.prophet_detector import ProphetDetector

    active = state.daily[~state.daily["missing_day"].astype(bool)]
    series = active[metric]
    series.name = metric

    logger.info("Fitting Prophet for %s (this may take a few seconds)...", metric)
    t0 = time.perf_counter()
    p = ProphetDetector(
        warning_interval=settings.prophet_interval_warning,
        critical_interval=settings.prophet_interval_critical,
    )
    p.fit(series)
    state._prophet_fitted[metric] = p
    logger.info("Fitted Prophet/%s in %.0f ms", metric, (time.perf_counter() - t0) * 1000)
    return p


def startup() -> None:
    """Called once at app startup."""
    logger.info("Loading data...")
    state.daily = load_data()
    logger.info("Loaded %d days of metrics", len(state.daily))

    logger.info("Fitting detectors...")
    fit_detectors(state.daily)
    logger.info("Detectors ready.")


def get_active_daily() -> pd.DataFrame:
    """Return daily metrics excluding missing days."""
    return state.daily[~state.daily["missing_day"].astype(bool)]
