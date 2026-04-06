import time
from dataclasses import asdict

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import HTMLResponse

from app.api.dependencies import (
    AVAILABLE_DETECTORS,
    UNIVARIATE_METRICS,
    get_active_daily,
    get_prophet_detector,
    state,
)
from app.detection.base import Anomaly
from app.detection.ensemble import EnsembleDetector
from app.models.schemas import (
    AnomalyResponse,
    DetectionRequest,
    DetectionResponse,
    HealthResponse,
    MetricSummary,
    MetricTimeSeries,
)
from app.visualization.plots import create_metric_plot

router = APIRouter()


# ---------- GET /health ----------

@router.get("/health", response_model=HealthResponse)
def health():
    loaded = state.daily is not None
    if loaded:
        dr = f"{state.daily.index.min().date()} - {state.daily.index.max().date()}"
        total = len(state.daily)
    else:
        dr = ""
        total = 0
    return HealthResponse(
        status="ok" if loaded else "no_data",
        data_loaded=loaded,
        date_range=dr,
        total_days=total,
        available_metrics=UNIVARIATE_METRICS,
        available_detectors=AVAILABLE_DETECTORS,
    )


# ---------- GET /metrics ----------

@router.get("/metrics", response_model=list[MetricSummary])
def list_metrics():
    active = get_active_daily()
    result = []
    for m in UNIVARIATE_METRICS:
        s = active[m]
        result.append(
            MetricSummary(
                metric_name=m,
                count=int(s.count()),
                mean=round(float(s.mean()), 2),
                std=round(float(s.std()), 2),
                min=round(float(s.min()), 2),
                max=round(float(s.max()), 2),
                last_value=round(float(s.iloc[-1]), 2),
                last_date=str(active.index[-1].date()),
            )
        )
    return result


# ---------- GET /metrics/{metric_name} ----------

@router.get("/metrics/{metric_name}", response_model=MetricTimeSeries)
def get_metric(
    metric_name: str,
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    if metric_name not in UNIVARIATE_METRICS:
        raise HTTPException(404, f"Unknown metric: {metric_name}")

    active = get_active_daily()
    series = active[metric_name]

    if start_date:
        series = series[series.index >= start_date]
    if end_date:
        series = series[series.index <= end_date]

    return MetricTimeSeries(
        metric_name=metric_name,
        dates=[str(d.date()) for d in series.index],
        values=[round(float(v), 2) for v in series.values],
    )


# ---------- POST /detect ----------

def _run_univariate(metric: str, detector_name: str) -> list[Anomaly]:
    active = get_active_daily()
    series = active[metric]
    series.name = metric

    if detector_name == "zscore":
        return state.zscore[metric].detect(series)
    elif detector_name == "iqr":
        return state.iqr[metric].detect(series)
    elif detector_name == "prophet":
        p = get_prophet_detector(metric)
        return p.detect(series)
    else:
        raise HTTPException(400, f"Unknown univariate detector: {detector_name}")


def _run_detection(req: DetectionRequest) -> list[Anomaly]:
    metrics = [req.metric_name] if req.metric_name else UNIVARIATE_METRICS

    if req.detector == "isolation_forest":
        active = get_active_daily()
        return state.iforest.detect(active)

    if req.detector == "ensemble":
        all_anomalies: list[list[Anomaly]] = []
        for m in metrics:
            all_anomalies.append(_run_univariate(m, "zscore"))
            all_anomalies.append(_run_univariate(m, "iqr"))
        # Add isolation forest
        active = get_active_daily()
        all_anomalies.append(state.iforest.detect(active))

        ens = EnsembleDetector(
            min_votes_warning=2,
            min_votes_critical=3,
        )
        return ens.combine(*all_anomalies)

    # Single detector
    results: list[Anomaly] = []
    for m in metrics:
        results.extend(_run_univariate(m, req.detector))
    return results


@router.post("/detect", response_model=DetectionResponse)
def detect(req: DetectionRequest):
    if req.detector not in AVAILABLE_DETECTORS:
        raise HTTPException(
            400, f"Unknown detector: {req.detector}. Choose from {AVAILABLE_DETECTORS}"
        )

    t0 = time.perf_counter()
    anomalies = _run_detection(req)

    # Filter by date range
    if req.start_date:
        anomalies = [a for a in anomalies if a.date >= req.start_date]
    if req.end_date:
        anomalies = [a for a in anomalies if a.date <= req.end_date]

    elapsed_ms = (time.perf_counter() - t0) * 1000

    # Cache for GET /anomalies
    state.anomaly_cache = anomalies

    warnings = sum(1 for a in anomalies if a.severity == "warning")
    criticals = sum(1 for a in anomalies if a.severity == "critical")

    return DetectionResponse(
        total_anomalies=len(anomalies),
        warnings=warnings,
        criticals=criticals,
        anomalies=[AnomalyResponse(**asdict(a)) for a in anomalies],
        detection_time_ms=round(elapsed_ms, 1),
    )


# ---------- GET /anomalies ----------

@router.get("/anomalies", response_model=list[AnomalyResponse])
def list_anomalies(
    severity: str | None = Query(None),
    metric: str | None = Query(None),
    start_date: str | None = Query(None),
    end_date: str | None = Query(None),
):
    results = state.anomaly_cache

    if severity:
        results = [a for a in results if a.severity == severity]
    if metric:
        results = [a for a in results if a.metric == metric]
    if start_date:
        results = [a for a in results if a.date >= start_date]
    if end_date:
        results = [a for a in results if a.date <= end_date]

    return [AnomalyResponse(**asdict(a)) for a in results]


# ---------- GET /plot/{metric_name} ----------

@router.get("/plot/{metric_name}", response_class=HTMLResponse)
def plot_metric(metric_name: str):
    if metric_name not in UNIVARIATE_METRICS:
        raise HTTPException(404, f"Unknown metric: {metric_name}")

    active = get_active_daily()
    series = active[metric_name]
    series.name = metric_name

    # Anomalies for this metric (from cache)
    metric_anomalies = [a for a in state.anomaly_cache if a.metric == metric_name]

    # Prophet forecast band (if fitted)
    forecast_lower, forecast_upper = None, None
    if metric_name in state._prophet_fitted:
        p = state._prophet_fitted[metric_name]
        fw = p._forecast_warn
        if fw is not None:
            fc = fw.set_index("ds")
            forecast_lower = fc["yhat_lower"].reindex(series.index)
            forecast_upper = fc["yhat_upper"].reindex(series.index)

    # Weekend mask
    weekend_mask = active["is_weekend"].astype(bool)

    html = create_metric_plot(
        metric_name=metric_name,
        series=series,
        anomalies=metric_anomalies,
        forecast_lower=forecast_lower,
        forecast_upper=forecast_upper,
        weekend_mask=weekend_mask,
    )
    return html
