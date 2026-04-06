import pandas as pd

from app.detection.base import Anomaly, BaseDetector, Severity


class ZScoreDetector(BaseDetector):
    """Rolling Z-score anomaly detector.

    Uses a rolling window (default 30 days) instead of global mean/std
    to avoid false positives caused by trend shifts.

    Thresholds:
        |z| > warning_threshold (default 2) -> WARNING
        |z| > critical_threshold (default 3) -> CRITICAL
    """

    def __init__(
        self,
        window: int = 30,
        warning_threshold: float = 2.0,
        critical_threshold: float = 3.0,
    ):
        self.window = window
        self.warning_threshold = warning_threshold
        self.critical_threshold = critical_threshold
        self._rolling_mean: pd.Series | None = None
        self._rolling_std: pd.Series | None = None

    def fit(self, series: pd.Series) -> "ZScoreDetector":
        self._rolling_mean = series.rolling(window=self.window, min_periods=1).mean()
        self._rolling_std = series.rolling(window=self.window, min_periods=1).std().fillna(1.0)
        # Avoid division by zero for constant stretches
        self._rolling_std = self._rolling_std.replace(0.0, 1.0)
        return self

    def detect(self, series: pd.Series) -> list[Anomaly]:
        if self._rolling_mean is None:
            self.fit(series)

        z_scores = (series - self._rolling_mean) / self._rolling_std
        anomalies: list[Anomaly] = []

        for date, z in z_scores.items():
            abs_z = abs(z)
            if abs_z < self.warning_threshold:
                continue

            severity = (
                Severity.CRITICAL if abs_z >= self.critical_threshold else Severity.WARNING
            )
            expected = self._rolling_mean[date]
            value = series[date]
            deviation_pct = ((value - expected) / expected * 100) if expected != 0 else 0.0

            anomalies.append(
                Anomaly(
                    date=str(date.date()) if hasattr(date, "date") else str(date),
                    metric=series.name or "unknown",
                    value=round(float(value), 2),
                    expected=round(float(expected), 2),
                    deviation_pct=round(float(deviation_pct), 1),
                    severity=severity,
                    detector="ZScoreDetector",
                    details=f"z-score={z:+.2f} (rolling {self.window}d window)",
                )
            )

        return anomalies
