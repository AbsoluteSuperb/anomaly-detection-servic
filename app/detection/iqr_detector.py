import pandas as pd

from app.detection.base import Anomaly, BaseDetector, Severity


class IQRDetector(BaseDetector):
    """Rolling IQR anomaly detector.

    Uses a rolling window (default 30 days) to compute Q1/Q3.
    More robust to outliers in the training window than Z-score.

    Thresholds:
        Outside Q1 - 1.5*IQR / Q3 + 1.5*IQR -> WARNING
        Outside Q1 - 3.0*IQR / Q3 + 3.0*IQR -> CRITICAL
    """

    def __init__(
        self,
        window: int = 30,
        warning_multiplier: float = 1.5,
        critical_multiplier: float = 3.0,
    ):
        self.window = window
        self.warning_multiplier = warning_multiplier
        self.critical_multiplier = critical_multiplier
        self._q1: pd.Series | None = None
        self._q3: pd.Series | None = None
        self._iqr: pd.Series | None = None

    def fit(self, series: pd.Series) -> "IQRDetector":
        rolling = series.rolling(window=self.window, min_periods=1)
        self._q1 = rolling.quantile(0.25)
        self._q3 = rolling.quantile(0.75)
        self._iqr = self._q3 - self._q1
        return self

    def detect(self, series: pd.Series) -> list[Anomaly]:
        if self._q1 is None:
            self.fit(series)

        warn_lo = self._q1 - self.warning_multiplier * self._iqr
        warn_hi = self._q3 + self.warning_multiplier * self._iqr
        crit_lo = self._q1 - self.critical_multiplier * self._iqr
        crit_hi = self._q3 + self.critical_multiplier * self._iqr

        median = (self._q1 + self._q3) / 2  # approximate expected value

        anomalies: list[Anomaly] = []

        for date in series.index:
            value = series[date]
            w_lo, w_hi = warn_lo[date], warn_hi[date]
            c_lo, c_hi = crit_lo[date], crit_hi[date]

            if w_lo <= value <= w_hi:
                continue

            severity = Severity.CRITICAL if (value < c_lo or value > c_hi) else Severity.WARNING
            expected = float(median[date])
            deviation_pct = ((value - expected) / expected * 100) if expected != 0 else 0.0

            anomalies.append(
                Anomaly(
                    date=str(date.date()) if hasattr(date, "date") else str(date),
                    metric=series.name or "unknown",
                    value=round(float(value), 2),
                    expected=round(float(expected), 2),
                    deviation_pct=round(float(deviation_pct), 1),
                    severity=severity,
                    detector="IQRDetector",
                    details=(
                        f"IQR bounds [{w_lo:,.0f} .. {w_hi:,.0f}] "
                        f"(rolling {self.window}d window)"
                    ),
                )
            )

        return anomalies
