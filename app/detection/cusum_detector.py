import pandas as pd

from app.detection.base import Anomaly, BaseDetector, Severity


class CUSUMDetector(BaseDetector):
    """Cumulative Sum (CUSUM) detector for level shifts and trend changes.

    Tracks cumulative deviation from the mean. When the cumulative sum
    exceeds a threshold, it signals that the process has shifted.

    Unlike Z-Score/IQR which catch single-day spikes, CUSUM catches
    sustained changes in the mean level of the series.

    Parameters
    ----------
    drift : float
        Allowable drift (slack) before accumulating. Higher = less sensitive.
        Default 0.5 * std of the series.
    warning_threshold : float
        CUSUM value to trigger WARNING (default 4 * std).
    critical_threshold : float
        CUSUM value to trigger CRITICAL (default 6 * std).
    """

    def __init__(
        self,
        drift_factor: float = 0.5,
        warning_factor: float = 4.0,
        critical_factor: float = 6.0,
    ):
        self.drift_factor = drift_factor
        self.warning_factor = warning_factor
        self.critical_factor = critical_factor
        self._mean: float = 0.0
        self._std: float = 1.0

    def fit(self, series: pd.Series) -> "CUSUMDetector":
        self._mean = series.mean()
        self._std = series.std()
        if self._std == 0:
            self._std = 1.0
        return self

    def detect(self, series: pd.Series) -> list[Anomaly]:
        drift = self.drift_factor * self._std
        warn_h = self.warning_factor * self._std
        crit_h = self.critical_factor * self._std

        # CUSUM tracks positive and negative shifts separately
        s_pos = 0.0  # detects upward shift
        s_neg = 0.0  # detects downward shift
        anomalies: list[Anomaly] = []

        for date, value in series.items():
            x = value - self._mean

            s_pos = max(0, s_pos + x - drift)
            s_neg = max(0, s_neg - x - drift)

            cusum_val = max(s_pos, s_neg)

            if cusum_val < warn_h:
                continue

            severity = Severity.CRITICAL if cusum_val >= crit_h else Severity.WARNING
            deviation_pct = (
                ((value - self._mean) / self._mean * 100) if self._mean != 0 else 0.0
            )

            direction = "upward" if s_pos >= s_neg else "downward"
            anomalies.append(
                Anomaly(
                    date=str(date.date()) if hasattr(date, "date") else str(date),
                    metric=series.name or "unknown",
                    value=round(float(value), 2),
                    expected=round(self._mean, 2),
                    deviation_pct=round(float(deviation_pct), 1),
                    severity=severity,
                    detector="CUSUMDetector",
                    details=f"CUSUM={cusum_val:,.0f} ({direction} shift)",
                )
            )

            # Reset after detection to catch next shift
            if cusum_val >= crit_h:
                s_pos = 0.0
                s_neg = 0.0

        return anomalies
