import pandas as pd
from prophet import Prophet

from app.detection.base import Anomaly, BaseDetector, Severity

# UK public holidays relevant to the dataset period (2009-2011)
UK_HOLIDAYS = pd.DataFrame(
    {
        "holiday": [
            "christmas",
            "christmas",
            "christmas",
            "boxing_day",
            "boxing_day",
            "boxing_day",
            "new_year",
            "new_year",
            "new_year",
            "easter_monday",
            "easter_monday",
            "easter_monday",
            "early_may_bank",
            "early_may_bank",
            "early_may_bank",
            "spring_bank",
            "spring_bank",
            "spring_bank",
            "summer_bank",
            "summer_bank",
            "summer_bank",
        ],
        "ds": pd.to_datetime(
            [
                "2009-12-25",
                "2010-12-25",
                "2011-12-25",
                "2009-12-26",
                "2010-12-26",
                "2011-12-26",
                "2010-01-01",
                "2011-01-01",
                "2012-01-01",
                "2010-04-05",
                "2011-04-25",
                "2012-04-09",
                "2010-05-03",
                "2011-05-02",
                "2012-05-07",
                "2010-05-31",
                "2011-05-30",
                "2012-06-04",
                "2010-08-30",
                "2011-08-29",
                "2012-08-27",
            ]
        ),
        "lower_window": 0,
        "upper_window": 1,
    }
)


class ProphetDetector(BaseDetector):
    """Prophet-based anomaly detector with trend + seasonality awareness.

    Uses Prophet's uncertainty intervals for detection.
    Fits weekly and yearly seasonality automatically.
    Includes UK bank holidays.

    Thresholds:
        Outside 95% interval -> WARNING
        Outside 99% interval -> CRITICAL
    """

    def __init__(
        self,
        warning_interval: float = 0.95,
        critical_interval: float = 0.99,
    ):
        self.warning_interval = warning_interval
        self.critical_interval = critical_interval
        self._model_warn: Prophet | None = None
        self._model_crit: Prophet | None = None
        self._forecast_warn: pd.DataFrame | None = None
        self._forecast_crit: pd.DataFrame | None = None

    def fit(self, series: pd.Series) -> "ProphetDetector":
        df = pd.DataFrame({"ds": series.index, "y": series.values})

        self._model_warn = Prophet(
            interval_width=self.warning_interval,
            holidays=UK_HOLIDAYS,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        self._model_warn.fit(df)
        self._forecast_warn = self._model_warn.predict(df)

        self._model_crit = Prophet(
            interval_width=self.critical_interval,
            holidays=UK_HOLIDAYS,
            weekly_seasonality=True,
            yearly_seasonality=True,
            daily_seasonality=False,
        )
        self._model_crit.fit(df)
        self._forecast_crit = self._model_crit.predict(df)

        return self

    def detect(self, series: pd.Series) -> list[Anomaly]:
        if self._forecast_warn is None:
            self.fit(series)

        fw = self._forecast_warn.set_index("ds")
        fc = self._forecast_crit.set_index("ds")

        anomalies: list[Anomaly] = []

        for date, value in series.items():
            if date not in fw.index:
                continue

            yhat = float(fw.loc[date, "yhat"])
            warn_lo = float(fw.loc[date, "yhat_lower"])
            warn_hi = float(fw.loc[date, "yhat_upper"])
            crit_lo = float(fc.loc[date, "yhat_lower"])
            crit_hi = float(fc.loc[date, "yhat_upper"])

            if warn_lo <= value <= warn_hi:
                continue

            is_critical = value < crit_lo or value > crit_hi
            severity = Severity.CRITICAL if is_critical else Severity.WARNING
            deviation_pct = ((value - yhat) / yhat * 100) if yhat != 0 else 0.0

            anomalies.append(
                Anomaly(
                    date=str(date.date()) if hasattr(date, "date") else str(date),
                    metric=series.name or "unknown",
                    value=round(float(value), 2),
                    expected=round(yhat, 2),
                    deviation_pct=round(deviation_pct, 1),
                    severity=severity,
                    detector="ProphetDetector",
                    details=(
                        f"Prophet forecast={yhat:,.0f}, "
                        f"{self.warning_interval:.0%} CI=[{warn_lo:,.0f}..{warn_hi:,.0f}]"
                    ),
                )
            )

        return anomalies
