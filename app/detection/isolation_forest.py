import pandas as pd
from sklearn.ensemble import IsolationForest as SklearnIsolationForest
from sklearn.preprocessing import StandardScaler

from app.detection.base import Anomaly, BaseDetector, Severity

DEFAULT_FEATURES = ["revenue", "orders", "avg_check", "unique_customers"]


class IsolationForestDetector(BaseDetector):
    """Multivariate anomaly detector using Isolation Forest.

    Feeds multiple metrics simultaneously to catch anomalies that look
    normal individually but are unusual in combination
    (e.g., normal revenue but 2x orders with 0.5x avg_check).

    Severity is based on anomaly_score:
        score < warning_score (default -0.15) -> WARNING
        score < critical_score (default -0.25) -> CRITICAL
    """

    def __init__(
        self,
        features: list[str] | None = None,
        contamination: float = 0.05,
        warning_score: float = -0.15,
        critical_score: float = -0.25,
        random_state: int = 42,
    ):
        self.features = features or DEFAULT_FEATURES
        self.contamination = contamination
        self.warning_score = warning_score
        self.critical_score = critical_score
        self.random_state = random_state
        self._model: SklearnIsolationForest | None = None
        self._scaler: StandardScaler | None = None

    def fit(self, df: pd.DataFrame) -> "IsolationForestDetector":
        """Fit on a DataFrame with columns matching self.features."""
        X = df[self.features].values
        self._scaler = StandardScaler()
        X_scaled = self._scaler.fit_transform(X)

        self._model = SklearnIsolationForest(
            contamination=self.contamination,
            random_state=self.random_state,
            n_estimators=200,
        )
        self._model.fit(X_scaled)
        return self

    def detect(self, df: pd.DataFrame) -> list[Anomaly]:
        """Detect anomalies in a DataFrame. Returns Anomaly per anomalous row."""
        if self._model is None:
            self.fit(df)

        X = df[self.features].values
        X_scaled = self._scaler.transform(X)

        predictions = self._model.predict(X_scaled)
        scores = self._model.decision_function(X_scaled)

        anomalies: list[Anomaly] = []

        for i, (date, row) in enumerate(df.iterrows()):
            if predictions[i] != -1:
                continue

            score = scores[i]
            if score > self.warning_score:
                continue

            severity = Severity.CRITICAL if score < self.critical_score else Severity.WARNING

            feature_vals = ", ".join(
                f"{f}={row[f]:,.0f}" for f in self.features
            )

            anomalies.append(
                Anomaly(
                    date=str(date.date()) if hasattr(date, "date") else str(date),
                    metric="multivariate",
                    value=round(float(row["revenue"]), 2),
                    expected=0.0,  # no single expected value for multivariate
                    deviation_pct=0.0,
                    severity=severity,
                    detector="IsolationForestDetector",
                    details=f"anomaly_score={score:.3f}, features: {feature_vals}",
                )
            )

        return anomalies
