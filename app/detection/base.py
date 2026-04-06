from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

import pandas as pd


class Severity(str, Enum):
    WARNING = "warning"
    CRITICAL = "critical"


@dataclass
class Anomaly:
    date: str
    metric: str
    value: float
    expected: float
    deviation_pct: float
    severity: Severity
    detector: str
    details: str


class BaseDetector(ABC):
    """Abstract base class for all anomaly detectors."""

    @abstractmethod
    def fit(self, series: pd.Series) -> "BaseDetector":
        """Fit the detector on historical data."""

    @abstractmethod
    def detect(self, series: pd.Series) -> list[Anomaly]:
        """Detect anomalies and return a list of Anomaly objects."""
