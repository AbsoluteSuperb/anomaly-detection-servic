from pydantic import BaseModel


class DetectionRequest(BaseModel):
    """Request body for /detect endpoint."""

    metric_name: str | None = None  # None = all metrics
    detector: str = "ensemble"  # zscore, iqr, prophet, isolation_forest, ensemble
    start_date: str | None = None
    end_date: str | None = None


class AnomalyResponse(BaseModel):
    """Single anomaly in API response."""

    date: str
    metric: str
    value: float
    expected: float
    deviation_pct: float
    severity: str  # "warning" | "critical"
    detector: str
    details: str


class DetectionResponse(BaseModel):
    """Response from /detect endpoint."""

    total_anomalies: int
    warnings: int
    criticals: int
    anomalies: list[AnomalyResponse]
    detection_time_ms: float


class MetricSummary(BaseModel):
    """Summary statistics for one metric."""

    metric_name: str
    count: int
    mean: float
    std: float
    min: float
    max: float
    last_value: float
    last_date: str


class MetricTimeSeries(BaseModel):
    """Time series data for a single metric."""

    metric_name: str
    dates: list[str]
    values: list[float]


class HealthResponse(BaseModel):
    """Response from /health endpoint."""

    status: str
    data_loaded: bool
    date_range: str
    total_days: int
    available_metrics: list[str]
    available_detectors: list[str]
