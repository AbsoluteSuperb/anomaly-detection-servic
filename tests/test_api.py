import pytest
from fastapi.testclient import TestClient

from app.main import app


@pytest.fixture(scope="module")
def client():
    with TestClient(app) as c:
        yield c


# ═══════════════════════════════════════════════════════════════════════════
#  GET /health
# ═══════════════════════════════════════════════════════════════════════════

def test_health_status(client):
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["data_loaded"] is True


def test_health_has_metrics_and_detectors(client):
    body = client.get("/api/v1/health").json()
    assert len(body["available_metrics"]) >= 5
    assert "revenue" in body["available_metrics"]
    assert "ensemble" in body["available_detectors"]
    assert body["total_days"] > 0
    assert "-" in body["date_range"]  # has a date range string


# ═══════════════════════════════════════════════════════════════════════════
#  GET /metrics
# ═══════════════════════════════════════════════════════════════════════════

def test_list_metrics_returns_all(client):
    resp = client.get("/api/v1/metrics")
    assert resp.status_code == 200
    metrics = resp.json()
    names = {m["metric_name"] for m in metrics}
    assert {"revenue", "orders", "avg_check", "unique_customers", "items_sold"} <= names


def test_metric_summary_fields(client):
    metrics = client.get("/api/v1/metrics").json()
    for m in metrics:
        expected = ["metric_name", "count", "mean", "std", "min", "max", "last_value", "last_date"]
        for field in expected:
            assert field in m, f"Missing field '{field}' in metric summary"


def test_get_single_metric(client):
    resp = client.get("/api/v1/metrics/revenue")
    assert resp.status_code == 200
    body = resp.json()
    assert body["metric_name"] == "revenue"
    assert len(body["dates"]) == len(body["values"])
    assert len(body["dates"]) > 0


def test_get_metric_date_filter(client):
    resp = client.get("/api/v1/metrics/revenue?start_date=2010-06-01&end_date=2010-06-30")
    assert resp.status_code == 200
    for d in resp.json()["dates"]:
        assert "2010-06-01" <= d <= "2010-06-30"


def test_get_unknown_metric_404(client):
    assert client.get("/api/v1/metrics/nonexistent").status_code == 404


# ═══════════════════════════════════════════════════════════════════════════
#  POST /detect
# ═══════════════════════════════════════════════════════════════════════════

def test_detect_zscore(client):
    resp = client.post("/api/v1/detect", json={"detector": "zscore", "metric_name": "revenue"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["total_anomalies"] == body["warnings"] + body["criticals"]
    assert body["detection_time_ms"] >= 0


def test_detect_iqr(client):
    resp = client.post("/api/v1/detect", json={"detector": "iqr", "metric_name": "revenue"})
    assert resp.status_code == 200
    assert resp.json()["total_anomalies"] >= 0


def test_detect_isolation_forest(client):
    resp = client.post("/api/v1/detect", json={"detector": "isolation_forest"})
    assert resp.status_code == 200
    for a in resp.json()["anomalies"]:
        assert a["metric"] == "multivariate"


def test_detect_ensemble(client):
    resp = client.post("/api/v1/detect", json={"detector": "ensemble"})
    assert resp.status_code == 200
    assert "anomalies" in resp.json()


def test_detect_all_metrics_at_once(client):
    """detector=zscore with no metric_name -> runs on all metrics."""
    resp = client.post("/api/v1/detect", json={"detector": "zscore"})
    assert resp.status_code == 200
    metrics = {a["metric"] for a in resp.json()["anomalies"]}
    # Should cover more than one metric
    assert len(metrics) >= 1


def test_detect_with_date_filter(client):
    resp = client.post(
        "/api/v1/detect",
        json={"detector": "zscore", "metric_name": "revenue", "start_date": "2011-01-01"},
    )
    assert resp.status_code == 200
    for a in resp.json()["anomalies"]:
        assert a["date"] >= "2011-01-01"


def test_detect_unknown_detector_400(client):
    resp = client.post("/api/v1/detect", json={"detector": "magic"})
    assert resp.status_code == 400


def test_detect_invalid_body_422(client):
    """Sending wrong types should trigger validation error."""
    # Pydantic v2 coerces int->str for 'detector', so send a truly invalid body
    resp = client.post("/api/v1/detect", json={"metric_name": ["not", "a", "string"]})
    assert resp.status_code == 422


def test_detect_anomaly_fields(client):
    """Each anomaly in the response must have all required fields."""
    resp = client.post("/api/v1/detect", json={"detector": "zscore", "metric_name": "revenue"})
    required = {
        "date", "metric", "value", "expected",
        "deviation_pct", "severity", "detector", "details",
    }
    for a in resp.json()["anomalies"]:
        assert required <= set(a.keys())
        assert a["severity"] in ("warning", "critical")


# ═══════════════════════════════════════════════════════════════════════════
#  GET /anomalies
# ═══════════════════════════════════════════════════════════════════════════

def test_anomalies_cache(client):
    # Populate cache
    client.post("/api/v1/detect", json={"detector": "zscore", "metric_name": "revenue"})
    resp = client.get("/api/v1/anomalies")
    assert resp.status_code == 200
    assert isinstance(resp.json(), list)


def test_anomalies_severity_filter(client):
    client.post("/api/v1/detect", json={"detector": "zscore", "metric_name": "revenue"})
    resp = client.get("/api/v1/anomalies?severity=critical")
    assert resp.status_code == 200
    for a in resp.json():
        assert a["severity"] == "critical"


def test_anomalies_metric_filter(client):
    client.post("/api/v1/detect", json={"detector": "zscore", "metric_name": "revenue"})
    resp = client.get("/api/v1/anomalies?metric=revenue")
    assert resp.status_code == 200
    for a in resp.json():
        assert a["metric"] == "revenue"


def test_anomalies_date_filter(client):
    client.post("/api/v1/detect", json={"detector": "zscore", "metric_name": "revenue"})
    resp = client.get("/api/v1/anomalies?start_date=2011-01-01&end_date=2011-06-30")
    assert resp.status_code == 200
    for a in resp.json():
        assert "2011-01-01" <= a["date"] <= "2011-06-30"


# ═══════════════════════════════════════════════════════════════════════════
#  GET /plot
# ═══════════════════════════════════════════════════════════════════════════

def test_plot_returns_html(client):
    resp = client.get("/api/v1/plot/revenue")
    assert resp.status_code == 200
    assert "plotly" in resp.text.lower()
    assert "<html" in resp.text.lower() or "<div" in resp.text.lower()


def test_plot_unknown_metric_404(client):
    assert client.get("/api/v1/plot/nonexistent").status_code == 404
