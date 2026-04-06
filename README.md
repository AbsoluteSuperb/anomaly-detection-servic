# Anomaly Detection Service

[![CI](https://github.com/<your-username>/anomaly-detection-service/actions/workflows/ci.yml/badge.svg)](https://github.com/<your-username>/anomaly-detection-service/actions)
![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

Production-ready microservice for detecting anomalies in e-commerce metrics. Built with FastAPI, Prophet, scikit-learn, and Streamlit. Analyzes daily revenue, orders, average check, and customer activity to catch spikes, drops, level shifts, and unusual patterns.

**Dataset:** [Online Retail II](https://archive.ics.uci.edu/dataset/502/online+retail+ii) (UCI) — 1M+ transactions, 2009–2011, UK-based online retailer.

## Quick Start

```bash
# 1. Clone and install
git clone https://github.com/<your-username>/anomaly-detection-service.git
cd anomaly-detection-service
make install

# 2. Place dataset at data/raw/online_retail_II.csv, then preprocess
make preprocess

# 3. Run the API
make run
# -> http://localhost:8000/docs

# 4. (Optional) Run the dashboard
make dashboard
# -> http://localhost:8501
```

### Docker

```bash
# Copy your .env file first (or create from example)
cp .env.example .env

# Build and run both API + dashboard
make docker
# API: http://localhost:8000  |  Dashboard: http://localhost:8501
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| `GET` | `/api/v1/health` | Service status, loaded data info, available detectors |
| `GET` | `/api/v1/metrics` | Summary statistics for all metrics |
| `GET` | `/api/v1/metrics/{name}` | Time series data (supports `?start_date=&end_date=`) |
| `POST` | `/api/v1/detect` | Run anomaly detection |
| `GET` | `/api/v1/anomalies` | Cached results (filters: `severity`, `metric`, `start_date`, `end_date`) |
| `GET` | `/api/v1/plot/{name}` | Interactive Plotly chart (HTML) |

### Examples

```bash
# Health check
curl http://localhost:8000/api/v1/health

# Run Z-Score detection on revenue
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"detector": "zscore", "metric_name": "revenue"}'

# Run full ensemble detection
curl -X POST http://localhost:8000/api/v1/detect \
  -H "Content-Type: application/json" \
  -d '{"detector": "ensemble"}'

# Get critical anomalies
curl "http://localhost:8000/api/v1/anomalies?severity=critical"
```

## Detection Methods

| Method | Type | Strengths | Weaknesses | Best for |
|--------|------|-----------|------------|----------|
| **Z-Score** | Statistical | Fast, interpretable | No seasonality awareness | Quick scans, simple metrics |
| **IQR** | Statistical | Robust to outliers in training data | No seasonality awareness | Skewed distributions |
| **Prophet** | ML (time series) | Handles trend + seasonality + holidays | Slow to fit (~5-10s) | Revenue, metrics with weekly/yearly patterns |
| **Isolation Forest** | ML (multivariate) | Catches joint anomalies invisible univariately | Less interpretable | Cross-metric analysis |
| **Ensemble** | Voting | Reduces false positives | Slower (runs all detectors) | Production alerting |

### Severity Levels

- **Warning:** mild deviation (Z: |z|>2, IQR: 1.5x, Prophet: outside 95% CI)
- **Critical:** strong deviation (Z: |z|>3, IQR: 3x, Prophet: outside 99% CI)
- **Ensemble:** 2 votes = warning, 3+ votes = critical

## Architecture

```mermaid
graph LR
    A[Raw CSV<br/>1M+ transactions] --> B[Cleaner<br/>remove returns,<br/>cancellations]
    B --> C[Aggregator<br/>daily metrics]
    C --> D[data/processed/<br/>daily_metrics.csv]
    D --> E[FastAPI]
    E --> F[Z-Score]
    E --> G[IQR]
    E --> H[Prophet]
    E --> I[Isolation Forest]
    F & G & H & I --> J[Ensemble<br/>voting]
    J --> K[API Response<br/>anomalies + severity]
    K --> L[Streamlit<br/>Dashboard]
    K --> M[Telegram<br/>Alerts]
```

## Evaluation on Synthetic Data

Synthetic dataset: 730 days with injected anomalies (point spikes, level shifts, trend changes, seasonal breaks). Ground truth allows computing precision/recall.

**Point anomaly detection** (the primary target for alerting):

| Detector | Precision | Recall | F1 |
|----------|-----------|--------|-----|
| Z-Score | 0.35 | 0.75 | 0.48 |
| IQR | 0.44 | 0.50 | 0.47 |
| Ensemble (2+ votes) | 0.50 | 0.50 | 0.50 |

- Z-Score has the highest recall (catches 75% of point anomalies) but more false positives.
- Ensemble improves precision to 0.50 via majority voting, reducing false alerts.
- Level shifts and trend changes are structural — detectors correctly adapt to them rather than flagging every day.

Generate your own synthetic data: `python -m scripts.generate_synthetic`

## Project Structure

```
anomaly-detection-service/
├── app/
│   ├── main.py                    # FastAPI app with lifespan startup
│   ├── config.py                  # All settings via pydantic-settings / env vars
│   ├── api/
│   │   ├── routes.py              # 6 API endpoints
│   │   └── dependencies.py        # Singleton data + fitted detectors
│   ├── detection/
│   │   ├── base.py                # BaseDetector ABC, Anomaly dataclass, Severity enum
│   │   ├── zscore_detector.py     # Rolling Z-score
│   │   ├── iqr_detector.py        # Rolling IQR
│   │   ├── prophet_detector.py    # Prophet with UK holidays
│   │   ├── isolation_forest.py    # Multivariate Isolation Forest
│   │   └── ensemble.py            # Majority voting
│   ├── preprocessing/
│   │   ├── cleaner.py             # Data cleaning pipeline
│   │   └── aggregator.py          # Transaction -> daily metrics
│   ├── models/schemas.py          # Pydantic request/response models
│   ├── alerting/telegram_alert.py # Telegram notifications
│   └── visualization/plots.py     # Plotly charts with anomaly overlay
├── dashboard/streamlit_app.py     # 4-page Streamlit dashboard
├── scripts/
│   ├── preprocess.py              # CLI: raw data -> daily metrics
│   └── generate_synthetic.py      # Synthetic data generator with ground truth
├── tests/                         # 55 tests (pytest)
├── notebooks/01_eda.ipynb         # Exploratory data analysis
├── Dockerfile & docker-compose.yml
├── .github/workflows/ci.yml       # GitHub Actions CI
└── pyproject.toml                 # Dependencies and tool config
```

## Tech Stack

- **API:** FastAPI, Pydantic, uvicorn
- **Data:** pandas, NumPy
- **Detection:** scikit-learn (Isolation Forest), Prophet, custom statistical detectors
- **Visualization:** Plotly, Streamlit
- **Testing:** pytest, httpx
- **CI/CD:** GitHub Actions, Docker, ruff (linting)

## Running Tests

```bash
make test
# or
pytest -v
```

## License

[MIT](LICENSE)
