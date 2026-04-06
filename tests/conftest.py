import numpy as np
import pandas as pd
import pytest

from scripts.generate_synthetic import generate_synthetic


@pytest.fixture
def sample_series() -> pd.Series:
    """Daily revenue-like series with one obvious spike at index 50."""
    rng = np.random.default_rng(0)
    dates = pd.date_range("2023-01-01", periods=100, freq="D")
    values = rng.normal(10_000, 500, 100)
    values[50] = 30_000  # injected anomaly
    return pd.Series(values, index=dates, name="revenue")


@pytest.fixture
def normal_series() -> pd.Series:
    """Smooth series with NO anomalies (stable mean, low noise)."""
    rng = np.random.default_rng(7)
    dates = pd.date_range("2023-01-01", periods=200, freq="D")
    values = rng.normal(10_000, 300, 200)
    return pd.Series(values, index=dates, name="revenue")


@pytest.fixture
def sample_dataframe(sample_series: pd.Series) -> pd.DataFrame:
    """Daily metrics DataFrame with an obvious multivariate anomaly at index 50."""
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "revenue": sample_series.values,
            "orders": rng.normal(50, 5, 100).astype(int),
            "avg_check": sample_series.values / 50,
            "unique_customers": rng.normal(40, 4, 100).astype(int),
        },
        index=sample_series.index,
    )
    return df


@pytest.fixture
def synthetic_data() -> pd.DataFrame:
    """730-day synthetic dataset with ground-truth anomaly labels."""
    return generate_synthetic(days=730, seed=42)


@pytest.fixture
def raw_transactions() -> pd.DataFrame:
    """Small fake raw transaction DataFrame for cleaner/aggregator tests."""
    return pd.DataFrame(
        {
            "Invoice": [
                "INV001", "INV001", "INV002", "C00099",  # C00099 = cancelled
                "INV003", "INV004", "INV005",
            ],
            "StockCode": ["A001", "A002", "A003", "A001", "POST", "A004", "A005"],
            "Description": ["Widget"] * 7,
            "Quantity": [2, 3, 1, -5, 1, 4, 2],
            "InvoiceDate": pd.to_datetime([
                "2023-01-02", "2023-01-02", "2023-01-02", "2023-01-02",
                "2023-01-02", "2023-01-04", "2023-01-04",
            ]),
            "Price": [10.0, 20.0, 15.0, 10.0, 5.0, 25.0, 30.0],
            "Customer ID": [1.0, 2.0, 3.0, 1.0, 2.0, 4.0, np.nan],
            "Country": ["United Kingdom"] * 7,
        }
    )
