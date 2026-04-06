import numpy as np
import pandas as pd

from app.preprocessing.aggregator import aggregate_daily
from app.preprocessing.cleaner import NON_PRODUCT_CODES

# ── Cleaner logic (tested via fixture) ──────────────────────────────────────

def test_cancellations_removed(raw_transactions: pd.DataFrame):
    """Invoices starting with 'C' must be dropped."""
    import os
    import tempfile

    from app.preprocessing.cleaner import load_and_clean

    # Write fixture to a temp CSV so we can use load_and_clean
    path = os.path.join(tempfile.gettempdir(), "test_raw.csv")
    raw_transactions.to_csv(path, index=False)
    df = load_and_clean(path)

    assert not df["Invoice"].astype(str).str.startswith("C").any()


def test_negative_quantity_removed(raw_transactions: pd.DataFrame):
    """Rows with Quantity <= 0 must be dropped."""
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "test_raw.csv")
    raw_transactions.to_csv(path, index=False)

    from app.preprocessing.cleaner import load_and_clean
    df = load_and_clean(path)

    assert (df["Quantity"] > 0).all()


def test_missing_customer_id_removed(raw_transactions: pd.DataFrame):
    """Rows with NaN Customer ID must be dropped."""
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "test_raw.csv")
    raw_transactions.to_csv(path, index=False)

    from app.preprocessing.cleaner import load_and_clean
    df = load_and_clean(path)

    assert df["Customer ID"].notna().all()


def test_non_product_codes_removed(raw_transactions: pd.DataFrame):
    """Service stock codes like POST must be dropped."""
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "test_raw.csv")
    raw_transactions.to_csv(path, index=False)

    from app.preprocessing.cleaner import load_and_clean
    df = load_and_clean(path)

    remaining_codes = set(df["StockCode"].astype(str).str.upper())
    assert remaining_codes.isdisjoint(NON_PRODUCT_CODES)


def test_revenue_column_added(raw_transactions: pd.DataFrame):
    """Revenue = Quantity * Price must be present after cleaning."""
    import os
    import tempfile

    path = os.path.join(tempfile.gettempdir(), "test_raw.csv")
    raw_transactions.to_csv(path, index=False)

    from app.preprocessing.cleaner import load_and_clean
    df = load_and_clean(path)

    assert "Revenue" in df.columns
    assert np.allclose(df["Revenue"], df["Quantity"] * df["Price"])


# ── Aggregator ──────────────────────────────────────────────────────────────

def _make_clean_transactions() -> pd.DataFrame:
    """Minimal clean transactions for aggregator tests."""
    return pd.DataFrame(
        {
            "InvoiceDate": pd.to_datetime(["2023-01-02", "2023-01-02", "2023-01-04"]),
            "Invoice": ["INV001", "INV001", "INV002"],
            "Revenue": [100.0, 200.0, 150.0],
            "Customer ID": [1, 2, 3],
            "Quantity": [2, 3, 1],
            "StockCode": ["A001", "A002", "A003"],
        }
    )


def test_aggregate_daily_columns():
    result = aggregate_daily(_make_clean_transactions())
    expected_cols = [
        "revenue", "orders", "avg_check", "unique_customers",
        "items_sold", "avg_items_per_order", "unique_products",
        "day_of_week", "month", "is_weekend", "missing_day",
    ]
    assert list(result.columns) == expected_cols


def test_aggregate_daily_revenue_sum():
    result = aggregate_daily(_make_clean_transactions())
    # Jan 2: 100 + 200 = 300
    assert result.loc["2023-01-02", "revenue"] == 300.0


def test_aggregate_daily_order_count():
    result = aggregate_daily(_make_clean_transactions())
    # Jan 2: 1 unique invoice
    assert result.loc["2023-01-02", "orders"] == 1


def test_aggregate_daily_avg_check():
    result = aggregate_daily(_make_clean_transactions())
    # avg_check = revenue / orders = 300 / 1
    assert result.loc["2023-01-02", "avg_check"] == 300.0


def test_aggregate_daily_unique_customers():
    result = aggregate_daily(_make_clean_transactions())
    assert result.loc["2023-01-02", "unique_customers"] == 2


def test_aggregate_daily_items_sold():
    result = aggregate_daily(_make_clean_transactions())
    assert result.loc["2023-01-02", "items_sold"] == 5  # 2 + 3


def test_missing_dates_filled():
    result = aggregate_daily(_make_clean_transactions())
    # Jan 2 -> Jan 4, so Jan 3 is missing
    assert len(result) == 3
    assert result.loc["2023-01-03", "missing_day"]
    assert result.loc["2023-01-03", "revenue"] == 0


def test_date_features():
    result = aggregate_daily(_make_clean_transactions())
    # Jan 2 2023 = Monday (0)
    assert result.loc["2023-01-02", "day_of_week"] == 0
    assert not result.loc["2023-01-02", "is_weekend"]
    assert result.loc["2023-01-02", "month"] == 1
