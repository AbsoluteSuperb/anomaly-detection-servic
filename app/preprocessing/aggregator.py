import pandas as pd


def aggregate_daily(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate cleaned transactions into daily e-commerce metrics.

    Metrics per day:
    - revenue: total Revenue
    - orders: unique Invoice count
    - avg_check: revenue / orders
    - unique_customers: unique Customer ID count
    - items_sold: total Quantity
    - avg_items_per_order: items_sold / orders
    - unique_products: unique StockCode count

    Also adds: day_of_week, month, is_weekend.
    Missing dates (store closed) are filled with 0 and flagged via `missing_day`.
    """
    df = df.copy()
    df["date"] = df["InvoiceDate"].dt.date

    daily = df.groupby("date").agg(
        revenue=("Revenue", "sum"),
        orders=("Invoice", "nunique"),
        unique_customers=("Customer ID", "nunique"),
        items_sold=("Quantity", "sum"),
        unique_products=("StockCode", "nunique"),
    )

    daily.index = pd.to_datetime(daily.index)
    daily = daily.sort_index()

    # Derived metrics
    daily["avg_check"] = daily["revenue"] / daily["orders"]
    daily["avg_items_per_order"] = daily["items_sold"] / daily["orders"]

    # Fill missing dates with 0
    full_range = pd.date_range(daily.index.min(), daily.index.max(), freq="D")
    daily = daily.reindex(full_range)
    daily.index.name = "date"
    daily["missing_day"] = daily["revenue"].isna()
    daily = daily.fillna(0)

    # Date features
    daily["day_of_week"] = daily.index.dayofweek  # 0=Mon, 6=Sun
    daily["month"] = daily.index.month
    daily["is_weekend"] = daily["day_of_week"].isin([5, 6])

    # Reorder columns
    daily = daily[
        [
            "revenue",
            "orders",
            "avg_check",
            "unique_customers",
            "items_sold",
            "avg_items_per_order",
            "unique_products",
            "day_of_week",
            "month",
            "is_weekend",
            "missing_day",
        ]
    ]

    return daily
