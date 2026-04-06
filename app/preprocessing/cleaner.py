import pandas as pd

# Service/non-product stock codes to exclude
NON_PRODUCT_CODES = {
    "POST",
    "DOT",
    "BANK CHARGES",
    "PADS",
    "M",
    "D",
    "S",
    "CRUK",
    "C2",
    "AMAZONFEE",
    "B",
}


def load_and_clean(path: str) -> pd.DataFrame:
    """Load raw Online Retail II data and apply cleaning rules.

    Steps:
    1. Parse InvoiceDate as datetime
    2. Remove cancelled invoices (starting with "C")
    3. Remove rows with Quantity <= 0 or Price <= 0
    4. Remove rows with missing Customer ID
    5. Remove non-product StockCodes (POST, DOT, BANK CHARGES, etc.)
    6. Add Revenue = Quantity * Price
    """
    df = pd.read_csv(path, parse_dates=["InvoiceDate"])

    rows_before = len(df)

    # 1. Remove cancelled orders (Invoice starts with "C")
    df = df[~df["Invoice"].astype(str).str.startswith("C")]

    # 2. Keep only positive quantities and prices
    df = df[(df["Quantity"] > 0) & (df["Price"] > 0)]

    # 3. Drop rows without Customer ID
    df = df.dropna(subset=["Customer ID"])

    # 4. Remove non-product stock codes
    df = df[~df["StockCode"].astype(str).str.upper().isin(NON_PRODUCT_CODES)]

    # 5. Add Revenue column
    df["Revenue"] = df["Quantity"] * df["Price"]

    rows_after = len(df)
    removed = rows_before - rows_after
    print(f"  Cleaning: {rows_before:,} -> {rows_after:,} rows ({removed:,} removed)")

    return df
