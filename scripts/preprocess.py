"""CLI script: aggregate raw transactions into daily metrics."""

import argparse
from pathlib import Path

from app.preprocessing.aggregator import aggregate_daily
from app.preprocessing.cleaner import load_and_clean


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preprocess raw Online Retail II data")
    parser.add_argument(
        "--input",
        default="data/raw/online_retail_II.csv",
        help="Path to raw CSV (default: data/raw/online_retail_II.csv)",
    )
    parser.add_argument(
        "--output",
        default="data/processed/daily_metrics.csv",
        help="Path to save aggregated CSV (default: data/processed/daily_metrics.csv)",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    print(f"Loading raw data from {args.input} ...")
    df = load_and_clean(args.input)
    print(f"  Cleaned: {len(df):,} rows")

    daily = aggregate_daily(df)
    d_min, d_max = daily.index.min().date(), daily.index.max().date()
    print(f"  Aggregated: {len(daily)} days ({d_min} - {d_max})")

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    daily.to_csv(out)
    print(f"  Saved to {out}")


if __name__ == "__main__":
    main()
