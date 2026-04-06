"""Generate synthetic daily metrics with injected anomalies for testing detectors.

Anomaly types:
  1. Point anomaly     - single day spike/dip
  2. Level shift       - sustained shift for N days
  3. Trend change      - slope changes mid-series
  4. Seasonal break    - weekend pattern inverts for N days

Ground truth is saved alongside the data so precision/recall/F1 can be computed.
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def _base_signal(days: int, rng: np.random.Generator) -> np.ndarray:
    """Trend + weekly seasonality + noise."""
    t = np.arange(days)
    trend = np.linspace(10_000, 15_000, days)
    weekly = 2_000 * np.sin(2 * np.pi * t / 7)
    noise = rng.normal(0, 500, days)
    return trend + weekly + noise


def _inject_point_anomalies(
    values: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n: int = 8,
) -> None:
    """Single-day spikes or dips."""
    days = len(values)
    idx = rng.choice(days, size=n, replace=False)
    for i in idx:
        direction = rng.choice([-1, 1])
        magnitude = rng.uniform(2.0, 4.0)
        values[i] += direction * magnitude * abs(values[i]) * 0.3
        labels[i] = 1


def _inject_level_shift(
    values: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
    n_shifts: int = 2,
    shift_len: int = 14,
) -> None:
    """Sustained level shift for `shift_len` consecutive days."""
    days = len(values)
    for _ in range(n_shifts):
        start = rng.integers(30, days - shift_len - 30)
        shift_amount = rng.uniform(0.4, 0.8) * values[start]
        direction = rng.choice([-1, 1])
        values[start : start + shift_len] += direction * shift_amount
        labels[start : start + shift_len] = 1


def _inject_trend_change(
    values: np.ndarray,
    labels: np.ndarray,
    rng: np.random.Generator,
) -> None:
    """One abrupt slope change mid-series."""
    days = len(values)
    change_point = rng.integers(days // 3, 2 * days // 3)
    extra_slope = rng.uniform(20, 50) * rng.choice([-1, 1])
    for i in range(change_point, days):
        values[i] += extra_slope * (i - change_point)
        labels[i] = 1


def _inject_seasonal_break(
    values: np.ndarray,
    labels: np.ndarray,
    dates: pd.DatetimeIndex,
    rng: np.random.Generator,
    break_len: int = 21,
) -> None:
    """Invert the weekend dip pattern for `break_len` days."""
    days = len(values)
    start = rng.integers(30, days - break_len - 30)
    for i in range(start, start + break_len):
        dow = dates[i].dayofweek
        if dow >= 5:  # weekend: normally low -> make high
            values[i] += abs(values[i]) * 0.5
            labels[i] = 1
        elif dow == 0:  # Monday: add anomalous spike
            values[i] += abs(values[i]) * 0.3
            labels[i] = 1


def generate_synthetic(days: int = 730, seed: int = 42) -> pd.DataFrame:
    """Generate synthetic daily revenue with known anomalies.

    Returns DataFrame with columns:
        revenue, orders, avg_check, anomaly_type, is_anomaly
    """
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2022-01-01", periods=days, freq="D")

    revenue = _base_signal(days, rng)
    labels = np.zeros(days, dtype=int)
    anom_type = np.full(days, "normal", dtype=object)

    # Weekend dips (normal behaviour)
    for i, d in enumerate(dates):
        if d.dayofweek >= 5:
            revenue[i] *= 0.6

    # Inject anomalies
    _inject_point_anomalies(revenue, labels, rng, n=8)
    point_mask = (labels == 1)
    anom_type[point_mask] = "point"

    prev_labels = labels.copy()
    _inject_level_shift(revenue, labels, rng, n_shifts=2, shift_len=14)
    anom_type[(labels == 1) & (prev_labels == 0)] = "level_shift"

    prev_labels = labels.copy()
    _inject_trend_change(revenue, labels, rng)
    anom_type[(labels == 1) & (prev_labels == 0)] = "trend_change"

    prev_labels = labels.copy()
    _inject_seasonal_break(revenue, labels, dates, rng, break_len=21)
    anom_type[(labels == 1) & (prev_labels == 0)] = "seasonal_break"

    # Derive other metrics from revenue
    orders = np.maximum(1, (revenue / rng.uniform(40, 60, days)).astype(int))
    avg_check = revenue / orders

    df = pd.DataFrame(
        {
            "revenue": np.round(revenue, 2),
            "orders": orders,
            "avg_check": np.round(avg_check, 2),
            "anomaly_type": anom_type,
            "is_anomaly": labels.astype(bool),
        },
        index=dates,
    )
    df.index.name = "date"
    return df


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic data with injected anomalies")
    parser.add_argument("--days", type=int, default=730, help="Number of days (default: 730)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--output",
        default="data/processed/synthetic_daily.csv",
        help="Output path (default: data/processed/synthetic_daily.csv)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    df = generate_synthetic(days=args.days, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out)

    total_anom = df["is_anomaly"].sum()
    by_type = df[df["is_anomaly"]]["anomaly_type"].value_counts()
    print(f"Generated {len(df)} days, {total_anom} anomaly days ({total_anom/len(df)*100:.1f}%)")
    print(f"Breakdown: {by_type.to_dict()}")
    print(f"Saved to {out}")
