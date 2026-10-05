"""
Generates synthetic price series for two assets: one pair that is
genuinely cointegrated (e.g. two stocks in the same sector sharing a
common stochastic trend + a mean-reverting spread), and one pair that
is NOT cointegrated (two independent random walks), so the pipeline
can be shown correctly rejecting a non-tradeable pair.

Real-world use: swap `load_cointegrated_pair()` / `load_uncorrelated_pair()`
for a loader that pulls actual OHLC data (e.g. via a market data API).
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _ou_process(n: int, theta: float, mu: float, sigma: float, x0: float, seed: int) -> np.ndarray:
    """Simulate an Ornstein-Uhlenbeck (mean-reverting) process."""
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    x[0] = x0
    for t in range(1, n):
        x[t] = x[t - 1] + theta * (mu - x[t - 1]) + sigma * rng.standard_normal()
    return x


def load_cointegrated_pair(n: int = 750, seed: int = 11) -> pd.DataFrame:
    """
    Asset A: arithmetic random walk with drift (the common stochastic trend).
    Asset B: same common trend + a mean-reverting spread (OU process).
    => A and B are cointegrated with cointegrating vector approx (1, -1).
    """
    rng = np.random.default_rng(seed)
    common_trend = np.cumsum(rng.normal(0.05, 1.0, n))
    spread = _ou_process(n, theta=0.08, mu=0.0, sigma=0.6, x0=0.0, seed=seed + 1)

    price_a = 100 + common_trend
    price_b = 100 + common_trend + spread  # beta ~ 1

    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"date": dates, "asset_a": price_a, "asset_b": price_b}).set_index("date")


def load_uncorrelated_pair(n: int = 750, seed: int = 9) -> pd.DataFrame:
    """Two independent random walks — should NOT be flagged as cointegrated."""
    rng = np.random.default_rng(seed)
    price_a = 100 + np.cumsum(rng.normal(0.02, 1.0, n))
    price_b = 50 + np.cumsum(rng.normal(0.03, 1.2, n))
    dates = pd.bdate_range("2023-01-02", periods=n)
    return pd.DataFrame({"date": dates, "asset_a": price_a, "asset_b": price_b}).set_index("date")


if __name__ == "__main__":
    coint_df = load_cointegrated_pair()
    uncorr_df = load_uncorrelated_pair()
    from pathlib import Path
    out = Path(__file__).resolve().parents[1] / "data"
    out.mkdir(exist_ok=True)
    coint_df.to_csv(out / "cointegrated_pair.csv")
    uncorr_df.to_csv(out / "uncorrelated_pair.csv")
    print("Saved sample data to ../data/")
    print(coint_df.head())
