"""
Causal feature engineering: every feature at cycle t uses only readings
from cycles <= t of the same machine, so it could be computed live.

Feature groups
--------------
raw        : current vibration, temperature, load
rolling    : rolling mean / std / least-squares slope over `window` cycles
baseline   : deviation from the machine's own commissioning baseline
             (median of its first `baseline_cycles` readings), in units of
             that baseline's spread -- removes machine-to-machine differences
load-norm  : vibration and temperature residuals after removing the
             fleet-level linear effect of load (fitted on training data only)
"""
from __future__ import annotations

import numpy as np
import pandas as pd

SENSORS = ["vibration_g", "temperature_c"]
RAW = ["vibration_g", "temperature_c", "load"]


def _rolling_slope(x: pd.Series, window: int) -> pd.Series:
    t = np.arange(window) - (window - 1) / 2
    denom = (t**2).sum()
    return x.rolling(window, min_periods=window).apply(lambda y: (t * y).sum() / denom, raw=True)


def add_features(df: pd.DataFrame, window: int = 10, baseline_cycles: int = 30,
                 load_coef: dict | None = None) -> tuple[pd.DataFrame, dict]:
    """Returns (featured_df, load_coef). Pass the training fold's load_coef
    when featurising a test fold so nothing is fitted on test data."""
    df = df.sort_values(["machine_id", "cycle"]).copy()
    g = df.groupby("machine_id")
    for c in SENSORS:                                # forward-fill dropped packets
        df[c] = g[c].ffill().bfill() if df[c].isna().any() else df[c]

    if load_coef is None:
        load_coef = {c: np.polyfit(df["load"], df[c], 1).tolist() for c in SENSORS}
    for c in SENSORS:
        slope, _ = load_coef[c]
        df[f"{c}_load_adj"] = df[c] - slope * (df["load"] - 0.7)

    g = df.groupby("machine_id")
    for c in SENSORS:
        adj = f"{c}_load_adj"
        df[f"{c}_roll_mean"] = g[adj].transform(lambda x: x.rolling(window, min_periods=1).mean())
        df[f"{c}_roll_std"] = g[adj].transform(lambda x: x.rolling(window, min_periods=2).std()).fillna(0)
        df[f"{c}_slope"] = g[adj].transform(lambda x: _rolling_slope(x, window)).fillna(0)
        base_med = g[adj].transform(lambda x: x.iloc[:baseline_cycles].median())
        base_sd = g[adj].transform(lambda x: x.iloc[:baseline_cycles].std()).clip(lower=1e-6)
        df[f"{c}_baseline_z"] = (df[f"{c}_roll_mean"] - base_med) / base_sd
        df[f"{c}_baseline_ratio"] = df[f"{c}_roll_mean"] / base_med
    return df, load_coef


FEATURE_SETS = {
    "raw": RAW,
    "engineered": RAW + [f"{c}_{s}" for c in SENSORS for s in
                         ("load_adj", "roll_mean", "roll_std", "slope", "baseline_z", "baseline_ratio")],
}
