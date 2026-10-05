"""
Load a trade journal into the standard schema used by analyze.py:

    entry_time, exit_time, instrument, direction, r_multiple, pnl_pct, session

Supported inputs
----------------
1. The standard schema itself (e.g. data/trade_log.csv).
2. A MetaTrader 5 "Positions" history export saved as CSV
   (columns: Time, Position, Symbol, Type, Volume, Price, S / L, T / P,
   Time, Price, Commission, Swap, Profit). R is derived from the stop:
       R = signed price move / |entry - stop|
   Trades without a stop get R = NaN and are reported, not guessed.
3. Any CSV plus a column mapping, e.g.
       load_journal("my.csv", mapping={"Open time": "entry_time", "Sym": "instrument",
                                       "R": "r_multiple", "PnL %": "pnl_pct"})

Session is derived from the entry hour in UTC:
    Asia 00-07, London 07-13, New York 13-21, Off-hours otherwise.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

REQUIRED = ["entry_time", "instrument", "r_multiple"]


def session_from_time(ts: pd.Series, utc_offset_hours: float = 0.0) -> pd.Series:
    hour = (pd.to_datetime(ts) - pd.Timedelta(hours=utc_offset_hours)).dt.hour
    return pd.Series(np.select([hour < 7, hour < 13, hour < 21], ["Asia", "London", "New York"],
                               default="Off-hours"), index=ts.index)


def from_mt5_positions(raw: pd.DataFrame, account_balance: float) -> pd.DataFrame:
    """Convert an MT5 positions export (pandas renames the duplicate
    Time/Price headers to Time.1/Price.1)."""
    cols = {c.strip(): c for c in raw.columns}
    get = lambda name: raw[cols[name]]  # noqa: E731
    entry, exit_ = get("Price").astype(float), get("Price.1").astype(float)
    stop = pd.to_numeric(get("S / L"), errors="coerce").replace(0, np.nan)
    side = np.where(get("Type").str.lower().str.startswith("buy"), 1.0, -1.0)
    profit = get("Profit").astype(float) + get("Commission").astype(float) + get("Swap").astype(float)
    return pd.DataFrame({
        "entry_time": pd.to_datetime(get("Time")),
        "exit_time": pd.to_datetime(get("Time.1")),
        "instrument": get("Symbol").str.upper().str.strip(),
        "direction": np.where(side > 0, "Long", "Short"),
        "r_multiple": side * (exit_ - entry) / (entry - stop).abs(),
        "pnl_pct": profit / account_balance * 100,
    })


def normalise(df: pd.DataFrame, utc_offset_hours: float = 0.0) -> pd.DataFrame:
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"journal is missing required columns: {missing}")
    df = df.copy()
    df["entry_time"] = pd.to_datetime(df["entry_time"])
    df["instrument"] = df["instrument"].astype(str).str.upper().str.strip()
    df["r_multiple"] = pd.to_numeric(df["r_multiple"], errors="coerce")
    if "pnl_pct" not in df.columns:
        df["pnl_pct"] = df["r_multiple"] * df.get("risk_pct", 1.0)
    if "session" not in df.columns:
        df["session"] = session_from_time(df["entry_time"], utc_offset_hours)
    n_bad = int(df["r_multiple"].isna().sum())
    if n_bad:
        print(f"[loader] {n_bad} trades have no R-multiple (e.g. no stop set); excluded from R stats")
    return df.sort_values("entry_time").reset_index(drop=True)


def load_journal(path, fmt: str = "standard", mapping: dict | None = None,
                 account_balance: float | None = None, utc_offset_hours: float = 0.0) -> pd.DataFrame:
    raw = pd.read_csv(path)
    if fmt == "mt5":
        if account_balance is None:
            raise ValueError("account_balance is required to express MT5 profit as %")
        raw = from_mt5_positions(raw, account_balance)
    elif mapping:
        raw = raw.rename(columns=mapping)
    return normalise(raw, utc_offset_hours)
