"""
Synthetic trade journal for developing and testing the analysis.

Deliberately *modest* and noisy, like a real discretionary journal:
  - overall edge around +0.1R per trade
  - losses cluster at -1R (the stop) with slippage and occasional gaps
  - winners vary (partials, runners)
  - session is derived from the entry timestamp, not assigned at random
  - edge differs by session/instrument, but by amounts that 400 trades
    can only partly resolve -- which is the point of the confidence
    intervals in analyze.py

`edge_scale=0` produces a journal with no edge anywhere, used to check
that the analysis does not "discover" patterns in pure noise.

Replace this file's output with a real export via loader.py.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

INSTRUMENTS = ["EURUSD", "GBPUSD", "XAUUSD", "NAS100", "BTCUSD"]
# UTC entry hours by session
SESSION_HOURS = {"Asia": (0, 7), "London": (7, 13), "New York": (13, 20)}
SESSION_WEIGHTS = [0.25, 0.40, 0.35]

# True expectancy contributions in R (additive), known only to the generator.
BASE_EDGE = 0.08
SESSION_EDGE = {"Asia": -0.25, "London": 0.10, "New York": 0.10}
INSTRUMENT_EDGE = {"EURUSD": 0.10, "GBPUSD": -0.05, "XAUUSD": 0.05, "NAS100": 0.0, "BTCUSD": -0.10}

AVG_WIN_R = 1.6
# stop (1R) + half-normal slippage (sd 0.03R) + 3% chance of a gap (mean 0.5R, capped)
AVG_LOSS_R = 1.0 + 0.03 * np.sqrt(2 / np.pi) + 0.03 * 0.5 * (1 - np.exp(-4))


def draw_r(edge, rng: np.random.Generator) -> np.ndarray:
    """R-multiples whose expectation equals `edge` (vectorised over edge)."""
    edge = np.asarray(edge, dtype=float)
    p_win = np.clip((edge + AVG_LOSS_R) / (AVG_WIN_R + AVG_LOSS_R), 0.05, 0.95)
    win = rng.random(edge.shape) < p_win
    win_r = AVG_WIN_R * rng.lognormal(-0.18, 0.6, edge.shape)          # multiplier has mean 1
    slip = np.abs(rng.normal(0.0, 0.03, edge.shape))
    gap = np.where(rng.random(edge.shape) < 0.03,
                   np.minimum(rng.exponential(0.5, edge.shape), 2.0), 0.0)  # rare gap through stop
    return np.where(win, win_r, -(1.0 + slip + gap))


def true_edges(sessions, instruments, edge_scale: float = 1.0) -> np.ndarray:
    if not edge_scale:
        return np.zeros(len(sessions))
    return BASE_EDGE + edge_scale * (pd.Series(sessions).map(SESSION_EDGE).to_numpy()
                                     + pd.Series(instruments).map(INSTRUMENT_EDGE).to_numpy())


def generate_trade_log(n_trades: int = 400, start: str = "2025-01-06", n_days: int = 260,
                       edge_scale: float = 1.0, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    sessions = rng.choice(list(SESSION_HOURS), size=n_trades, p=SESSION_WEIGHTS)
    lo = np.array([SESSION_HOURS[s][0] for s in sessions])
    hi = np.array([SESSION_HOURS[s][1] for s in sessions])
    hours = lo + (rng.random(n_trades) * (hi - lo)).astype(int)
    entry = (days[rng.integers(n_days, size=n_trades)]
             + pd.to_timedelta(hours, unit="h") + pd.to_timedelta(rng.integers(60, size=n_trades), unit="m"))
    instruments = rng.choice(INSTRUMENTS, size=n_trades)
    r = draw_r(true_edges(sessions, instruments, edge_scale), rng)
    risk_pct = rng.choice([0.5, 1.0, 1.0, 1.0], size=n_trades)
    df = pd.DataFrame({
        "entry_time": entry,
        "exit_time": entry + pd.to_timedelta(rng.exponential(90, n_trades).astype(int) + 3, unit="m"),
        "instrument": instruments,
        "direction": rng.choice(["Long", "Short"], size=n_trades),
        "risk_pct": risk_pct,
        "r_multiple": np.round(r, 3),
    }).sort_values("entry_time").reset_index(drop=True)
    df.insert(0, "trade_id", np.arange(1, len(df) + 1))
    df["pnl_pct"] = (df["r_multiple"] * df["risk_pct"]).round(4)
    return df


if __name__ == "__main__":
    out = Path(__file__).resolve().parents[1] / "data" / "trade_log.csv"
    df = generate_trade_log()
    df.to_csv(out, index=False)
    print(f"{len(df)} trades -> {out}")
    print(f"mean R {df.r_multiple.mean():+.3f}, win rate {(df.r_multiple > 0).mean():.1%}, "
          f"worst {df.r_multiple.min():.2f}R")
