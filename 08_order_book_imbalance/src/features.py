"""
Order-book features and forward targets on a regular clock grid.

Event-level quantities (after each event n)
-------------------------------------------
mid        (ask1 + bid1) / 2
spread     ask1 - bid1
imbalance  I1 = (bid_q1 - ask_q1) / (bid_q1 + ask_q1)            in [-1, 1]
depth imb  I3 = same with queue sizes summed over levels 1..3
microprice ask1 * bid_q1/(bid_q1+ask_q1) + bid1 * ask_q1/(bid_q1+ask_q1)
OFI        Cont, Kukanov & Stoikov (2014) order-flow imbalance increment
           e_n = bid_q_n 1{b_n >= b_{n-1}} - bid_q_{n-1} 1{b_n <= b_{n-1}}
               - ask_q_n 1{a_n <= a_{n-1}} + ask_q_{n-1} 1{a_n >= a_{n-1}}
           (net size added to the bid minus net size added to the ask)

Clock grid
----------
Every `dt` seconds, take the book as of the last event at or before t
(never after), the OFI summed over (t - window, t], and the target
mid(t + h) - mid(t) in ticks for each horizon h.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from lobster import TICK

OPEN, CLOSE = 34200.0, 57600.0          # 09:30 and 16:00 in seconds after midnight
TRIM = 300.0                            # skip the first/last 5 minutes (auction effects)


def event_features(df: pd.DataFrame) -> pd.DataFrame:
    out = pd.DataFrame({"time": df["time"].to_numpy()})
    a, b = df["ask_p1"].to_numpy(), df["bid_p1"].to_numpy()
    qa, qb = df["ask_q1"].to_numpy(), df["bid_q1"].to_numpy()
    out["mid"] = (a + b) / 2
    out["spread_ticks"] = np.round((a - b) / TICK)
    out["imb1"] = (qb - qa) / (qb + qa)
    qa3 = df[["ask_q1", "ask_q2", "ask_q3"]].sum(axis=1).to_numpy()
    qb3 = df[["bid_q1", "bid_q2", "bid_q3"]].sum(axis=1).to_numpy()
    out["imb3"] = (qb3 - qa3) / (qb3 + qa3)
    out["micro_minus_mid_ticks"] = (a * qb / (qb + qa) + b * qa / (qb + qa) - out["mid"]) / TICK
    out["ofi"] = ofi_increments(a, qa, b, qb)
    return out


def ofi_increments(a, qa, b, qb) -> np.ndarray:
    a0, qa0, b0, qb0 = np.r_[a[0], a[:-1]], np.r_[qa[0], qa[:-1]], np.r_[b[0], b[:-1]], np.r_[qb[0], qb[:-1]]
    e = qb * (b >= b0) - qb0 * (b <= b0) - qa * (a <= a0) + qa0 * (a >= a0)
    e[0] = 0.0
    return e


def clock_grid(ev: pd.DataFrame, dt: float = 1.0, horizons=(1, 5, 10, 30, 60),
               ofi_window: float = 10.0, start: float = OPEN + TRIM, end: float = CLOSE - TRIM) -> pd.DataFrame:
    grid = np.arange(start, end - max(horizons) + 1e-9, dt)
    t = ev["time"].to_numpy()
    idx = np.searchsorted(t, grid, side="right") - 1              # last event at or before grid time
    g = ev.iloc[idx][["mid", "spread_ticks", "imb1", "imb3", "micro_minus_mid_ticks"]].reset_index(drop=True)
    g.insert(0, "time", grid)
    cum = np.r_[0.0, np.cumsum(ev["ofi"].to_numpy())]
    lo = np.searchsorted(t, grid - ofi_window, side="right")
    g["ofi_past"] = cum[idx + 1] - cum[lo]
    for h in horizons:
        j = np.searchsorted(t, grid + h, side="right") - 1
        g[f"dmid_{h}s"] = (ev["mid"].to_numpy()[j] - g["mid"].to_numpy()) / TICK
        # OFI over the *same* future window, for the contemporaneous regression
        g[f"ofi_next_{h}s"] = cum[j + 1] - cum[idx + 1]
    return g


def next_mid_move(ev: pd.DataFrame) -> pd.DataFrame:
    """Event-time view: for every event after which the mid is about to change,
    the book state and the direction (+1/-1) of the next mid change."""
    mid = ev["mid"].to_numpy()
    change_idx = np.flatnonzero(np.diff(mid) != 0) + 1              # events that move the mid
    nxt = np.searchsorted(change_idx, np.arange(len(mid)), side="right")
    valid = nxt < len(change_idx)
    direction = np.zeros(len(mid))
    direction[valid] = np.sign(mid[change_idx[nxt[valid]]] - mid[np.arange(len(mid))[valid]])
    out = ev.loc[valid, ["time", "imb1", "imb3", "spread_ticks"]].copy()
    out["next_move"] = direction[valid]
    return out[out["next_move"] != 0]
