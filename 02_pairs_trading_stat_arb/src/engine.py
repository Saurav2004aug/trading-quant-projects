"""
Pairs-trading backtest engine (one engine for synthetic and real data).

Spread and sizing
-----------------
    spread_t = B_t - alpha - beta * A_t
A long-spread position holds +u units of B and -beta*u units of A, so its
daily P&L is exactly u * (change in spread). The sign of beta is kept
(a negatively related pair is traded as B + |beta|*A). u is set at entry
so the gross notional |u*B| + |beta*u*A| equals `capital`.

Timing (no look-ahead)
----------------------
- The z-score at bar t uses the spread up to and including bar t.
- A signal generated at the close of bar t is executed at the close of
  bar t + `execution_lag` (default 1, i.e. next close).
- P&L at bar t comes from units held at the end of bar t-1.

Signal state machine
--------------------
flat  -> long   if z < -entry_z        flat -> short if z > entry_z
long  -> flat   if z > -exit_z  (reverted)   or z < -stop_z (stop-loss)
short -> flat   if z <  exit_z  (reverted)   or z >  stop_z (stop-loss)
After a stop-loss, re-entry is blocked until |z| < exit_z.

Costs
-----
`cost_bps` is charged on traded notional (|dUnits_A|*A + |dUnits_B|*B) at
every entry and exit, and flows into both the daily P&L series and the
per-trade P&L, so sum(trade net P&L) == total net P&L exactly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TRADING_DAYS = 252


@dataclass
class Params:
    lookback: int = 30
    entry_z: float = 2.0
    exit_z: float = 0.5
    stop_z: float | None = 4.0
    cost_bps: float = 2.0
    execution_lag: int = 1
    capital: float = 1.0


def rolling_zscore(spread: pd.Series, window: int) -> pd.Series:
    mean = spread.rolling(window).mean()
    std = spread.rolling(window).std(ddof=1)
    return (spread - mean) / std.replace(0.0, np.nan)


def generate_signals(z: np.ndarray, p: Params) -> tuple[np.ndarray, np.ndarray]:
    """Target position (-1/0/+1) decided at each bar, and the reason for exits."""
    pos, blocked = 0, False
    target = np.zeros(len(z), dtype=int)
    reason = np.full(len(z), "", dtype=object)
    stop = np.inf if p.stop_z is None else p.stop_z
    for t, zt in enumerate(z):
        if np.isnan(zt):
            target[t] = pos
            continue
        if blocked and abs(zt) < p.exit_z:
            blocked = False
        if pos == 0 and not blocked:
            if zt < -p.entry_z and zt > -stop:
                pos = 1
            elif zt > p.entry_z and zt < stop:
                pos = -1
        elif pos == 1:
            if zt < -stop:
                pos, blocked, reason[t] = 0, True, "stop"
            elif zt > -p.exit_z:
                pos, reason[t] = 0, "revert"
        elif pos == -1:
            if zt > stop:
                pos, blocked, reason[t] = 0, True, "stop"
            elif zt < p.exit_z:
                pos, reason[t] = 0, "revert"
        target[t] = pos
    return target, reason


def backtest(data: pd.DataFrame, alpha: float, beta: float, p: Params,
             start: int = 0, end: int | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Trade bars [start, end) of `data` (columns asset_a, asset_b).

    Bars before `start` are only used to warm up the rolling z-score, so the
    caller can pass training history without trading on it. Any position
    still open at the last bar is closed there (with costs).

    Returns (daily, trades).
    """
    end = len(data) if end is None else end
    a_all = data["asset_a"].to_numpy(float)
    b_all = data["asset_b"].to_numpy(float)
    spread_all = pd.Series(b_all - alpha - beta * a_all, index=data.index)
    z_all = rolling_zscore(spread_all, p.lookback).to_numpy()

    idx = data.index[start:end]
    a, b, z = a_all[start:end], b_all[start:end], z_all[start:end]
    signal, reason = generate_signals(z, p)
    lag = p.execution_lag
    held = np.concatenate([np.zeros(lag, dtype=int), signal[:-lag]]) if lag else signal.copy()
    exit_reason = np.concatenate([np.full(lag, "", dtype=object), reason[:-lag]]) if lag else reason
    held[-1] = 0  # flatten at the end of the test window
    rate = p.cost_bps / 10_000

    n = len(idx)
    units_a = np.zeros(n)
    units_b = np.zeros(n)
    gross = np.zeros(n)
    cost = np.zeros(n)
    trades, open_trade = [], None
    ua = ub = 0.0

    for t in range(n):
        if t > 0:
            gross[t] = ua * (a[t] - a[t - 1]) + ub * (b[t] - b[t - 1])
            if open_trade is not None:
                open_trade["gross"] += gross[t]
                open_trade["days"] += 1
        prev = int(np.sign(ub)) if ub else 0
        if held[t] != prev:
            new_ub = held[t] * p.capital / (b[t] + abs(beta) * a[t]) if held[t] else 0.0
            new_ua = -beta * new_ub
            cost[t] = (abs(new_ua - ua) * a[t] + abs(new_ub - ub) * b[t]) * rate
            if prev != 0:                       # close the open trade
                c_exit = (abs(ua) * a[t] + abs(ub) * b[t]) * rate
                open_trade["cost"] += c_exit
                open_trade.update(exit_date=idx[t],
                                  exit_reason=exit_reason[t] or ("end" if t == n - 1 else "flip"))
                trades.append(open_trade)
                open_trade = None
            if held[t] != 0:                    # open a new trade
                c_entry =(abs(new_ua) * a[t] + abs(new_ub) * b[t]) * rate
                open_trade = {"entry_date": idx[t], "direction": "long" if held[t] > 0 else "short",
                              "entry_z": z[t - lag] if t >= lag else np.nan,
                              "gross": 0.0, "cost": c_entry, "days": 0}
            ua, ub = new_ua, new_ub
        units_a[t], units_b[t] = ua, ub

    daily = pd.DataFrame({
        "asset_a": a, "asset_b": b, "spread": spread_all.iloc[start:end].to_numpy(),
        "zscore": z, "signal": signal, "position": np.sign(units_b).astype(int),
        "units_a": units_a, "units_b": units_b,
        "gross_pnl": gross, "cost": cost, "net_pnl": gross - cost,
    }, index=idx)
    daily["cum_net"] = daily["net_pnl"].cumsum()
    daily["cum_gross"] = daily["gross_pnl"].cumsum()

    trades = pd.DataFrame(trades, columns=["entry_date", "exit_date", "direction", "entry_z",
                                           "days", "gross", "cost", "exit_reason"])
    trades["net"] = trades["gross"] - trades["cost"]
    return daily, trades


def summarize(daily: pd.DataFrame, trades: pd.DataFrame, capital: float = 1.0) -> dict:
    r = daily["net_pnl"] / capital
    n_years = len(r) / TRADING_DAYS
    vol = r.std(ddof=1) * np.sqrt(TRADING_DAYS)
    equity = capital + daily["net_pnl"].cumsum()
    dd = (equity.cummax() - equity) / equity.cummax()
    wins, losses = trades.net[trades.net > 0], trades.net[trades.net < 0]
    ann_ret = r.sum() / n_years if n_years else np.nan
    return {
        "net_return": float(r.sum()),
        "gross_return": float(daily["gross_pnl"].sum() / capital),
        "costs": float(daily["cost"].sum() / capital),
        "annual_return": float(ann_ret),
        "annual_vol": float(vol),
        "sharpe": float(r.mean() / r.std(ddof=1) * np.sqrt(TRADING_DAYS)) if r.std() > 0 else np.nan,
        "t_stat_mean": float(r.mean() / r.std(ddof=1) * np.sqrt(len(r))) if r.std() > 0 else np.nan,
        "max_drawdown": float(dd.max()),
        "trades": int(len(trades)),
        "win_rate": float((trades.net > 0).mean()) if len(trades) else np.nan,
        "profit_factor": float(wins.sum() / -losses.sum()) if len(losses) and losses.sum() else np.nan,
        "avg_trade": float(trades.net.mean()) if len(trades) else np.nan,
        "avg_hold_days": float(trades.days.mean()) if len(trades) else np.nan,
        "stop_outs": int((trades.exit_reason == "stop").sum()),
        "time_in_market": float((daily["position"] != 0).mean()),
        "years": float(n_years),
    }
