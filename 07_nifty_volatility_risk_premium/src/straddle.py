"""
Short ATM NIFTY straddle, delta-hedged daily, on real NIFTY / India VIX data.

Trade (one cycle)
-----------------
- Day 0: sell one 30-calendar-day ATM call + put. Strike = spot rounded to
  the 50-point NIFTY strike grid. Priced with Black-Scholes at implied vol
  = India VIX (VIX is a 30-day at-the-money-weighted implied vol, so this is
  a proxy for the ATM straddle's IV). India VIX is built from the whole
  out-of-the-money strip, including the put skew, so true ATM IV is usually
  somewhat lower; `iv_offset` prices and marks the straddle at VIX minus a
  haircut, and the analysis reports results for 0-3 vol points.
- Every trading day: re-mark the straddle at that day's VIX and time left,
  and rebalance a futures-style hedge to the straddle's delta.
- Expiry after 21 trading days: settle at intrinsic value.
- Costs: `opt_cost` as a fraction of premium, charged once at the sale
  (bid-ask, brokerage, STT and exchange fees; options are cash-settled at
  expiry), and `hedge_bps` on every traded hedge notional.

Cycles are back-to-back (one position at a time). Results are averaged
over all 21 possible start offsets so the answer does not depend on which
day the first trade happened to start.

P&L attribution
---------------
Daily hedged P&L ~ theta + gamma + vega:
    0.5 * Gamma * S^2 * (r_t^2 - sigma_imp^2 * dt)   (gamma vs theta, short sign flipped)
  + Vega * dIV                                        (re-marking at new VIX)
`attribution()` reports how much of the realised P&L each piece explains.

All P&L is reported in % of spot notional at entry (NIFTY points / spot).
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import norm

from data import TRADING_DAYS

STRIKE_STEP = 50.0


@dataclass
class Config:
    days: int = 21                 # trading days to expiry
    calendar_days: int = 30
    r: float = 0.065               # INR risk-free (approx. repo rate)
    opt_cost: float = 0.005        # 0.5% of premium, round trip on the option sale
    hedge_bps: float = 1.0         # per side on hedge notional
    hedge: bool = True
    iv_offset: float = 0.0         # ATM IV = VIX - iv_offset (vol points); VIX includes skew


def bs(S, K, T, r, sig):
    """Straddle price, delta, gamma, vega (per 1.00 vol) under Black-Scholes."""
    T = np.maximum(T, 1e-8)
    sq = sig * np.sqrt(T)
    d1 = (np.log(S / K) + (r + 0.5 * sig**2) * T) / sq
    d2 = d1 - sq
    call = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
    delta = 2 * norm.cdf(d1) - 1
    gamma = 2 * norm.pdf(d1) / (S * sq)
    vega = 2 * S * norm.pdf(d1) * np.sqrt(T)
    return call + put, delta, gamma, vega


def run_cycle(spot: np.ndarray, vix: np.ndarray, cfg: Config) -> dict:
    """One short straddle over spot[0..days] (vix in %, no NaNs)."""
    n = cfg.days
    S0, K = spot[0], STRIKE_STEP * round(spot[0] / STRIKE_STEP)
    T = cfg.calendar_days / 365 * (1 - np.arange(n + 1) / n)
    sig = np.maximum(vix[: n + 1] - cfg.iv_offset, 1.0) / 100
    price, delta, gamma, vega = bs(spot, K, T, cfg.r, sig)
    price[n] = abs(spot[n] - K)                          # settle at intrinsic
    delta[n] = 0.0
    held = delta if cfg.hedge else np.zeros(n + 1)       # long underlying against the short straddle
    held[n] = 0.0
    dS = np.diff(spot)
    opt_pnl = -np.diff(price)                            # short the straddle
    hedge_pnl = held[:-1] * dS
    hedge_trades = np.abs(np.diff(np.r_[0.0, held]))     # entry, daily rebalances, final unwind
    hedge_cost = hedge_trades * spot * cfg.hedge_bps / 1e4
    daily = opt_pnl + hedge_pnl - hedge_cost[1:]
    daily[0] -= cfg.opt_cost * price[0] + hedge_cost[0]
    # attribution with the previous day's greeks
    gamma_pnl = -0.5 * gamma[:-1] * dS**2
    theta_pnl = 0.5 * gamma[:-1] * spot[:-1] ** 2 * sig[:-1] ** 2 / TRADING_DAYS
    vega_pnl = -vega[:-1] * np.diff(sig)
    vega_pnl[-1] = 0.0                                   # last day settles at intrinsic, no re-mark
    return {"S0": S0, "K": K, "premium_pct": price[0] / S0 * 100, "iv_entry": sig[0] * 100,
            "rv_realised": float(np.sqrt(np.sum(np.diff(np.log(spot)) ** 2) * TRADING_DAYS / n) * 100),
            "pnl_pct": daily.sum() / S0 * 100,
            "cost_pct": (cfg.opt_cost * price[0] + hedge_cost.sum()) / S0 * 100,
            "gamma_pct": gamma_pnl.sum() / S0 * 100, "theta_pct": theta_pnl.sum() / S0 * 100,
            "vega_pct": vega_pnl.sum() / S0 * 100, "daily_pct": daily / S0 * 100}


def backtest(df: pd.DataFrame, cfg: Config = Config(), offset: int = 0,
             entry_filter=None) -> pd.DataFrame:
    """Back-to-back cycles starting `offset` days into the VIX sample.

    entry_filter(row) -> bool decides at entry, using only data known then,
    whether to sell this cycle (else stay flat for the cycle).
    """
    d = df.loc[df["vix"].first_valid_index():].copy()
    d["vix"] = d["vix"].ffill()
    spot, vix = d["spot"].to_numpy(), d["vix"].to_numpy()
    rows = []
    for start in range(offset, len(d) - cfg.days, cfg.days):
        entry = d.iloc[start]
        if entry_filter is not None and not entry_filter(entry):
            rows.append({"entry": d.index[start], "traded": False, "pnl_pct": 0.0})
            continue
        res = run_cycle(spot[start:start + cfg.days + 1], vix[start:start + cfg.days + 1], cfg)
        res.pop("daily_pct")
        rows.append({"entry": d.index[start], "exit": d.index[start + cfg.days], "traded": True, **res})
    return pd.DataFrame(rows)


def all_offsets(df: pd.DataFrame, cfg: Config = Config(), entry_filter=None) -> pd.DataFrame:
    return pd.concat([backtest(df, cfg, o, entry_filter).assign(offset=o) for o in range(cfg.days)],
                     ignore_index=True)


def stats(trades: pd.DataFrame, cycles_per_year: float = TRADING_DAYS / 21) -> dict:
    t = trades[trades.traded]
    p = trades["pnl_pct"]
    return {
        "cycles": len(trades), "traded": int(trades.traded.sum()),
        "mean_pnl_pct": p.mean(), "median_pnl_pct": p.median(),
        "win_rate": (t.pnl_pct > 0).mean(),
        "worst_pct": p.min(), "best_pct": p.max(),
        "annual_return_pct": p.mean() * cycles_per_year,
        "annual_vol_pct": p.std() * np.sqrt(cycles_per_year),
        "sharpe": p.mean() / p.std() * np.sqrt(cycles_per_year) if p.std() > 0 else np.nan,
        "skew": p.skew(),
        "mean_premium_pct": t.premium_pct.mean() if "premium_pct" in t else np.nan,
        "mean_cost_pct": t.cost_pct.mean() if "cost_pct" in t else np.nan,
    }


if __name__ == "__main__":
    from data import load
    from vrp import build
    df = build(load())
    pd.set_option("display.width", 160)
    for name, cfg in [("hedged", Config()), ("unhedged", Config(hedge=False))]:
        tr = all_offsets(df, cfg)
        print(name, {k: round(v, 3) if isinstance(v, float) else v for k, v in stats(tr).items()})
    tr = backtest(df, Config())
    print(tr[["entry", "iv_entry", "rv_realised", "premium_pct", "pnl_pct", "gamma_pct", "theta_pct",
              "vega_pct"]].sort_values("pnl_pct").head(6).to_string(index=False, float_format="%.2f"))
