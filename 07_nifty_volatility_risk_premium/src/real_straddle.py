"""
Short NIFTY monthly ATM straddle at *real* NSE closing prices, delta-hedged
daily with the same-expiry future. No volatility proxy is involved.

Cycle k
-------
entry   first trading day after monthly expiry k-1, at the close
option  monthly expiry k, strike = traded strike closest to the implied
        forward (both legs must have traded that day)
marks   daily closing prices of that call and put (forward-filled if a leg
        did not trade that day, which is flagged)
hedge   futures position = Black-76 delta of the straddle (dV/dF) at each
        leg's own implied vol, rebalanced at every close
expiry  options settle at |S_T - K| with S_T the NIFTY close on expiry day
        (NSE final settlement for index options); the hedge closes there too
costs   opt_cost x premium at the sale, hedge_bps on traded futures notional

P&L is in % of the forward at entry, so cycles are comparable over time.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import load
from options_data import black76_delta, build_iv30, implied_vol, load_chain


def run_cycles(chain: pd.DataFrame, ts: pd.DataFrame, spot: pd.Series, opt_cost: float = 0.005,
               hedge_bps: float = 1.0, hedge: bool = True) -> pd.DataFrame:
    expiries = sorted(chain["expiry"].unique())
    fwd = ts.set_index(["date", "expiry"])["forward"]
    rows = []
    for prev_exp, exp in zip(expiries[:-1], expiries[1:]):
        days = sorted(chain.loc[(chain.expiry == exp) & (chain.date > prev_exp) & (chain.date <= exp), "date"].unique())
        if len(days) < 10 or exp not in spot.index:
            continue
        entry = days[0]
        if (entry, exp) not in fwd.index:
            continue
        g = chain[(chain.expiry == exp)]
        g0 = g[g.date == entry]
        w0 = g0.pivot_table(index="strike", columns="type", values="contracts", aggfunc="last")
        traded = w0[(w0.get("CE", 0) > 0) & (w0.get("PE", 0) > 0)].index
        if len(traded) == 0:
            continue
        F0 = fwd.loc[(entry, exp)]
        K = traded[np.argmin(np.abs(traded - F0))]
        legs = g[g.strike == K].pivot_table(index="date", columns="type", values=["close", "contracts"], aggfunc="last")
        legs = legs.reindex(days)
        stale = int(((legs[("contracts", "CE")].fillna(0) == 0) | (legs[("contracts", "PE")].fillna(0) == 0)).sum())
        call = legs[("close", "CE")].ffill().to_numpy().copy()
        put = legs[("close", "PE")].ffill().to_numpy().copy()
        # parity forward where available, otherwise spot (basis is a few points near expiry)
        F = np.array([fwd.get((d, exp), spot.get(d, np.nan)) for d in days], dtype=float)
        missing_fwd = int(sum((d, exp) not in fwd.index for d in days[:-1]))
        T = np.array([(exp - d).days / 365 for d in days])
        S_T = spot.loc[exp]
        value = call + put
        value[-1] = abs(S_T - K)                            # settlement on expiry day
        F[-1] = S_T
        premium = call[0] + put[0]
        # hedge ratios from each leg's own implied vol
        h = np.zeros(len(days))
        if hedge:
            last_iv = np.nan
            for i in range(len(days) - 1):
                ivs = [implied_vol(call[i], F[i], K, T[i], "C"), implied_vol(put[i], F[i], K, T[i], "P")]
                if np.isfinite(ivs).any():
                    last_iv = np.nanmean(ivs)
                # if the vol cannot be inverted (price at intrinsic near expiry), reuse the
                # last good vol but always recompute the delta at today's forward
                h[i] = black76_delta(F[i], K, T[i], last_iv, kind="C") \
                    + black76_delta(F[i], K, T[i], last_iv, kind="P")
        opt_pnl = -np.diff(value)
        hedge_pnl = h[:-1] * np.diff(F)
        hedge_cost = np.abs(np.diff(np.r_[0.0, h])) * F * hedge_bps / 1e4
        pnl = opt_pnl.sum() + hedge_pnl.sum() - hedge_cost.sum() - opt_cost * premium
        rv = np.sqrt(np.sum(np.diff(np.log(spot.loc[entry:exp].to_numpy())) ** 2) * 252 / (len(days) - 1)) * 100
        iv0 = np.nanmean([implied_vol(call[0], F0, K, T[0], "C"), implied_vol(put[0], F0, K, T[0], "P")]) * 100
        rows.append({"entry": entry, "expiry": exp, "days": len(days), "strike": K, "forward0": F0,
                     "premium": premium, "premium_pct": premium / F0 * 100, "settle_value": value[-1],
                     "iv_entry": iv0, "rv_realised": rv, "stale_marks": stale, "missing_forward_days": missing_fwd,
                     "option_pnl_pct": opt_pnl.sum() / F0 * 100, "hedge_pnl_pct": hedge_pnl.sum() / F0 * 100,
                     "cost_pct": (hedge_cost.sum() + opt_cost * premium) / F0 * 100,
                     "pnl_pct": pnl / F0 * 100})
    return pd.DataFrame(rows)


def summarise(c: pd.DataFrame) -> dict:
    p = c["pnl_pct"]
    per_year = 12
    return {"cycles": len(c), "first_entry": c.entry.min().date(), "last_expiry": c.expiry.max().date(),
            "mean_pnl_pct": p.mean(), "median_pnl_pct": p.median(), "t_stat": p.mean() / p.std() * np.sqrt(len(p)),
            "win_rate": (p > 0).mean(), "worst_pct": p.min(), "worst_entry": c.loc[p.idxmin(), "entry"].date(),
            "best_pct": p.max(), "sharpe": p.mean() / p.std() * np.sqrt(per_year), "skew": p.skew(),
            "mean_premium_pct": c.premium_pct.mean(), "mean_iv_entry": c.iv_entry.mean(),
            "mean_rv_realised": c.rv_realised.mean(), "mean_cost_pct": c.cost_pct.mean(),
            "stale_marks_share": c.stale_marks.sum() / c.days.sum()}


if __name__ == "__main__":
    chain = load_chain()
    ts, _ = build_iv30()
    spot = load()["spot"]
    for name, kw in [("hedged", {}), ("unhedged", {"hedge": False})]:
        c = run_cycles(chain, ts, spot, **kw)
        print(name, {k: (round(v, 3) if isinstance(v, float) else v) for k, v in summarise(c).items()})
    c = run_cycles(chain, ts, spot)
    print(c.sort_values("pnl_pct").head(5)[["entry", "expiry", "strike", "iv_entry", "rv_realised",
                                             "premium_pct", "pnl_pct"]].to_string(index=False, float_format="%.2f"))
