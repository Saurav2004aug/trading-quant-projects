"""
Position-size sweep: growth vs risk from 0.25% to well past full Kelly.

Shows the three regimes of fixed-fractional sizing:
  f < f*/2     low growth, low drawdown      (where professionals sit)
  f = f*       maximum median growth, large drawdowns
  f > 2 f*     median growth turns negative even though every trade has
               positive expectancy -- over-betting destroys a real edge.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from simulate import (exact_median_final, kelly_fraction, max_drawdown, ruin_probability,
                      simulate_equity_paths)


def sweep(win_rate: float = 0.45, payoff: float = 1.5, n_trades: int = 240,
          n_paths: int = 10_000, risk_fracs=None, dd_limit: float = 0.2,
          empirical_r=None, seed: int = 1) -> pd.DataFrame:
    if risk_fracs is None:
        k = kelly_fraction(win_rate, payoff) if empirical_r is None else 0.1
        risk_fracs = np.unique(np.round(np.r_[np.linspace(0.0025, 2.6 * k, 40), k / 2, k], 5))
    rows = []
    for i, f in enumerate(risk_fracs):
        paths = simulate_equity_paths(n_trades, n_paths, win_rate, payoff, f, empirical_r, seed + i)
        final = paths[:, -1]
        rows.append({
            "risk_frac": f,
            "p_ruin_50pct": ruin_probability(paths, 0.5),
            f"p_dd_ge_{int(dd_limit * 100)}pct": float((max_drawdown(paths) >= dd_limit).mean()),
            "median_final": float(np.median(final)),
            "mean_final": float(final.mean()),
            "p05_final": float(np.quantile(final, 0.05)),
            "p_loss": float((final < 1).mean()),
            "theory_median": float(exact_median_final(f, win_rate, payoff, n_trades))
            if empirical_r is None else np.nan,
        })
    return pd.DataFrame(rows)


if __name__ == "__main__":
    p, b = 0.45, 1.5
    k = kelly_fraction(p, b)
    print(f"Win rate {p:.0%}, payoff {b}R -> expectancy {p*b-(1-p):+.3f}R, full Kelly {k:.2%}")
    df = sweep(p, b)
    pd.set_option("display.width", 140)
    print(df.round(4).to_string(index=False))
    best = df.loc[df.median_final.idxmax()]
    print(f"\nMedian final equity peaks at f = {best.risk_frac:.2%} (Kelly = {k:.2%})")
