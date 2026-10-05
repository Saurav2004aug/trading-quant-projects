"""
How many trades does a journal need before a breakdown means anything?

Two simulations on synthetic journals where the true edges are known:

1. Power: the generator gives the Asia session a real handicap (about
   -0.35R versus the other sessions). For each journal size, what fraction
   of journals flag Asia as significantly weaker (Welch t-test, p < 0.05)?

2. False discoveries: journals with *no* edge anywhere (edge_scale=0).
   How often does at least one of the 5 instruments look "significantly"
   different -- with raw p-values vs Benjamini-Hochberg corrected ones?

Welch's t-test is used here instead of the permutation test in analyze.py
only for speed; on samples this size they agree closely.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind

from analyze import benjamini_hochberg
from generate_trade_log import generate_trade_log


def _bucket_pvalues(df: pd.DataFrame, by: str) -> pd.Series:
    return pd.Series({k: ttest_ind(g.r_multiple, df.loc[df[by] != k, "r_multiple"],
                                   equal_var=False).pvalue for k, g in df.groupby(by)})


def power_curve(sizes=(100, 200, 400, 800, 1600, 3200), reps: int = 300, seed: int = 100):
    rows = []
    for n in sizes:
        hits = 0
        for i in range(reps):
            df = generate_trade_log(n_trades=n, seed=seed + i)
            df["session"] = np.select([df.entry_time.dt.hour < 7, df.entry_time.dt.hour < 13],
                                      ["Asia", "London"], "New York")
            asia = df[df.session == "Asia"].r_multiple
            rest = df[df.session != "Asia"].r_multiple
            p = ttest_ind(asia, rest, equal_var=False).pvalue
            hits += (p < 0.05) and asia.mean() < rest.mean()
        rows.append({"n_trades": n, "power": hits / reps})
    return pd.DataFrame(rows)


def false_discovery_rate(n_trades: int = 400, reps: int = 500, seed: int = 5000):
    raw = bh = 0
    for i in range(reps):
        df = generate_trade_log(n_trades=n_trades, edge_scale=0.0, seed=seed + i)
        p = _bucket_pvalues(df, "instrument").values
        raw += (p < 0.05).any()
        bh += (benjamini_hochberg(p) < 0.1).any()
    return {"n_trades": n_trades, "any_flag_raw_p05": raw / reps, "any_flag_bh_q10": bh / reps}


if __name__ == "__main__":
    print(power_curve().to_string(index=False))
    print(false_discovery_rate())
