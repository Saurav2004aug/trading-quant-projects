"""
Trade-journal analytics with uncertainty attached to every number.

The question a journal should answer is not "which bucket has the best
average" -- with a few dozen trades per bucket, some bucket will always
look best by luck -- but "which differences are larger than noise".
So every expectancy comes with a bootstrap confidence interval, every
bucket is tested against the rest of the journal with a permutation test,
and the p-values are corrected for testing many buckets at once
(Benjamini-Hochberg false-discovery-rate control).

Usage:
    python analyze.py                               # bundled synthetic journal
    python analyze.py path/to/journal.csv           # standard schema
    python analyze.py export.csv --mt5 --balance 100000
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from loader import load_journal

DATA = Path(__file__).resolve().parents[1] / "data" / "trade_log.csv"
TRADING_DAYS = 252


# ------------------------------------------------------------------ statistics
def bootstrap_ci(x, stat=np.mean, n_boot: int = 5000, level: float = 0.95, seed: int = 0):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(seed)
    boots = stat(rng.choice(x, size=(n_boot, len(x)), replace=True), axis=1)
    a = (1 - level) / 2
    return float(np.quantile(boots, a)), float(np.quantile(boots, 1 - a))


def permutation_pvalue(group, rest, n_perm: int = 5000, seed: int = 0) -> float:
    """Two-sided p-value for a difference in mean R between a bucket and the rest."""
    group, rest = np.asarray(group, float), np.asarray(rest, float)
    observed = abs(group.mean() - rest.mean())
    pooled = np.concatenate([group, rest])
    rng = np.random.default_rng(seed)
    idx = np.argsort(rng.random((n_perm, len(pooled))), axis=1)[:, :len(group)]
    g = pooled[idx].mean(axis=1)
    r = (pooled.sum() - g * len(group)) / len(rest)
    return float((np.sum(np.abs(g - r) >= observed - 1e-12) + 1) / (n_perm + 1))


def benjamini_hochberg(p: np.ndarray) -> np.ndarray:
    """BH-adjusted p-values (q-values)."""
    p = np.asarray(p, float)
    order = np.argsort(p)
    ranked = p[order] * len(p) / np.arange(1, len(p) + 1)
    q = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty_like(q)
    out[order] = np.minimum(q, 1.0)
    return out


def max_losing_streak(r: pd.Series) -> int:
    streak = best = 0
    for x in r:
        streak = streak + 1 if x < 0 else 0
        best = max(best, streak)
    return best


# ------------------------------------------------------------------ journal metrics
def daily_returns(df: pd.DataFrame) -> pd.Series:
    """Daily % return on *every* business day, zero on days without trades
    (dropping flat days would overstate the Sharpe ratio)."""
    by_day = df.groupby(df["entry_time"].dt.normalize())["pnl_pct"].sum() / 100
    full = pd.bdate_range(by_day.index.min(), by_day.index.max())
    return by_day.reindex(full, fill_value=0.0)


def equity_curve(df: pd.DataFrame) -> pd.Series:
    return (1 + df["pnl_pct"] / 100).cumprod()


def overall_stats(df: pd.DataFrame) -> dict:
    r = df["r_multiple"].dropna()
    wins, losses = r[r > 0], r[r <= 0]
    daily = daily_returns(df)
    sharpe = daily.mean() / daily.std() * np.sqrt(TRADING_DAYS) if daily.std() > 0 else np.nan
    sharpe_ci = bootstrap_ci(daily.values,
                             stat=lambda a, axis: a.mean(axis) / a.std(axis, ddof=1) * np.sqrt(TRADING_DAYS))
    eq = equity_curve(df)
    dd = eq / eq.cummax() - 1
    lo, hi = bootstrap_ci(r)
    return {
        "n_trades": len(r),
        "win_rate": wins.size / len(r),
        "avg_win_r": wins.mean(),
        "avg_loss_r": losses.mean(),
        "expectancy_r": r.mean(),
        "expectancy_ci_low": lo,
        "expectancy_ci_high": hi,
        "t_stat_expectancy": r.mean() / (r.std(ddof=1) / np.sqrt(len(r))),
        "profit_factor": wins.sum() / -losses.sum() if losses.sum() else np.nan,
        "worst_trade_r": r.min(),
        "trades_beyond_minus_1_5r": int((r < -1.5).sum()),
        "max_losing_streak": max_losing_streak(r),
        "sharpe_daily_annualised": sharpe,
        "sharpe_ci_low": sharpe_ci[0],
        "sharpe_ci_high": sharpe_ci[1],
        "total_return_pct": (eq.iloc[-1] - 1) * 100,
        "max_drawdown_pct": dd.min() * 100,
        "trading_days": len(daily),
    }


def breakdown(df: pd.DataFrame, by: str, min_trades: int = 20) -> pd.DataFrame:
    rows = []
    for key, g in df.groupby(by):
        r = g["r_multiple"].dropna()
        rest = df.loc[df[by] != key, "r_multiple"].dropna()
        lo, hi = bootstrap_ci(r)
        rows.append({by: key, "n_trades": len(r), "win_rate": (r > 0).mean(),
                     "expectancy_r": r.mean(), "ci_low": lo, "ci_high": hi,
                     "total_r": r.sum(), "p_vs_rest": permutation_pvalue(r, rest)})
    out = pd.DataFrame(rows).set_index(by)
    out["q_value"] = benjamini_hochberg(out["p_vs_rest"].values)
    out["verdict"] = np.select(
        [out.n_trades < min_trades, (out.q_value < 0.1) & (out.expectancy_r < 0),
         (out.q_value < 0.1) & (out.expectancy_r > 0)],
        ["too few trades", "weaker than rest", "stronger than rest"], default="not distinguishable")
    return out.sort_values("expectancy_r", ascending=False)


def rolling_expectancy(df: pd.DataFrame, window: int = 50) -> pd.Series:
    return df["r_multiple"].rolling(window).mean()


def report(df: pd.DataFrame) -> str:
    s = overall_stats(df)
    lines = ["=== Overall ==="]
    lines.append(f"  trades {s['n_trades']}, win rate {s['win_rate']:.1%}, "
                 f"avg win {s['avg_win_r']:.2f}R, avg loss {s['avg_loss_r']:.2f}R")
    lines.append(f"  expectancy {s['expectancy_r']:+.3f}R  "
                 f"(95% CI {s['expectancy_ci_low']:+.3f} to {s['expectancy_ci_high']:+.3f}, "
                 f"t = {s['t_stat_expectancy']:.2f})")
    lines.append(f"  Sharpe (all business days) {s['sharpe_daily_annualised']:.2f}  "
                 f"(95% CI {s['sharpe_ci_low']:.2f} to {s['sharpe_ci_high']:.2f})")
    lines.append(f"  return {s['total_return_pct']:.1f}%, max drawdown {s['max_drawdown_pct']:.1f}%, "
                 f"longest losing streak {s['max_losing_streak']}")
    lines.append(f"  worst trade {s['worst_trade_r']:.2f}R, trades worse than -1.5R: "
                 f"{s['trades_beyond_minus_1_5r']}  (stop discipline check)")
    for by in ("session", "instrument"):
        b = breakdown(df, by)
        lines.append(f"\n=== By {by} ===")
        lines.append(b[["n_trades", "win_rate", "expectancy_r", "ci_low", "ci_high",
                        "p_vs_rest", "q_value", "verdict"]].round(3).to_string())
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("path", nargs="?", default=str(DATA))
    ap.add_argument("--mt5", action="store_true", help="input is an MT5 positions export")
    ap.add_argument("--balance", type=float, help="account balance (needed for --mt5)")
    ap.add_argument("--utc-offset", type=float, default=0.0, help="broker time minus UTC, hours")
    args = ap.parse_args(argv)
    df = load_journal(args.path, fmt="mt5" if args.mt5 else "standard",
                      account_balance=args.balance, utc_offset_hours=args.utc_offset)
    print(report(df))


if __name__ == "__main__":
    main()
