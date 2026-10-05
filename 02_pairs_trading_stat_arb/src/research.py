"""
Walk-forward pairs-trading research on real (or synthetic) prices.

Protocol
--------
The history is cut into consecutive blocks. For every block:
  1. Use only the preceding `train_days` bars to (a) run Engle-Granger on
     every candidate pair, (b) estimate alpha/beta for pairs that pass.
  2. Trade the passing pairs (equal capital) over the next `test_days`
     bars with those frozen parameters. Positions are closed at block end.
  3. Roll forward and repeat.
A pair that is not cointegrated in its training window is not traded.
When more than one pair is screened, the 1% critical value is used to
limit false discoveries from testing many pairs.

The single chronological split (train 60% / test 40%) is reported too,
as the simplest possible out-of-sample check.

Usage
-----
    python research.py                                   # EURUSD vs GBPUSD, 10y, Yahoo
    python research.py --tickers EWA EWC                 # any Yahoo tickers
    python research.py --tickers EWA EWC EWU EWG EWQ     # screen a universe
    python research.py --csv prices.csv                  # your own data: date + price columns
    python research.py --csv ../data/fx_daily_fred.csv --cols EURUSD GBPUSD --start 2016-01-01
    python research.py --synthetic                       # offline demo on simulated data

Outputs go to ../results/<run-name>/ and summary.md in that folder is
written for pasting into the README.
"""
from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import numpy as np
import pandas as pd

from cointegration import engle_granger_test
from engine import Params, backtest, summarize

ROOT = Path(__file__).resolve().parents[1]


# ---------------------------------------------------------------- data
def load_yahoo(tickers: list[str], period: str) -> pd.DataFrame:
    try:
        import yfinance as yf
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("pip install yfinance  (or use --csv / --synthetic)") from exc
    raw = yf.download(tickers, period=period, interval="1d", auto_adjust=True,
                      progress=False, threads=False)
    if raw.empty:
        raise SystemExit("Yahoo Finance returned no data (network or ticker problem).")
    close = raw["Close"] if isinstance(raw.columns, pd.MultiIndex) else raw[["Close"]]
    return clean_prices(close[tickers])


def load_csv(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, index_col=0, parse_dates=True)
    return clean_prices(df.select_dtypes("number"))


def load_synthetic(n: int = 2500, seed: int = 3) -> pd.DataFrame:
    """Three series: X and Y share a trend with a mean-reverting spread whose
    strength changes over time (so cointegration comes and goes); Z is an
    unrelated random walk."""
    rng = np.random.default_rng(seed)
    trend = 100 + np.cumsum(rng.normal(0.02, 1.0, n))
    theta = np.where((np.arange(n) // 500) % 2 == 0, 0.08, 0.005)   # regime switch
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = s[t - 1] - theta[t] * s[t - 1] + 0.7 * rng.standard_normal()
    x = trend
    y = 5 + 0.8 * trend + s
    z = 80 + np.cumsum(rng.normal(0, 1.0, n))
    idx = pd.bdate_range("2014-01-01", periods=n)
    return pd.DataFrame({"X": x, "Y": y, "Z": z}, index=idx)


def clean_prices(df: pd.DataFrame) -> pd.DataFrame:
    df = df.sort_index().dropna(how="any")
    df = df[(df > 0).all(axis=1)]
    if len(df) < 500:
        raise SystemExit(f"Only {len(df)} complete rows; need at least 500.")
    return df


# ---------------------------------------------------------------- research
def walk_forward(prices: pd.DataFrame, p: Params, train_days: int = 504,
                 test_days: int = 63, significance: str | None = None,
                 require_coint: bool = True):
    """Returns (daily, trades, blocks)."""
    pairs = list(itertools.combinations(prices.columns, 2))
    significance = significance or ("5%" if len(pairs) == 1 else "1%")
    daily_parts, trade_parts, blocks = [], [], []

    for start in range(train_days, len(prices), test_days):
        end = min(start + test_days, len(prices))
        train = prices.iloc[start - train_days:start]
        selected = []
        for a_col, b_col in pairs:
            eg = engle_granger_test(train[a_col], train[b_col], significance)
            blocks.append({"block_start": prices.index[start], "pair": f"{a_col}/{b_col}",
                           "t_stat": eg["adf"]["t_stat"],
                           "critical": eg["adf"]["critical_values"][significance],
                           "beta": eg["beta"], "half_life": eg["half_life"],
                           "traded": bool(eg["is_cointegrated"] or not require_coint)})
            if eg["is_cointegrated"] or not require_coint:
                selected.append((a_col, b_col, eg))

        window = prices.iloc[start - p.lookback - 1:end]
        block_pnl = pd.DataFrame(0.0, index=prices.index[start:end],
                                 columns=["gross_pnl", "cost", "net_pnl"])
        block_pnl["n_pairs"] = len(selected)
        block_pnl["in_market"] = 0
        for a_col, b_col, eg in selected:
            pp = Params(**{**p.__dict__, "capital": p.capital / len(selected)})
            frame = window[[a_col, b_col]].set_axis(["asset_a", "asset_b"], axis=1)
            d, tr = backtest(frame, eg["alpha"], eg["beta"], pp, start=p.lookback + 1)
            block_pnl[["gross_pnl", "cost", "net_pnl"]] += d[["gross_pnl", "cost", "net_pnl"]].values
            block_pnl["in_market"] |= (d["position"] != 0).astype(int).values
            if len(tr):
                trade_parts.append(tr.assign(pair=f"{a_col}/{b_col}"))
        daily_parts.append(block_pnl)

    daily = pd.concat(daily_parts)
    daily["position"] = daily["in_market"]
    daily["cum_net"] = daily["net_pnl"].cumsum()
    daily["cum_gross"] = daily["gross_pnl"].cumsum()
    trades = pd.concat(trade_parts, ignore_index=True) if trade_parts else pd.DataFrame(
        columns=["entry_date", "exit_date", "direction", "entry_z", "days", "gross",
                 "cost", "exit_reason", "net", "pair"])
    return daily, trades, pd.DataFrame(blocks)


def single_split(prices: pd.DataFrame, p: Params, train_frac: float = 0.6):
    """Classic one-shot out-of-sample test = walk-forward with one block."""
    train_days = int(len(prices) * train_frac)
    return walk_forward(prices, p, train_days=train_days, test_days=len(prices) - train_days)


def cost_sensitivity(prices, p, bps=(0, 1, 2, 5, 10), **wf):
    rows = []
    for c in bps:
        d, t, _ = walk_forward(prices, Params(**{**p.__dict__, "cost_bps": c}), **wf)
        rows.append({"cost_bps": c, **summarize(d, t, p.capital)})
    return pd.DataFrame(rows)


def parameter_grid(prices, p, lookbacks=(20, 30, 60, 90), entries=(1.5, 2.0, 2.5, 3.0), **wf):
    rows = []
    for lb in lookbacks:
        for ez in entries:
            d, t, _ = walk_forward(prices, Params(**{**p.__dict__, "lookback": lb, "entry_z": ez}), **wf)
            rows.append({"lookback": lb, "entry_z": ez, **summarize(d, t, p.capital)})
    return pd.DataFrame(rows)


def md_table(df: pd.DataFrame, index: bool = False) -> str:
    """Minimal markdown table (avoids a dependency on `tabulate`)."""
    if index:
        df = df.reset_index()
    fmt = lambda v: f"{v:.3f}".rstrip("0").rstrip(".") if isinstance(v, float) else str(v)  # noqa: E731
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "---|" * len(df.columns)
    rows = ["| " + " | ".join(fmt(v) for v in r) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, sep, *rows])


def write_summary(out: Path, label: str, wf_stats: dict, split_stats: dict,
                  costs: pd.DataFrame, grid: pd.DataFrame, blocks: pd.DataFrame, p: Params):
    pct = lambda x: f"{x:.1%}" if pd.notna(x) else "n/a"  # noqa: E731
    num = lambda x: f"{x:.2f}" if pd.notna(x) else "n/a"  # noqa: E731
    traded_share = blocks.groupby("block_start")["traded"].any().mean()
    lines = [
        f"# Results: {label}",
        "",
        f"Default parameters: lookback {p.lookback}, entry |z| {p.entry_z}, exit |z| {p.exit_z}, "
        f"stop |z| {p.stop_z}, cost {p.cost_bps} bps per side, execution at next close.",
        "",
        "| Metric | Walk-forward | Single 60/40 split |",
        "|---|---|---|",
    ]
    for k, f in [("net_return", pct), ("annual_return", pct), ("annual_vol", pct),
                 ("sharpe", num), ("t_stat_mean", num), ("max_drawdown", pct),
                 ("trades", str), ("win_rate", pct), ("profit_factor", num),
                 ("avg_hold_days", num), ("stop_outs", str), ("costs", pct), ("years", num)]:
        lines.append(f"| {k} | {f(wf_stats[k])} | {f(split_stats[k])} |")
    lines += ["", f"Share of walk-forward blocks where at least one pair passed the "
              f"cointegration screen: {traded_share:.0%}", "",
              "## Cost sensitivity (walk-forward)", "",
              md_table(costs[["cost_bps", "net_return", "sharpe", "trades"]]), "",
              "## Parameter stability (walk-forward Sharpe)", "",
              md_table(grid.pivot(index="lookback", columns="entry_z", values="sharpe").round(2), index=True), "",
              "A t-stat below ~2 means the mean daily return is not statistically "
              "distinguishable from zero over this sample."]
    (out / "summary.md").write_text("\n".join(lines))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group()
    src.add_argument("--tickers", nargs="+", default=["EURUSD=X", "GBPUSD=X"])
    src.add_argument("--csv")
    src.add_argument("--synthetic", action="store_true")
    ap.add_argument("--cols", nargs="+", help="with --csv: which price columns to use")
    ap.add_argument("--start", help="first date to use, e.g. 2016-01-01")
    ap.add_argument("--period", default="10y")
    ap.add_argument("--train-days", type=int, default=504)
    ap.add_argument("--test-days", type=int, default=63)
    ap.add_argument("--cost-bps", type=float, default=2.0)
    ap.add_argument("--lookback", type=int, default=30)
    ap.add_argument("--entry-z", type=float, default=2.0)
    ap.add_argument("--exit-z", type=float, default=0.5)
    ap.add_argument("--stop-z", type=float, default=4.0)
    ap.add_argument("--name")
    args = ap.parse_args(argv)

    if args.synthetic:
        prices, label = load_synthetic(), "synthetic regime-switching demo (not market data)"
    elif args.csv:
        prices = pd.read_csv(args.csv, index_col=0, parse_dates=True)
        if args.cols:
            prices = prices[args.cols]
        if args.start:
            prices = prices.loc[args.start:]
        prices = clean_prices(prices.select_dtypes("number"))
        label = f"{' / '.join(prices.columns)} from {Path(args.csv).name}, " \
                f"{prices.index[0]:%Y-%m-%d} to {prices.index[-1]:%Y-%m-%d}"
    else:
        prices, label = load_yahoo(args.tickers, args.period), " / ".join(args.tickers) + f" ({args.period}, Yahoo Finance)"
    name = args.name or ("synthetic" if args.synthetic else "_".join(c.replace("=X", "") for c in prices.columns))
    out = ROOT / "results" / name
    out.mkdir(parents=True, exist_ok=True)

    p = Params(lookback=args.lookback, entry_z=args.entry_z, exit_z=args.exit_z,
               stop_z=args.stop_z, cost_bps=args.cost_bps)
    wf = dict(train_days=args.train_days, test_days=args.test_days)

    daily, trades, blocks = walk_forward(prices, p, **wf)
    s_daily, s_trades, _ = single_split(prices, p)
    wf_stats, split_stats = summarize(daily, trades), summarize(s_daily, s_trades)
    costs = cost_sensitivity(prices, p, **wf)
    grid = parameter_grid(prices, p, **wf)

    prices.to_csv(out / "prices.csv")
    daily.to_csv(out / "walk_forward_daily.csv")
    trades.to_csv(out / "walk_forward_trades.csv", index=False)
    blocks.to_csv(out / "walk_forward_blocks.csv", index=False)
    s_daily.to_csv(out / "single_split_daily.csv")
    costs.to_csv(out / "cost_sensitivity.csv", index=False)
    grid.to_csv(out / "parameter_grid.csv", index=False)
    (out / "stats.json").write_text(json.dumps({"label": label, "walk_forward": wf_stats,
                                                "single_split": split_stats}, indent=2))
    write_summary(out, label, wf_stats, split_stats, costs, grid, blocks, p)

    from plot_research import plot_run
    plot_run(out)
    print((out / "summary.md").read_text())
    print(f"\nSaved to {out}")


if __name__ == "__main__":
    main()
