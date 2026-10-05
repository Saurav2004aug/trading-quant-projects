"""Figures for the synthetic validation (../plots). Real-data figures are
written next to each run by research.py."""
from pathlib import Path

import numpy as np
import pandas as pd

import plot_style as ps
import matplotlib.pyplot as plt
from cointegration import critical_values, engle_granger_test
from data_gen import load_cointegrated_pair, load_uncorrelated_pair
from engine import Params, backtest, summarize

PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)


def eg_size_simulation(n_sims: int = 1000, n: int = 500, seed: int = 0) -> np.ndarray:
    """Engle-Granger t-stats for pairs of *independent* random walks."""
    rng = np.random.default_rng(seed)
    stats = np.empty(n_sims)
    for i in range(n_sims):
        a = pd.Series(np.cumsum(rng.standard_normal(n)))
        b = pd.Series(np.cumsum(rng.standard_normal(n)))
        stats[i] = engle_granger_test(a, b)["adf"]["t_stat"]
    return stats


def plot_size_test():
    stats = eg_size_simulation()
    adf5 = critical_values(500, 1)["5%"]
    eg5 = critical_values(500, 2)["5%"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(stats, bins=50, color=ps.BLUE, alpha=0.8)
    for cv, name, color in [(adf5, "plain ADF 5%", ps.ORANGE), (eg5, "Engle-Granger 5%", ps.AQUA)]:
        rate = (stats < cv).mean()
        ax.axvline(cv, color=color, linestyle="--", linewidth=1.5,
                   label=f"{name} critical value ({cv:.2f}): flags {rate:.1%} as cointegrated")
    ax.set_title("1,000 pairs of independent random walks: false-positive rate")
    ax.set_xlabel("ADF t-statistic of regression residual")
    ax.set_ylabel("Count")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "eg_size_test.png")
    plt.close(fig)


def plot_cointegration_check():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, df, title in zip(axes, [load_cointegrated_pair(), load_uncorrelated_pair()],
                             ["Cointegrated by construction", "Independent random walks"]):
        res = engle_granger_test(df.asset_a, df.asset_b)
        ax.plot(df.index, df.asset_a, color=ps.BLUE, label="Asset A")
        ax.plot(df.index, df.asset_b, color=ps.ORANGE, label="Asset B")
        ax.set_title(f"{title}\nEG t = {res['adf']['t_stat']:.2f} "
                     f"(5% cv {res['adf']['critical_values']['5%']:.2f})")
        ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "cointegration_check.png")
    plt.close(fig)


def synthetic_backtest():
    data = load_cointegrated_pair()
    split = len(data) // 2
    eg = engle_granger_test(data.asset_a.iloc[:split], data.asset_b.iloc[:split])
    p = Params()
    daily, trades = backtest(data, eg["alpha"], eg["beta"], p, start=split)
    return daily, trades, eg, p


def plot_spread_and_equity():
    daily, trades, eg, p = synthetic_backtest()
    stats = summarize(daily, trades)
    fig, axes = plt.subplots(2, 1, figsize=(10, 6.2), sharex=True)
    ax = axes[0]
    ax.plot(daily.index, daily.zscore, color=ps.BLUE, linewidth=1.3)
    for lvl, style in [(p.entry_z, "--"), (p.exit_z, ":")]:
        ax.axhline(lvl, color=ps.MUTED, linestyle=style, linewidth=1)
        ax.axhline(-lvl, color=ps.MUTED, linestyle=style, linewidth=1)
    longs = trades[trades.direction == "long"].entry_date
    shorts = trades[trades.direction == "short"].entry_date
    ax.plot(longs, daily.zscore.reindex(longs), "^", color=ps.AQUA, markersize=8, label="Enter long spread")
    ax.plot(shorts, daily.zscore.reindex(shorts), "v", color=ps.ORANGE, markersize=8, label="Enter short spread")
    ax.set_title(f"Out-of-sample z-score  |  spread = B {'-' if eg['alpha'] >= 0 else '+'} {abs(eg['alpha']):.2f} - {eg['beta']:.3f} A "
                 f"(fitted on first half only)")
    ax.legend(loc="upper left", ncol=2)
    ax = axes[1]
    ax.plot(daily.index, daily.cum_gross * 100, color=ps.MUTED, linewidth=1.4, label="Before costs")
    ax.plot(daily.index, daily.cum_net * 100, color=ps.BLUE, label="After 2 bps per side")
    ax.axhline(0, color=ps.INK, linewidth=0.8)
    ax.set_ylabel("Return on gross capital (%)")
    ax.set_title(f"Equity  |  Sharpe {stats['sharpe']:.2f}, max DD {stats['max_drawdown']:.1%}, "
                 f"{stats['trades']} trades")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "synthetic_backtest.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_size_test()
    plot_cointegration_check()
    plot_spread_and_equity()
    daily, trades, _, _ = synthetic_backtest()
    for k, v in summarize(daily, trades).items():
        print(f"{k:>16}: {v:.4f}" if isinstance(v, float) else f"{k:>16}: {v}")
    print("Plots written to", PLOTS)
