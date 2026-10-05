"""Figures for a research run folder written by research.py."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

import plot_style as ps
import matplotlib.pyplot as plt


def plot_run(run_dir: Path) -> None:
    run_dir = Path(run_dir)
    label = json.loads((run_dir / "stats.json").read_text())["label"]
    prices = pd.read_csv(run_dir / "prices.csv", index_col=0, parse_dates=True)
    daily = pd.read_csv(run_dir / "walk_forward_daily.csv", index_col=0, parse_dates=True)
    split = pd.read_csv(run_dir / "single_split_daily.csv", index_col=0, parse_dates=True)
    blocks = pd.read_csv(run_dir / "walk_forward_blocks.csv", parse_dates=["block_start"])
    costs = pd.read_csv(run_dir / "cost_sensitivity.csv")
    grid = pd.read_csv(run_dir / "parameter_grid.csv")

    # 1. Prices, normalised to 100
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, c in enumerate(prices.columns[:8]):
        ax.plot(prices.index, prices[c] / prices[c].iloc[0] * 100, color=ps.SERIES[i],
                linewidth=1.4, label=c)
    ax.set_title(f"Prices rebased to 100  |  {label}")
    ax.legend(ncol=min(4, len(prices.columns)))
    fig.tight_layout()
    fig.savefig(run_dir / "prices.png")
    plt.close(fig)

    # 2. Equity: walk-forward gross vs net, single split net, and where the screen passed
    fig, axes = plt.subplots(2, 1, figsize=(10, 6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    ax.plot(daily.index, daily.cum_gross * 100, color=ps.MUTED, linewidth=1.4,
            label="Walk-forward, before costs")
    ax.plot(daily.index, daily.cum_net * 100, color=ps.BLUE, label="Walk-forward, after costs")
    ax.plot(split.index, split.cum_net * 100, color=ps.ORANGE, linewidth=1.4,
            label="Single 60/40 split, after costs")
    ax.axhline(0, color=ps.INK, linewidth=0.8)
    ax.set_ylabel("Cumulative return (%)")
    ax.set_title("Out-of-sample equity")
    ax.legend(loc="upper left")
    ax = axes[1]
    passed = blocks.groupby("block_start")["traded"].any()
    width = passed.index.to_series().diff().dt.days.median() if len(passed) > 1 else 90
    ax.bar(passed.index, passed.astype(int), width=width, color=ps.AQUA, align="edge", linewidth=0)
    ax.set_yticks([0, 1], ["no", "yes"])
    ax.set_title("Cointegrated in training window? (traded only when yes)")
    fig.tight_layout()
    fig.savefig(run_dir / "equity.png")
    plt.close(fig)

    # 3. Cost sensitivity and parameter stability
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    axes[0].plot(costs.cost_bps, costs.sharpe, "o-", color=ps.BLUE)
    axes[0].axhline(0, color=ps.INK, linewidth=0.8)
    axes[0].set_xlabel("Cost per side (bps of notional)")
    axes[0].set_ylabel("Walk-forward Sharpe")
    axes[0].set_title("Cost sensitivity")
    pivot = grid.pivot(index="lookback", columns="entry_z", values="sharpe")
    lim = max(abs(pivot.min().min()), abs(pivot.max().max()), 0.5)
    im = axes[1].imshow(pivot.values, cmap="RdBu", vmin=-lim, vmax=lim, aspect="auto")
    axes[1].set_xticks(range(len(pivot.columns)), [str(c) for c in pivot.columns])
    axes[1].set_yticks(range(len(pivot.index)), [str(i) for i in pivot.index])
    axes[1].set_xlabel("Entry |z|")
    axes[1].set_ylabel("Lookback (days)")
    axes[1].set_title("Parameter stability (Sharpe)")
    axes[1].grid(False)
    for i in range(pivot.shape[0]):
        for j in range(pivot.shape[1]):
            v = pivot.iloc[i, j]
            axes[1].text(j, i, f"{v:.2f}", ha="center", va="center",
                         color="white" if abs(v) > 0.6 * lim else ps.INK)
    fig.colorbar(im, ax=axes[1])
    fig.tight_layout()
    fig.savefig(run_dir / "robustness.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_run(Path(sys.argv[1]))
