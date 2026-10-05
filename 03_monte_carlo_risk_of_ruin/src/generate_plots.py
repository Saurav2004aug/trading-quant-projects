"""Regenerate every figure in ../plots."""
from pathlib import Path

import numpy as np

import plot_style as ps
import matplotlib.pyplot as plt
from prop_firm import ChallengeRules, challenge_sweep
from risk_of_ruin import sweep
from simulate import exact_median_final, kelly_fraction, simulate_equity_paths

PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)
P, B, N = 0.45, 1.5, 240
K = kelly_fraction(P, B)
EDGE = f"win rate {P:.0%}, payoff {B}R, {N} trades"


def mark_kelly(ax):
    for x, name in [(K / 2, "½ Kelly"), (K, "Kelly")]:
        ax.axvline(x * 100, color=ps.MUTED, linestyle="--", linewidth=1)
        ax.text(x * 100, 1.0, f" {name}", transform=ax.get_xaxis_transform(),
                va="top", color=ps.INK_2)


def plot_sample_paths():
    paths = simulate_equity_paths(N, 2000, P, B, 0.01, seed=3)
    fig, ax = plt.subplots(figsize=(9, 4.8))
    for p in paths[:40]:
        ax.plot(p, color=ps.BLUE, alpha=0.25, linewidth=0.8)
    lo, med, hi = np.quantile(paths, [0.05, 0.5, 0.95], axis=0)
    ax.fill_between(range(N + 1), lo, hi, color=ps.BLUE, alpha=0.08, linewidth=0,
                    label="5th-95th percentile of 2,000 paths")
    ax.plot(med, color=ps.ORANGE, label="Median path")
    ax.axhline(1.0, color=ps.INK, linewidth=0.8)
    ax.set_title(f"Equity paths at 1% risk per trade  |  {EDGE}")
    ax.set_xlabel("Trade #")
    ax.set_ylabel("Equity (start = 1.0)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "sample_equity_paths.png")
    plt.close(fig)


def plot_growth_and_risk(df):
    fine = np.linspace(0.001, df.risk_frac.max(), 400)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.plot(fine * 100, exact_median_final(fine, P, B, N), color=ps.BLUE, label="Median (exact)")
    ax.plot(df.risk_frac * 100, df.median_final, "o", color=ps.BLUE, markersize=4,
            label="Median (simulated)")
    ax.plot(df.risk_frac * 100, df.mean_final, color=ps.ORANGE, linestyle="--", label="Mean (simulated)")
    ax.axhline(1, color=ps.INK, linewidth=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("Risk per trade (% of current equity)")
    ax.set_ylabel("Final equity multiple (log)")
    ax.set_title("Median growth peaks at Kelly, then collapses")
    ax.legend(loc="lower left")
    mark_kelly(ax)
    ax = axes[1]
    ax.plot(df.risk_frac * 100, df.p_dd_ge_20pct * 100, color=ps.BLUE, label="Max drawdown ≥ 20%")
    ax.plot(df.risk_frac * 100, df.p_ruin_50pct * 100, color=ps.ORANGE, label="Equity ever ≤ 50% of start")
    ax.plot(df.risk_frac * 100, df.p_loss * 100, color=ps.AQUA, label="Finish below start")
    ax.set_xlabel("Risk per trade (% of current equity)")
    ax.set_ylabel("Probability (%)")
    ax.set_title("Risk rises long before growth peaks")
    ax.legend(loc="center right")
    mark_kelly(ax)
    fig.suptitle(f"Position sizing sweep  |  {EDGE}", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "growth_vs_risk.png")
    plt.close(fig)


def plot_challenge():
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4), sharey=True)
    for ax, dd in zip(axes, ("static", "trailing")):
        df = challenge_sweep(rules=ChallengeRules(drawdown_type=dd))
        x = df.risk_per_trade * 100
        parts = [(df["pass"], "Pass", ps.BLUE), (df.fail_max_loss, "Fail: max loss", ps.ORANGE),
                 (df.fail_daily_loss, "Fail: daily loss", ps.YELLOW), (df.timeout, "Timeout", "#c9c8c2")]
        ax.stackplot(x, *[p[0] * 100 for p in parts], labels=[p[1] for p in parts],
                     colors=[p[2] for p in parts], edgecolor=ps.SURFACE, linewidth=0.6)
        best = df.loc[df["pass"].idxmax()]
        ax.annotate(f"best {best.risk_per_trade:.2%}: pass {best['pass']:.0%}",
                    (best.risk_per_trade * 100, best["pass"] * 100), xytext=(12, 8),
                    textcoords="offset points", color=ps.INK,
                    arrowprops=dict(arrowstyle="-", color=ps.INK, linewidth=0.8))
        ax.set_xlim(x.min(), x.max())
        ax.set_ylim(0, 100)
        ax.set_title(f"{dd.capitalize()} 10% max loss, 5% daily limit")
        ax.set_xlabel("Risk per trade (% of initial balance)")
        ax.grid(False)
    axes[0].set_ylabel("Share of simulated challenges (%)")
    axes[1].legend(loc="upper right", frameon=True, facecolor=ps.SURFACE, edgecolor=ps.SURFACE)
    fig.suptitle("Prop-firm challenge outcomes: +10% target in 30 days, 3 trades/day, "
                 f"win rate {P:.0%}, payoff {B}R", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "prop_challenge_outcomes.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_sample_paths()
    plot_growth_and_risk(sweep(P, B, N))
    plot_challenge()
    print("Plots written to", PLOTS)
