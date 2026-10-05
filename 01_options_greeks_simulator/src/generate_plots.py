"""Regenerate every figure in ../plots. Run from any directory."""
from pathlib import Path

import numpy as np

import plot_style as ps
import matplotlib.pyplot as plt
from binomial import crr_price
from black_scholes import all_greeks, price
from delta_hedge_sim import frequency_study, run_hedge_simulation, theoretical_mean_pnl

PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)


def plot_greeks_vs_spot():
    K, T, r, sigma = 100, 30 / 365, 0.05, 0.20
    spots = np.linspace(70, 130, 241)
    call = all_greeks(spots, K, T, r, sigma, "call")
    put = all_greeks(spots, K, T, r, sigma, "put")
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    labels = {"delta": "Delta", "gamma": "Gamma", "vega": "Vega (per vol pt)",
              "theta": "Theta (per day)"}
    for ax, key in zip(axes.flat, labels):
        ax.plot(spots, call[key], color=ps.BLUE, label="Call")
        if key in ("delta", "theta"):
            ax.plot(spots, put[key], color=ps.ORANGE, label="Put")
        ax.axvline(K, color=ps.MUTED, linestyle="--", linewidth=1)
        ax.set_title(labels[key])
        ax.set_xlabel("Spot")
    axes[0, 0].legend()
    fig.suptitle("Greeks vs spot  |  30-day option, K = 100, implied vol 20%",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "greeks_vs_spot.png")
    plt.close(fig)


def plot_hedge_pnl_distribution():
    fig, ax = plt.subplots(figsize=(8, 4.6))
    for n, label, color in [(30, "Daily (30 rebalances)", ps.BLUE),
                            (4, "Weekly (4 rebalances)", ps.ORANGE)]:
        df = run_hedge_simulation(n_rebalances=n, n_paths=10000, seed=1)
        ax.hist(df.pnl, bins=70, range=(-4, 2), alpha=0.55, density=True, color=color,
                label=f"{label}: std {df.pnl.std():.2f}")
    ax.axvline(0, color=ps.INK, linewidth=0.8)
    ax.set_title("Hedged P&L of a short 30-day ATM call: discretisation risk")
    ax.set_xlabel("Final hedged P&L per option (implied = realised = 20%, no costs)")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "hedge_pnl_dist.png")
    plt.close(fig)


def plot_vol_mismatch():
    implied = 0.20
    grid = np.linspace(0.10, 0.35, 11)
    sims = [run_hedge_simulation(sigma_realised=rv, n_rebalances=120, n_paths=6000, seed=7)
            for rv in grid]
    means = np.array([d.pnl.mean() for d in sims])
    p05 = np.array([d.pnl.quantile(0.05) for d in sims])
    p95 = np.array([d.pnl.quantile(0.95) for d in sims])
    fine = np.linspace(0.10, 0.35, 200)
    theory = [theoretical_mean_pnl(sigma_realised=rv) for rv in fine]

    fig, ax = plt.subplots(figsize=(8, 4.6))
    ax.fill_between(grid * 100, p05, p95, color=ps.BLUE, alpha=0.15, linewidth=0,
                    label="Simulated 5th-95th percentile")
    ax.plot(fine * 100, theory, color=ps.ORANGE, label=r"Theory: $e^{rT}[C(\sigma_{imp})-C(\sigma_{real})]$")
    ax.plot(grid * 100, means, "o", color=ps.BLUE, markersize=6, label="Simulated mean")
    ax.axvline(implied * 100, color=ps.MUTED, linestyle="--", linewidth=1)
    ax.text(implied * 100 + 0.3, ax.get_ylim()[1] * 0.85, "sold at 20%", color=ps.INK_2)
    ax.axhline(0, color=ps.INK, linewidth=0.8)
    ax.set_title("Short-gamma P&L vs realised volatility (short call, delta-hedged)")
    ax.set_xlabel("Realised volatility (%)")
    ax.set_ylabel("Hedged P&L per option")
    ax.legend(loc="lower left")
    fig.tight_layout()
    fig.savefig(PLOTS / "vol_mismatch.png")
    plt.close(fig)


def plot_hedge_frequency():
    fs = frequency_study(cost_bps=5.0)
    best = fs.loc[fs.p05.idxmax()]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    ax = axes[0]
    ax.plot(fs.n_rebalances, fs.std_pnl, "o-", color=ps.BLUE, label="Std of P&L (hedging error)")
    ax.plot(fs.n_rebalances, fs.mean_cost, "s-", color=ps.ORANGE, label="Mean transaction cost")
    ax.set_xscale("log")
    ax.set_xlabel("Rebalances over 30 days (log)")
    ax.set_ylabel("Per option")
    ax.set_title("Hedging error falls, cost rises")
    ax.legend()
    ax = axes[1]
    ax.plot(fs.n_rebalances, fs.p05, "o-", color=ps.AQUA)
    ax.plot([best.n_rebalances], [best.p05], "o", markersize=11, markerfacecolor="none",
            markeredgecolor=ps.INK, markeredgewidth=1.5)
    ax.annotate(f"best: {int(best.n_rebalances)} rebalances", (best.n_rebalances, best.p05),
                xytext=(-60, -22), textcoords="offset points", color=ps.INK_2)
    ax.set_xscale("log")
    ax.set_xlabel("Rebalances over 30 days (log)")
    ax.set_title("5th-percentile P&L (risk-adjusted objective)")
    fig.suptitle("Optimal hedging frequency with 5 bps proportional cost",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "hedge_frequency_tradeoff.png")
    plt.close(fig)


def plot_american_put():
    K, T, r, sigma = 100, 1.0, 0.05, 0.25
    spots = np.linspace(60, 140, 41)
    am = np.array([crr_price(s, K, T, r, sigma, "put", True, n_steps=400) for s in spots])
    eu = price(spots, K, T, r, sigma, "put")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.3))
    axes[0].plot(spots, am, color=ps.BLUE, label="American put (CRR tree)")
    axes[0].plot(spots, eu, color=ps.ORANGE, label="European put (Black-Scholes)")
    axes[0].plot(spots, np.maximum(K - spots, 0), color=ps.MUTED, linestyle="--",
                 linewidth=1, label="Intrinsic value")
    axes[0].set_title("Put value vs spot (1y, r = 5%, vol 25%)")
    axes[0].set_xlabel("Spot")
    axes[0].legend()
    axes[1].plot(spots, am - eu, color=ps.BLUE)
    axes[1].set_title("Early-exercise premium")
    axes[1].set_xlabel("Spot")
    fig.tight_layout()
    fig.savefig(PLOTS / "american_put_premium.png")
    plt.close(fig)


if __name__ == "__main__":
    plot_greeks_vs_spot()
    plot_hedge_pnl_distribution()
    plot_vol_mismatch()
    plot_hedge_frequency()
    plot_american_put()
    print("Plots written to", PLOTS)
