"""Regenerate ../plots from the analysis (about 30 seconds)."""
from pathlib import Path

import numpy as np
import pandas as pd

import plot_style as ps
import matplotlib.pyplot as plt
from analysis import run
from real_straddle import summarise

PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)


def plot_implied_vs_realised(o):
    df, comp = o["df"].loc["2019-12-01":], o["comp"]
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    ax.plot(df.index, df.rv_fwd, color=ps.AQUA, linewidth=1.3, label="Realised vol, next 21 days")
    ax.plot(df.index, df.vix, color=ps.ORANGE, linewidth=1.3, label="India VIX")
    ax.plot(comp.index, comp.atm_iv30, color=ps.BLUE, linewidth=1.3, label="Real ATM IV, 30-day (NSE options)")
    for date, text in [("2020-03-05", "COVID crash"), ("2024-06-04", "Election result"),
                       ("2022-02-24", "Russia-Ukraine")]:
        ax.annotate(text, (pd.Timestamp(date), 56), ha="left", color=ps.INK_2, fontsize=8)
        ax.axvline(pd.Timestamp(date), color=ps.MUTED, linestyle=":", linewidth=1)
    ax.set_ylim(0, 60)
    ax.set_ylabel("Annualised volatility (%)")
    ax.set_title("Implied volatility usually sits above the volatility that follows, until it doesn't")
    ax.legend(loc="upper right", bbox_to_anchor=(1, 0.9))
    ax.text(0.01, 0.02, "y-axis clipped at 60%; VIX and realised vol both exceeded 80% in March 2020",
            transform=ax.transAxes, color=ps.MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(PLOTS / "implied_vs_realised.png")
    plt.close(fig)


def plot_premium(o):
    df = o["df"].dropna(subset=["vix", "rv_fwd"])
    vs = o["vrp"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    ax = axes[0]
    ax.scatter(df.vix, df.rv_fwd, s=6, alpha=0.35, color=ps.BLUE, linewidths=0)
    lim = 90
    x = np.linspace(8, lim, 50)
    ax.plot(x, x, color=ps.MUTED, linestyle="--", linewidth=1, label="Realised = VIX")
    ax.plot(x, vs["mz_vix_a"] + vs["mz_vix_b"] * x, color=ps.ORANGE,
            label=f"Fit: RV = {vs['mz_vix_a']:.1f} + {vs['mz_vix_b']:.2f} x VIX (R² {vs['mz_vix_r2']:.2f})")
    ax.set_xlim(8, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("India VIX (%)")
    ax.set_ylabel("Realised vol, next 21 days (%)")
    ax.set_title("Most points sit below the diagonal")
    ax.legend(loc="upper left")
    ax = axes[1]
    ax.hist(df.vrp_vol.clip(-30, 20), bins=np.arange(-30, 20.5, 1), color=ps.BLUE, alpha=0.85,
            edgecolor=ps.SURFACE)
    ax.axvline(0, color=ps.INK, linewidth=0.8)
    ax.axvline(vs["median_vrp_vol"], color=ps.ORANGE, linewidth=1.5)
    ax.text(vs["median_vrp_vol"], 0.95, f" median +{vs['median_vrp_vol']:.1f}", color=ps.INK,
            transform=ax.get_xaxis_transform())
    ax.text(-29.5, 0.95, f"worst {vs['worst_vrp_vol']:.0f}\n({vs['worst_vrp_date']})\nclipped at -30",
            color=ps.INK_2, va="top", transform=ax.get_xaxis_transform(), fontsize=8)
    ax.set_xlabel("VIX minus subsequent realised vol (vol points)")
    ax.set_ylabel("Days")
    ax.set_title(f"Premium positive on {vs['share_vix_above_rv']:.0%} of days, with a fat left tail")
    fig.tight_layout()
    fig.savefig(PLOTS / "premium_distribution.png")
    plt.close(fig)


def plot_real_straddle(o):
    h, u = o["real_h"], o["real_u"]
    s = summarise(h)
    fig, axes = plt.subplots(2, 1, figsize=(10.5, 6.2), sharex=True, gridspec_kw={"height_ratios": [1.2, 1]})
    ax = axes[0]
    colors = [ps.BLUE if v >= 0 else ps.ORANGE for v in h.pnl_pct]
    ax.bar(h.expiry, h.pnl_pct, width=20, color=colors, linewidth=0)
    ax.axhline(0, color=ps.INK, linewidth=0.8)
    worst = h.loc[h.pnl_pct.idxmin()]
    ax.annotate(f"Feb 2022 (Russia-Ukraine): {worst.pnl_pct:.1f}%", (worst.expiry, worst.pnl_pct),
                xytext=(12, -2), textcoords="offset points", color=ps.INK_2, fontsize=8)
    ax.set_ylabel("P&L per cycle (% of forward)")
    ax.set_title(f"Short monthly ATM straddle at real NSE prices, delta-hedged: {s['cycles']} cycles, "
                 f"mean {s['mean_pnl_pct']:+.2f}%, t = {s['t_stat']:.2f}, win rate {s['win_rate']:.0%}")
    ax = axes[1]
    ax.plot(h.expiry, h.pnl_pct.cumsum(), color=ps.BLUE, label="Delta-hedged daily")
    ax.plot(u.expiry, u.pnl_pct.cumsum(), color=ps.MUTED, label="Unhedged")
    ax.axhline(0, color=ps.INK, linewidth=0.8)
    ax.set_ylabel("Cumulative (% of forward)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "real_straddle.png")
    plt.close(fig)


def plot_sensitivity(o):
    hc, gap = o["haircut"], o["gap"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    ax = axes[0]
    ax.plot(hc.atm_iv_below_vix_pts, hc.sharpe, "o-", color=ps.BLUE)
    ax.axvspan(gap.quantile(0.1), gap.quantile(0.9), color=ps.AQUA, alpha=0.15, linewidth=0,
               label="Measured VIX - ATM IV (10th-90th pct)")
    ax.axvline(gap.mean(), color=ps.AQUA, linewidth=1.5)
    ax.axhline(0, color=ps.INK, linewidth=0.8)
    ax.set_xlabel("ATM implied vol assumed below India VIX (vol points)")
    ax.set_ylabel("Model Sharpe (2020-2026)")
    ax.set_title("Pricing at VIX overstates the edge")
    ax.set_ylim(hc.sharpe.min() - 0.1, hc.sharpe.max() + 0.35)
    ax.legend(loc="upper right")
    ax = axes[1]
    rc = o["real_costs"]
    labels = [f"{a:g}% prem\n{b:g} bps" for a, b in zip(rc.opt_cost_pct_of_premium, rc.hedge_bps)]
    ax.bar(labels, rc.mean_pnl_pct, color=ps.BLUE, width=0.55)
    for i, (m, t) in enumerate(zip(rc.mean_pnl_pct, rc.t_stat)):
        ax.text(i, m + 0.01, f"t = {t:.1f}", ha="center", color=ps.INK_2)
    ax.set_ylim(0, rc.mean_pnl_pct.max() * 1.15)
    ax.set_ylabel("Mean P&L per cycle (% of forward)")
    ax.set_title("Real-price straddle vs trading costs")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(PLOTS / "sensitivity.png")
    plt.close(fig)


if __name__ == "__main__":
    o = run()
    plot_implied_vs_realised(o)
    plot_premium(o)
    plot_real_straddle(o)
    plot_sensitivity(o)
    print("Plots written to", PLOTS)
