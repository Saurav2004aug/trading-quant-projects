"""Regenerate ../plots (a few seconds)."""
from pathlib import Path

import numpy as np
import pandas as pd

import plot_style as ps
import matplotlib.pyplot as plt
from study import FEATURES, ols_predict, run

PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)


def hhmm(seconds):
    return [f"{int(s // 3600):02d}:{int(s % 3600 // 60):02d}" for s in seconds]


def plot_day(o):
    g = o["grid"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"width_ratios": [2, 1]})
    ax = axes[0]
    ax.plot(g.time, g.mid, color=ps.BLUE, linewidth=1)
    ax.axvline(g.time.median(), color=ps.MUTED, linestyle="--", linewidth=1)
    ax.text(g.time.median(), 0.97, "  fit | test", transform=ax.get_xaxis_transform(), va="top", color=ps.INK_2)
    ticks = np.arange(34200 + 1800, 57600, 3600)
    ax.set_xticks(ticks, hhmm(ticks))
    ax.set_ylabel("Mid price ($)")
    ax.set_title("AAPL mid price, 21 June 2012")
    ax = axes[1]
    ax.hist(g.spread_ticks, bins=np.arange(0.5, 45.5, 1), color=ps.BLUE, edgecolor=ps.SURFACE)
    ax.axvline(g.spread_ticks.median(), color=ps.ORANGE, linewidth=1.5)
    ax.text(g.spread_ticks.median(), 0.95, f" median {g.spread_ticks.median():.0f}", color=ps.INK,
            transform=ax.get_xaxis_transform())
    ax.set_xlabel("Bid-ask spread (ticks of $0.01)")
    ax.set_title("A wide spread to cross")
    fig.tight_layout()
    fig.savefig(PLOTS / "day_overview.png")
    plt.close(fig)


def plot_explain_vs_predict(o):
    test, cont, pred = o["test"], o["cont"], o["pred"]
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.2), gridspec_kw={"width_ratios": [1, 1, 1.1]})
    sample = test.sample(4000, random_state=0)
    ax = axes[0]
    ax.scatter(sample.ofi_next_10s / 1000, sample.dmid_10s, s=5, alpha=0.35, color=ps.BLUE, linewidths=0)
    x = np.linspace(-8, 8, 10)
    b = cont.loc[cont.horizon_s == 10, "ticks_per_1000_shares"].iloc[0]
    ax.plot(x, b * x, color=ps.ORANGE)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-60, 60)
    ax.set_xlabel("Order-flow imbalance, same 10 s (1000 shares)")
    ax.set_ylabel("Mid change over 10 s (ticks)")
    ax.set_title(f"Explains: R² {cont.loc[cont.horizon_s == 10, 'r2_test'].iloc[0]:.2f} (test)")
    ax = axes[1]
    p = ols_predict(o["models"][10], sample[FEATURES].to_numpy())
    ax.scatter(p, sample.dmid_10s, s=5, alpha=0.35, color=ps.BLUE, linewidths=0)
    ax.set_xlim(-8, 8)
    ax.set_ylim(-60, 60)
    ax.set_xlabel("Predicted mid change from current book (ticks)")
    ax.set_title(f"Predicts: R² {pred.loc[pred.horizon_s == 10, 'r2_test'].iloc[0]:.3f} (test)")
    ax = axes[2]
    ax.plot(cont.horizon_s, cont.r2_test, "o-", color=ps.ORANGE, label="Same-window order flow (explains)")
    ax.plot(pred.horizon_s, pred.r2_test.clip(lower=0), "o-", color=ps.BLUE, label="Book state now (predicts)")
    ax.set_xscale("log")
    ax.set_xticks(cont.horizon_s, [f"{h}s" for h in cont.horizon_s])
    ax.set_ylabel("Out-of-sample R²")
    ax.set_xlabel("Horizon")
    ax.set_title("Explaining is easy, predicting is not")
    ax.legend(loc="center right")
    fig.tight_layout()
    fig.savefig(PLOTS / "explain_vs_predict.png")
    plt.close(fig)


def plot_event_time(o):
    et = o["event_time"]
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.plot(et.bucket_mid, et.p_up * 100, "o-", color=ps.BLUE, label="All book states")
    ax.axhline(50, color=ps.MUTED, linewidth=1)
    ax.set_xlabel("Level-1 imbalance (bid size - ask size) / (bid size + ask size)")
    ax.set_ylabel("P(next mid change is up) %")
    ax.set_title("The next tick is predictable, a little (afternoon, out of sample)")
    for x, y, n in zip(et.bucket_mid, et.p_up * 100, et.n):
        ax.annotate(f"{n / 1000:.0f}k", (x, y), xytext=(0, 7), textcoords="offset points",
                    ha="center", fontsize=7, color=ps.MUTED)
    ax.text(0.99, 0.03, "labels: number of events per bucket", transform=ax.transAxes, ha="right",
            fontsize=7, color=ps.MUTED)
    fig.tight_layout()
    fig.savefig(PLOTS / "next_move_probability.png")
    plt.close(fig)


def plot_economics(o):
    tr, adv = o["trades"], o["adverse"]
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))
    ax = axes[0]
    labels = [f"{int(h)}s, {'top 10%' if i % 2 == 0 else 'top 1%'}\n({n} trades)"
              for i, (h, n) in enumerate(zip(tr.horizon_s, tr.trades))]
    x = np.arange(len(tr))
    ax.bar(x - 0.2, tr.gross_at_mid_ticks, width=0.4, color=ps.BLUE, label="Signal: move in predicted direction")
    ax.bar(x + 0.2, tr.spread_cost_ticks + tr.fees_ticks, width=0.4, color=ps.ORANGE,
           label="Cost: cross spread twice + fees")
    ax.set_xticks(x, labels, fontsize=7.5)
    ax.set_ylabel("Ticks per share")
    ratio = (tr.spread_cost_ticks + tr.fees_ticks) / tr.gross_at_mid_ticks
    ax.set_title(f"Taking liquidity: cost is {ratio.min():.0f}-{ratio.max():.0f}x the signal")
    ax.set_ylim(0, (tr.spread_cost_ticks + tr.fees_ticks).max() * 1.35)
    ax.legend(loc="upper left", ncol=1)
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    ax.plot(adv.seconds_after_trade, adv.mean_move_ticks, "o-", color=ps.BLUE,
            label="Mid move after a trade, aggressor's direction")
    ax.axhline(adv.half_spread_ticks.iloc[0], color=ps.ORANGE, linestyle="--",
               label="Half-spread a market maker earns")
    ax.set_xscale("log")
    ax.set_xticks(adv.seconds_after_trade, [f"{s:g}s" for s in adv.seconds_after_trade])
    ax.set_ylim(0, 7)
    ax.set_ylabel("Ticks")
    ax.set_xlabel("Time after trade")
    ax.set_title("Providing liquidity: adverse selection eats the spread")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "trading_economics.png")
    plt.close(fig)


if __name__ == "__main__":
    o = run()
    plot_day(o)
    plot_explain_vs_predict(o)
    plot_event_time(o)
    plot_economics(o)
    print("Plots written to", PLOTS)
