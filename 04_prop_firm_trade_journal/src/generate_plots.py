"""Regenerate every figure in ../plots from data/trade_log.csv."""
from pathlib import Path

import numpy as np
import pandas as pd

import plot_style as ps
import matplotlib.pyplot as plt
from analyze import DATA, breakdown, equity_curve, overall_stats
from generate_trade_log import BASE_EDGE, INSTRUMENT_EDGE, SESSION_EDGE, SESSION_WEIGHTS
from loader import load_journal
from power import false_discovery_rate, power_curve

PLOTS = Path(__file__).resolve().parents[1] / "plots"
PLOTS.mkdir(exist_ok=True)


def plot_equity(df):
    s = overall_stats(df)
    eq = equity_curve(df)
    dd = eq / eq.cummax() - 1
    fig, axes = plt.subplots(2, 1, figsize=(9, 5.6), sharex=True,
                             gridspec_kw={"height_ratios": [3, 1.2]})
    axes[0].plot(df.entry_time, (eq - 1) * 100, color=ps.BLUE)
    axes[0].axhline(0, color=ps.INK, linewidth=0.8)
    axes[0].set_ylabel("Cumulative return (%)")
    axes[0].set_title(f"Account equity  |  {s['n_trades']} trades, expectancy {s['expectancy_r']:+.2f}R "
                      f"(95% CI {s['expectancy_ci_low']:+.2f} to {s['expectancy_ci_high']:+.2f}), "
                      f"Sharpe {s['sharpe_daily_annualised']:.2f}")
    axes[1].fill_between(df.entry_time, dd * 100, 0, color=ps.ORANGE, alpha=0.6, linewidth=0)
    axes[1].set_ylabel("Drawdown (%)")
    fig.tight_layout()
    fig.savefig(PLOTS / "equity_curve.png")
    plt.close(fig)


def _true_bucket_edges(df, by):
    """Expected R per bucket implied by the generator (synthetic data only)."""
    s = df.entry_time.dt.hour
    session = np.select([s < 7, s < 13], ["Asia", "London"], "New York")
    true = BASE_EDGE + pd.Series(session).map(SESSION_EDGE).values + df.instrument.map(INSTRUMENT_EDGE).values
    return pd.Series(true, index=df.index).groupby(df[by]).mean()


def plot_breakdowns(df):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2))
    for ax, by in zip(axes, ("session", "instrument")):
        b = breakdown(df, by).iloc[::-1]
        y = np.arange(len(b))
        colors = [ps.ORANGE if v == "weaker than rest" else ps.AQUA if v == "stronger than rest"
                  else ps.BLUE for v in b.verdict]
        ax.errorbar(b.expectancy_r, y, xerr=[b.expectancy_r - b.ci_low, b.ci_high - b.expectancy_r],
                    fmt="none", ecolor=ps.MUTED, elinewidth=1.5, capsize=3)
        ax.scatter(b.expectancy_r, y, c=colors, s=45, zorder=3, label="Estimated (95% CI)")
        true = _true_bucket_edges(df, by).reindex(b.index)
        ax.scatter(true, y, marker="|", s=220, color=ps.INK, zorder=4, label="True edge (generator)")
        ax.axvline(0, color=ps.INK, linewidth=0.8)
        ax.set_yticks(y, [f"{k}  (n={n}, q={q:.2f})" for k, n, q in zip(b.index, b.n_trades, b.q_value)])
        ax.set_xlabel("Expectancy (R per trade)")
        ax.set_title(f"By {by}")
    from matplotlib.lines import Line2D
    handles = [Line2D([], [], marker="o", linestyle="none", color=ps.BLUE, label="Estimated, 95% CI"),
               Line2D([], [], marker="|", linestyle="none", markersize=14, color=ps.INK,
                      label="True edge (known to generator)")]
    axes[0].legend(handles=handles, loc="center left")
    fig.suptitle("Where is the edge? Orange/green = different from the rest after FDR correction (q < 0.1)",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "breakdown_with_ci.png")
    plt.close(fig)


def plot_r_distribution(df):
    r = df.r_multiple
    fig, ax = plt.subplots(figsize=(8, 4.4))
    ax.hist(r, bins=np.arange(-3, 6.25, 0.25), color=ps.BLUE, alpha=0.85, edgecolor=ps.SURFACE)
    ax.axvline(-1, color=ps.MUTED, linestyle="--", linewidth=1)
    ax.text(-1, 0.97, " planned stop (-1R)", transform=ax.get_xaxis_transform(), va="top", color=ps.INK_2)
    ax.axvline(r.mean(), color=ps.ORANGE, linewidth=1.5)
    ax.text(r.mean(), 0.85, f" mean {r.mean():+.2f}R", transform=ax.get_xaxis_transform(), color=ps.INK)
    ax.set_title(f"Trade outcomes in R  |  {(r < -1.5).sum()} of {len(r)} losses exceeded 1.5R (slippage/gaps)")
    ax.set_xlabel("R-multiple")
    ax.set_ylabel("Trades")
    fig.tight_layout()
    fig.savefig(PLOTS / "r_multiple_distribution.png")
    plt.close(fig)


def plot_power():
    pc = power_curve()
    fdr = false_discovery_rate()
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), gridspec_kw={"width_ratios": [1.6, 1]})
    ax = axes[0]
    ax.plot(pc.n_trades, pc.power * 100, "o-", color=ps.BLUE)
    ax.axhline(80, color=ps.MUTED, linestyle="--", linewidth=1)
    ax.text(pc.n_trades.min(), 81, "80% power", color=ps.INK_2)
    ax.set_xscale("log")
    ax.set_xticks(pc.n_trades, [str(n) for n in pc.n_trades])
    ax.set_xlabel("Trades in journal (log)")
    ax.set_ylabel("Chance of detecting it (%)")
    ax.set_ylim(0, 105)
    asia_gap = SESSION_EDGE["Asia"] - (SESSION_EDGE["London"] * SESSION_WEIGHTS[1]
                                       + SESSION_EDGE["New York"] * SESSION_WEIGHTS[2]) / sum(SESSION_WEIGHTS[1:])
    ax.set_title(f"Detecting a real {asia_gap:+.2f}R session handicap")
    ax = axes[1]
    vals = [fdr["any_flag_raw_p05"] * 100, fdr["any_flag_bh_q10"] * 100]
    ax.bar(["Raw p < 0.05", "BH-corrected\nq < 0.1"], vals, color=[ps.ORANGE, ps.BLUE], width=0.55)
    for i, v in enumerate(vals):
        ax.text(i, v + 1, f"{v:.0f}%", ha="center", color=ps.INK)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.grid(axis="x", visible=False)
    ax.set_ylabel("Journals with a false 'edge' (%)")
    ax.set_title("No-edge journals, 5 instruments, 400 trades")
    fig.suptitle("How much data does a journal breakdown need? (300-500 simulated journals per point)",
                 fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "power_and_false_discoveries.png")
    plt.close(fig)


if __name__ == "__main__":
    df = load_journal(DATA)
    plot_equity(df)
    plot_breakdowns(df)
    plot_r_distribution(df)
    plot_power()
    print("Plots written to", PLOTS)
