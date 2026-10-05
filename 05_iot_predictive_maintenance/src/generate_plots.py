"""Run the full cross-validation, save result tables to ../results and
figures to ../plots. Takes about two minutes."""
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import precision_recall_curve

import plot_style as ps
import matplotlib.pyplot as plt
from simulate_sensor_data import make_classification_labels, simulate_fleet
from train_model import cross_validate, permutation_importance_cv, summarise

ROOT = Path(__file__).resolve().parents[1]
PLOTS, RESULTS = ROOT / "plots", ROOT / "results"
PLOTS.mkdir(exist_ok=True)
RESULTS.mkdir(exist_ok=True)
BEST = "Random forest, engineered"
BASE = "Raw vibration only (no model)"
MODEL_COLORS = {BASE: ps.MUTED, "Logistic regression, raw": ps.YELLOW, "Random forest, raw": ps.ORANGE,
                "Logistic regression, engineered": ps.AQUA, BEST: ps.BLUE}


def plot_signatures(oof: pd.DataFrame, outcomes: pd.DataFrame):
    """One typical (median-warning, caught) machine per failure mode."""
    o = outcomes[(outcomes.model == BEST) & (outcomes.outcome == "caught")]
    fig, axes = plt.subplots(2, 2, figsize=(11, 5.6), sharex="col")
    for col, mode in enumerate(("gradual", "abrupt")):
        om = o[o.failure_mode == mode].sort_values("warning_cycles")
        mid = om.machine_id.iloc[len(om) // 2]
        g = oof[oof.machine_id == mid]
        onset = g.cycle[g.true_degradation > 0].min()
        above = g[f"prob::{BEST}"].to_numpy() >= g[f"thr::{BEST}"].iloc[0]
        run = np.r_[False, above[1:] & above[:-1]]
        alarm = g.cycle.to_numpy()[np.argmax(run)] if run.any() else None
        for row, (c, label) in enumerate([("vibration_g", "Vibration RMS (g)"),
                                          ("temperature_c", "Temperature (°C)")]):
            ax = axes[row, col]
            ax.plot(g.cycle, g[c], color=ps.BLUE if row == 0 else ps.ORANGE, linewidth=1.2)
            ax.axvspan(g.cycle.max() - 20, g.cycle.max(), color=ps.RED, alpha=0.08, linewidth=0)
            ax.axvline(onset, color=ps.MUTED, linestyle=":", linewidth=1.2)
            if alarm is not None:
                ax.axvline(alarm, color=ps.INK, linestyle="--", linewidth=1.2)
            ax.set_ylabel(label)
        axes[0, col].set_title(f"{mode.capitalize()} failure, machine {mid}: "
                               f"alarm {g.cycle.max() - alarm} cycles early")
        axes[1, col].set_xlabel("Operating cycle")
    axes[0, 0].plot([], [], color=ps.MUTED, linestyle=":", label="Degradation starts (hidden)")
    axes[0, 0].plot([], [], color=ps.INK, linestyle="--", label="Model alarm")
    axes[0, 0].fill_between([], [], color=ps.RED, alpha=0.15, label="Label window (last 20 cycles)")
    axes[0, 0].legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "degradation_signature.png")
    plt.close(fig)


def plot_model_comparison(metrics: pd.DataFrame, oof: pd.DataFrame):
    order = metrics.groupby("model").avg_precision.mean().sort_values().index
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1.25, 1]})
    ax = axes[0]
    for i, m in enumerate(order):
        v = metrics[metrics.model == m].avg_precision
        ax.scatter(v, np.full(len(v), i), color=MODEL_COLORS[m], alpha=0.5, s=22)
        ax.plot([v.mean()], [i], "D", color=MODEL_COLORS[m], markersize=8)
    ax.set_yticks(range(len(order)), order)
    ax.set_xlabel("Average precision (PR-AUC) on held-out machines")
    ax.set_title("5 grouped folds (dots) and mean (diamond)")
    ax = axes[1]
    y = oof.will_fail_soon.to_numpy()
    for m in (BASE, "Random forest, raw", BEST):
        prec, rec, _ = precision_recall_curve(y, oof[f"prob::{m}"])
        ax.plot(rec, prec, color=MODEL_COLORS[m], label=m, linewidth=1.8)
    ax.axhline(y.mean(), color=ps.MUTED, linestyle=":", linewidth=1)
    ax.text(0.02, y.mean() + 0.02, f"base rate {y.mean():.0%}", color=ps.INK_2)
    ax.set_xlabel("Recall (failure-window readings)")
    ax.set_ylabel("Precision")
    ax.set_title("Pooled out-of-fold precision-recall")
    ax.legend(loc="upper right", fontsize=8)
    fig.suptitle("Does the model beat a simple vibration threshold?", fontweight="bold", x=0.02, ha="left")
    fig.tight_layout()
    fig.savefig(PLOTS / "model_comparison.png")
    plt.close(fig)


def plot_warning_time(outcomes: pd.DataFrame):
    o = outcomes[outcomes.model == BEST]
    fig, ax = plt.subplots(figsize=(8, 4.8))
    lim = o.degradation_cycles.max() * 1.05
    ax.plot([0, lim], [0, lim], color=ps.MUTED, linestyle="--", linewidth=1)
    ax.text(lim * 0.62, lim * 0.66, "warning = whole degradation period", color=ps.INK_2, rotation=0)
    for mode, color in [("gradual", ps.BLUE), ("abrupt", ps.ORANGE)]:
        g = o[o.failure_mode == mode]
        c = g[g.outcome == "caught"]
        n_fa, n_miss = (g.outcome == "false alarm").sum(), (g.outcome == "missed").sum()
        ax.scatter(c.degradation_cycles, c.warning_cycles, color=color, s=34, alpha=0.85,
                   label=f"{mode.capitalize()}: {len(c)}/{len(g)} caught, {n_fa} false alarm, {n_miss} missed")
    ax.set_xlim(0, lim)
    ax.set_ylim(0, lim)
    ax.set_xlabel("Cycles the machine was actually degrading (ground truth)")
    ax.set_ylabel("Cycles of warning from the alarm")
    ax.set_title("Alarm lead time on held-out machines (random forest, engineered)")
    ax.legend(loc="upper left")
    fig.tight_layout()
    fig.savefig(PLOTS / "warning_time.png")
    plt.close(fig)


def plot_importance(imp: pd.DataFrame):
    imp = imp.iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.barh(imp.index, imp["mean"], xerr=imp["std"], color=ps.BLUE, ecolor=ps.MUTED, height=0.6)
    ax.set_xlabel("Drop in held-out average precision when shuffled")
    ax.set_title("Permutation importance (grouped CV, random forest)")
    fig.tight_layout()
    fig.savefig(PLOTS / "feature_importance.png")
    plt.close(fig)


if __name__ == "__main__":
    raw = make_classification_labels(simulate_fleet())
    metrics, outcomes, oof = cross_validate(raw)
    summary = summarise(metrics, outcomes)
    imp = permutation_importance_cv(raw)
    metrics.to_csv(RESULTS / "fold_metrics.csv", index=False)
    outcomes.to_csv(RESULTS / "machine_outcomes.csv", index=False)
    summary.round(4).to_csv(RESULTS / "model_summary.csv")
    imp.round(4).to_csv(RESULTS / "permutation_importance.csv")
    plot_signatures(oof, outcomes)
    plot_model_comparison(metrics, oof)
    plot_warning_time(outcomes)
    plot_importance(imp)
    print(summary.round(3).to_string())
    print("Plots written to", PLOTS)
