"""Data-quality figure for the README: where every raw row ended up, and
whether each injected problem was caught (uses raw_data/injected_issues.json)."""
import json
import sqlite3
from pathlib import Path

import pandas as pd

import plot_style as ps
import matplotlib.pyplot as plt
from etl import DB_PATH, RAW_CSV, run_etl

ROOT = Path(__file__).resolve().parents[1]
PLOTS = ROOT / "plots"
PLOTS.mkdir(exist_ok=True)

EXPECTED = {"unknown_instrument": "unknown_instrument", "bad_date": "unparseable_date",
            "missing_pnl": "missing_pnl", "bad_volume": "invalid_volume",
            "price_outlier": "price_out_of_range", "conflicting_duplicate": "conflicting_duplicate"}


def main():
    report = run_etl(RAW_CSV, DB_PATH)
    issues = json.loads((ROOT / "raw_data" / "injected_issues.json").read_text())
    with sqlite3.connect(DB_PATH) as c:
        q = pd.read_sql("SELECT reason, trade_id FROM quarantine", c)
        t = pd.read_sql("SELECT trade_id, volume_outlier_flag, price_imputed FROM trades", c)

    rows = []
    for issue, reason in EXPECTED.items():
        got = set(q[q.reason == reason].trade_id)
        want = set(issues[issue])
        rows.append((issue.replace("_", " "), len(want), len(got & want), len(got - want)))
    for issue, col in [("volume_outlier", "volume_outlier_flag"), ("missing_price", "price_imputed")]:
        got = set(t[t[col] == 1].trade_id)
        want = set(issues[issue])
        rows.append((issue.replace("_", " ") + (" (flagged)" if col.startswith("vol") else " (imputed)"),
                     len(want), len(got & want), len(got - want)))
    det = pd.DataFrame(rows, columns=["issue", "injected", "caught", "false_flags"])

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4), gridspec_kw={"width_ratios": [1, 1.3]})
    ax = axes[0]
    parts = [("Loaded", report["rows_loaded"], ps.BLUE),
             ("Quarantined", report["rows_quarantined"], ps.ORANGE),
             ("Exact duplicates\ndropped", report["exact_duplicates_dropped"], ps.MUTED)]
    ax.bar([p[0] for p in parts], [p[1] for p in parts], color=[p[2] for p in parts], width=0.6)
    for i, p in enumerate(parts):
        ax.text(i, p[1] + 30, f"{p[1]:,}", ha="center", color=ps.INK)
    ax.set_title(f"{report['rows_in']:,} raw rows: every one accounted for")
    ax.set_ylabel("Rows")
    ax.grid(axis="x", visible=False)
    ax = axes[1]
    y = range(len(det))[::-1]
    ax.barh(list(y), det.injected, color="#dddcd6", height=0.6, label="Injected")
    ax.barh(list(y), det.caught, color=ps.AQUA, height=0.35, label="Caught")
    for yi, r in zip(y, det.itertuples()):
        ax.text(r.injected + 1, yi, f"{r.caught}/{r.injected}, {r.false_flags} false", va="center",
                color=ps.INK_2, fontsize=8)
    ax.set_yticks(list(y), det.issue)
    ax.set_xlim(0, det.injected.max() * 1.35)
    ax.set_title("Detection vs ground truth")
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(PLOTS / "data_quality.png")
    plt.close(fig)
    print(det.to_string(index=False))


if __name__ == "__main__":
    main()
