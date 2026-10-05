"""
Generate a messy trade-level export, plus a ground-truth file listing every
problem that was injected, so the ETL's detection can be measured.

Clean base data: trades on 5 instruments over 180 business days, with
instrument-appropriate price levels (EURUSD ~1.08, XAUUSD ~2300, BTCUSD
~60000 ...) following a daily random walk, lognormal sizes, and P&L.

Injected problems (recorded in raw_data/injected_issues.json):
  - three date formats, one per upstream system: 2025-03-04, 04/03/2025
    (day first), 03-04-2025 (month first)
  - instrument spelling: case, whitespace, "EUR/USD" style separators
  - unknown instruments (e.g. "EURUSDX", "TEST")
  - unparseable dates
  - missing pnl, missing avg_price
  - negative / zero volume
  - fat-finger volume (x1000) and price (x10 decimal slip)
  - exact duplicate rows (a re-export) and conflicting duplicates
    (same trade_id, different values -- an amended trade)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PRICE_LEVEL = {"EURUSD": 1.08, "GBPUSD": 1.27, "XAUUSD": 2300.0, "BTCUSD": 60000.0, "NAS100": 18000.0}
DAILY_VOL = {"EURUSD": 0.005, "GBPUSD": 0.006, "XAUUSD": 0.01, "BTCUSD": 0.03, "NAS100": 0.012}
DATE_FORMATS = ["%Y-%m-%d", "%d/%m/%Y", "%m-%d-%Y"]


def _spelling_variant(sym: str, rng) -> str:
    v = rng.integers(4)
    if v == 0:
        return f" {sym.lower()} "
    if v == 1:
        return sym.capitalize()
    if v == 2:
        return f"{sym[:3]}/{sym[3:]}"
    return f"{sym}  "


def generate_raw_csv(n_trades: int = 3000, seed: int = 17, out_dir: Path | None = None):
    rng = np.random.default_rng(seed)
    out_dir = out_dir or ROOT / "raw_data"
    out_dir.mkdir(parents=True, exist_ok=True)

    days = pd.bdate_range("2025-01-01", periods=180)
    paths = {s: PRICE_LEVEL[s] * np.exp(np.cumsum(rng.normal(0, DAILY_VOL[s], len(days))))
             for s in PRICE_LEVEL}
    inst = rng.choice(list(PRICE_LEVEL), size=n_trades)
    day_idx = rng.integers(len(days), size=n_trades)
    price = np.array([paths[s][d] for s, d in zip(inst, day_idx)]) * (1 + rng.normal(0, 0.002, n_trades))
    df = pd.DataFrame({
        "trade_id": [f"T{100000 + i}" for i in range(n_trades)],
        "trade_date": [days[d].strftime(DATE_FORMATS[rng.integers(3)]) for d in day_idx],
        "instrument": inst.astype(object),
        "side": rng.choice(["BUY", "SELL"], size=n_trades),
        "volume": rng.lognormal(2.0, 0.6, n_trades).round(2),
        "avg_price": price.round(5),
        "pnl": rng.normal(0, 250, n_trades).round(2),
    })

    issues: dict[str, list[str]] = {}

    def pick(k, name):
        pool = df.index[~df.trade_id.isin(sum(issues.values(), []))]
        idx = rng.choice(pool, size=k, replace=False)
        issues[name] = df.loc[idx, "trade_id"].tolist()
        return idx

    idx = rng.choice(df.index, size=int(0.2 * n_trades), replace=False)       # cosmetic only
    df.loc[idx, "instrument"] = [_spelling_variant(s, rng) for s in df.loc[idx, "instrument"]]
    df.loc[pick(12, "unknown_instrument"), "instrument"] = rng.choice(["EURUSDX", "TEST", "XAGUSD"], 12)
    df.loc[pick(10, "bad_date"), "trade_date"] = rng.choice(["2025-13-01", "31/02/2025", "", "n/a"], 10)
    df.loc[pick(90, "missing_pnl"), "pnl"] = np.nan
    df.loc[pick(45, "missing_price"), "avg_price"] = np.nan
    df.loc[pick(8, "bad_volume"), "volume"] = rng.choice([0.0, -5.0, -1.0], 8)
    i = pick(10, "volume_outlier")
    df.loc[i, "volume"] = df.loc[i, "volume"] * 1000
    i = pick(10, "price_outlier")
    df.loc[i, "avg_price"] = df.loc[i, "avg_price"] * 10

    # Duplicates are drawn from otherwise-clean rows so each trade has one issue.
    clean_pool = df.index[~df.trade_id.isin(sum(issues.values(), []))]
    exact = df.loc[rng.choice(clean_pool, 60, replace=False)]
    issues["exact_duplicate"] = exact.trade_id.tolist()
    remaining = clean_pool.difference(exact.index)
    amended = df.loc[rng.choice(remaining, 15, replace=False)].copy()
    amended["pnl"] = (amended["pnl"] + rng.normal(0, 50, len(amended))).round(2)
    issues["conflicting_duplicate"] = amended.trade_id.tolist()

    raw = pd.concat([df, exact, amended], ignore_index=True).sample(frac=1.0, random_state=seed)
    raw.to_csv(out_dir / "trades_raw.csv", index=False)
    (out_dir / "injected_issues.json").write_text(json.dumps(issues, indent=1))
    print(f"{len(raw)} raw rows -> {out_dir / 'trades_raw.csv'}")
    return raw, issues


if __name__ == "__main__":
    generate_raw_csv()
