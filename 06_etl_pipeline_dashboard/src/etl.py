"""
ETL: raw trade export -> validated SQLite warehouse.

Principle: no row disappears silently. Every input row ends up in exactly
one place and the run report reconciles:

    rows_in = rows_loaded + rows_quarantined + exact_duplicates_dropped

Tables written
--------------
trades      clean trades, PRIMARY KEY trade_id, with flags
            price_imputed and volume_outlier_flag
quarantine  rejected rows, the raw values as received, and a reason code
etl_runs    one row per run: run id, time, source file hash, JSON report
daily_activity (view)  per-day, per-instrument aggregates for the dashboard

Rules
-----
schema          required columns present, else the run fails loudly
trade_date      parsed with the source system's explicit format, chosen by
                separator ('-' ISO year-first, '/' day-first, '-' with a
                4-digit year last = month-first); never inferred
instrument      strip, upper-case, drop '/' and spaces; must be in the
                reference list                            -> unknown_instrument
volume          must be > 0                               -> invalid_volume
pnl             must be present (never imputed)           -> missing_pnl
reference price median of the instrument's prices within +/-5 calendar days
                (robust even on days with only one or two trades)
price sanity    more than 20% away from the reference     -> price_out_of_range
avg_price       missing -> imputed with the reference price and flagged;
                if no reference exists                    -> missing_price
volume outlier  robust z-score of log(volume) within instrument > 5
                -> kept, flagged (a big trade may be real; analyst decides)
duplicates      identical rows with the same trade_id  -> one kept, rest counted
                same trade_id with different values    -> all versions
                                                          -> conflicting_duplicate
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import uuid
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW_CSV = ROOT / "raw_data" / "trades_raw.csv"
DB_PATH = ROOT / "data" / "warehouse.db"

VALID_INSTRUMENTS = {"EURUSD", "GBPUSD", "XAUUSD", "BTCUSD", "NAS100"}
REQUIRED = ["trade_id", "trade_date", "instrument", "side", "volume", "avg_price", "pnl"]
PRICE_TOLERANCE = 0.20
OUTLIER_Z = 5.0

SCHEMA = """
CREATE TABLE IF NOT EXISTS trades (
    trade_id TEXT PRIMARY KEY,
    trade_date TEXT NOT NULL,
    instrument TEXT NOT NULL,
    side TEXT NOT NULL CHECK (side IN ('BUY','SELL')),
    volume REAL NOT NULL CHECK (volume > 0),
    avg_price REAL NOT NULL CHECK (avg_price > 0),
    pnl REAL NOT NULL,
    price_imputed INTEGER NOT NULL,
    volume_outlier_flag INTEGER NOT NULL,
    load_run_id TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_trades_date ON trades(trade_date);
CREATE INDEX IF NOT EXISTS idx_trades_instrument ON trades(instrument);
CREATE TABLE IF NOT EXISTS quarantine (
    run_id TEXT, reason TEXT, trade_id TEXT, trade_date TEXT, instrument TEXT,
    side TEXT, volume TEXT, avg_price TEXT, pnl TEXT
);
CREATE TABLE IF NOT EXISTS etl_runs (
    run_id TEXT PRIMARY KEY, started_at TEXT, source_file TEXT, source_sha256 TEXT, report TEXT
);
CREATE VIEW IF NOT EXISTS daily_activity AS
    SELECT trade_date, instrument, COUNT(*) AS n_trades, SUM(volume) AS volume,
           SUM(pnl) AS pnl, SUM(volume_outlier_flag) AS n_outliers
    FROM trades GROUP BY trade_date, instrument;
"""


# ------------------------------------------------------------------ extract
def extract(path: Path = RAW_CSV) -> pd.DataFrame:
    df = pd.read_csv(path, dtype=str, keep_default_na=False)     # keep raw text for quarantine
    missing = [c for c in REQUIRED if c not in df.columns]
    if missing:
        raise ValueError(f"source file is missing required columns: {missing}")
    return df[REQUIRED]


# ------------------------------------------------------------------ transform helpers
def parse_trade_date(s: str):
    """Explicit format per source system; ambiguous strings are never guessed."""
    s = (s or "").strip()
    if "/" in s:
        fmt = "%d/%m/%Y"                     # system B exports day-first with slashes
    elif len(s) == 10 and s[4] == "-":
        fmt = "%Y-%m-%d"                     # system A: ISO
    elif len(s) == 10 and s[2] == "-":
        fmt = "%m-%d-%Y"                     # system C: US month-first with dashes
    else:
        return pd.NaT
    return pd.to_datetime(s, format=fmt, errors="coerce")


def normalise_instrument(s: pd.Series) -> pd.Series:
    return s.str.upper().str.replace(r"[\s/]", "", regex=True)


def reference_price(df: pd.DataFrame, days: int = 5) -> pd.Series:
    """Median price of the same instrument within +/- `days` calendar days."""
    ref = pd.Series(np.nan, index=df.index)
    for _, g in df.groupby("instrument"):
        dates = g["trade_date"].to_numpy()
        prices = g["avg_price"].to_numpy()
        for d in np.unique(dates):
            window = np.abs(dates - d) <= np.timedelta64(days, "D")
            vals = prices[window]
            vals = vals[~np.isnan(vals)]
            if len(vals):
                ref[g.index[dates == d]] = np.median(vals)
    return ref


def robust_z(x: pd.Series) -> pd.Series:
    med = x.median()
    mad = (x - med).abs().median() * 1.4826
    return (x - med) / mad if mad > 0 else x * 0.0


# ------------------------------------------------------------------ transform
def transform(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    report = {"rows_in": len(raw)}
    df = raw.copy()
    df["_row"] = np.arange(len(df))
    quarantined = []

    def reject(mask: pd.Series, reason: str):
        nonlocal df
        bad = df[mask]
        if len(bad):
            quarantined.append(raw.loc[bad.index].assign(reason=reason))
        report[f"quarantined_{reason}"] = int(mask.sum())
        df = df[~mask]

    # 1. duplicates first, on the raw text
    exact = df.duplicated(subset=REQUIRED, keep="first")
    report["exact_duplicates_dropped"] = int(exact.sum())
    df = df[~exact]
    reject(df.duplicated("trade_id", keep=False), "conflicting_duplicate")

    # 2. parse and validate
    df["instrument"] = normalise_instrument(df["instrument"])
    df["trade_date"] = df["trade_date"].map(parse_trade_date)
    df["side"] = df["side"].str.strip().str.upper()
    for c in ("volume", "avg_price", "pnl"):
        df[c] = pd.to_numeric(df[c].replace("", np.nan), errors="coerce")
    reject(df["trade_date"].isna(), "unparseable_date")
    reject(~df["instrument"].isin(VALID_INSTRUMENTS), "unknown_instrument")
    reject(~df["side"].isin(["BUY", "SELL"]), "invalid_side")
    reject(~(df["volume"] > 0), "invalid_volume")
    reject(df["pnl"].isna(), "missing_pnl")

    # 3. price: sanity check against a robust local reference, then impute gaps
    ref = reference_price(df)
    off = (df["avg_price"] / ref - 1).abs() > PRICE_TOLERANCE
    reject(off & df["avg_price"].notna(), "price_out_of_range")
    ref = reference_price(df)
    df["price_imputed"] = df["avg_price"].isna() & ref.notna()
    df["avg_price"] = df["avg_price"].fillna(ref)
    report["price_imputed"] = int(df["price_imputed"].sum())
    reject(df["avg_price"].isna(), "missing_price")

    # 4. volume outliers: flag, keep
    z = df.groupby("instrument")["volume"].transform(lambda v: robust_z(np.log(v)))
    df["volume_outlier_flag"] = z.abs() > OUTLIER_Z
    report["volume_outliers_flagged"] = int(df["volume_outlier_flag"].sum())

    clean = df.assign(trade_date=df["trade_date"].dt.strftime("%Y-%m-%d"),
                      price_imputed=df["price_imputed"].astype(int),
                      volume_outlier_flag=df["volume_outlier_flag"].astype(int))
    clean = clean[REQUIRED + ["price_imputed", "volume_outlier_flag"]] \
        .sort_values(["trade_date", "trade_id"]).reset_index(drop=True)
    quarantine = pd.concat(quarantined, ignore_index=True) if quarantined else \
        pd.DataFrame(columns=REQUIRED + ["reason"])

    report["rows_loaded"] = len(clean)
    report["rows_quarantined"] = len(quarantine)
    report["reconciles"] = (report["rows_in"] == report["rows_loaded"] + report["rows_quarantined"]
                            + report["exact_duplicates_dropped"])
    if not report["reconciles"]:
        raise RuntimeError(f"row accounting does not reconcile: {report}")
    return clean, quarantine, report


# ------------------------------------------------------------------ load
def load(clean: pd.DataFrame, quarantine: pd.DataFrame, report: dict, source: Path,
         db_path: Path = DB_PATH) -> str:
    """Replace the warehouse contents atomically and append to the run log."""
    run_id = uuid.uuid4().hex[:12]
    db_path.parent.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(Path(source).read_bytes()).hexdigest()
    with sqlite3.connect(db_path) as conn:
        conn.executescript(SCHEMA)
        conn.execute("BEGIN")
        conn.execute("DELETE FROM trades")
        conn.execute("DELETE FROM quarantine")
        clean.assign(load_run_id=run_id).to_sql("trades", conn, if_exists="append", index=False)
        quarantine.assign(run_id=run_id)[["run_id", "reason"] + REQUIRED] \
            .to_sql("quarantine", conn, if_exists="append", index=False)
        conn.execute("INSERT INTO etl_runs VALUES (?,?,?,?,?)",
                     (run_id, datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      Path(source).name, sha, json.dumps(report)))
        conn.commit()
    return run_id


def run_etl(raw_csv: Path = RAW_CSV, db_path: Path = DB_PATH) -> dict:
    clean, quarantine, report = transform(extract(raw_csv))
    report["run_id"] = load(clean, quarantine, report, raw_csv, db_path)
    return report


if __name__ == "__main__":
    rep = run_etl()
    print("ETL run complete")
    for k, v in rep.items():
        print(f"  {k}: {v}")
