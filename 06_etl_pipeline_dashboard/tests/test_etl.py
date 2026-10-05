import os
import runpy
import sqlite3
import sys
from pathlib import Path

import pandas as pd

from etl import extract, parse_trade_date, run_etl, transform
from generate_raw_data import generate_raw_csv

TESTS = Path(__file__).resolve().parent


def _build(tmp_path):
    raw, issues = generate_raw_csv(out_dir=tmp_path)
    db = tmp_path / "wh.db"
    report = run_etl(tmp_path / "trades_raw.csv", db)
    return raw, issues, db, report


def test_every_injected_problem_is_caught(tmp_path):
    _, issues, db, _ = _build(tmp_path)
    with sqlite3.connect(db) as c:
        q = pd.read_sql("SELECT reason, trade_id FROM quarantine", c)
        t = pd.read_sql("SELECT * FROM trades", c)
    expected = {"unknown_instrument": "unknown_instrument", "bad_date": "unparseable_date",
                "missing_pnl": "missing_pnl", "bad_volume": "invalid_volume",
                "price_outlier": "price_out_of_range", "conflicting_duplicate": "conflicting_duplicate"}
    for issue, reason in expected.items():
        assert set(q[q.reason == reason].trade_id) == set(issues[issue]), issue
    assert set(t[t.volume_outlier_flag == 1].trade_id) == set(issues["volume_outlier"])
    assert set(t[t.price_imputed == 1].trade_id) == set(issues["missing_price"])
    # exact re-exports are loaded once, not quarantined
    assert set(issues["exact_duplicate"]) <= set(t.trade_id)


def test_row_accounting_reconciles(tmp_path):
    raw, _, _, report = _build(tmp_path)
    assert report["reconciles"]
    assert report["rows_in"] == len(raw)
    assert report["rows_loaded"] + report["rows_quarantined"] + report["exact_duplicates_dropped"] == len(raw)


def test_idempotent_and_run_log(tmp_path):
    _, _, db, _ = _build(tmp_path)
    with sqlite3.connect(db) as c:
        first = pd.read_sql("SELECT * FROM trades ORDER BY trade_id", c).drop(columns="load_run_id")
    run_etl(tmp_path / "trades_raw.csv", db)
    with sqlite3.connect(db) as c:
        second = pd.read_sql("SELECT * FROM trades ORDER BY trade_id", c).drop(columns="load_run_id")
        runs = pd.read_sql("SELECT * FROM etl_runs", c)
    pd.testing.assert_frame_equal(first, second)
    assert len(runs) == 2 and runs.source_sha256.nunique() == 1


def test_primary_key_enforced(tmp_path):
    _, _, db, _ = _build(tmp_path)
    with sqlite3.connect(db) as c:
        row = c.execute("SELECT * FROM trades LIMIT 1").fetchone()
        try:
            c.execute(f"INSERT INTO trades VALUES ({','.join('?' * len(row))})", row)
            assert False, "duplicate trade_id accepted"
        except sqlite3.IntegrityError:
            pass


def test_clean_values_are_sane(tmp_path):
    _, _, db, _ = _build(tmp_path)
    with sqlite3.connect(db) as c:
        t = pd.read_sql("SELECT * FROM trades", c)
    assert set(t.instrument) == {"EURUSD", "GBPUSD", "XAUUSD", "BTCUSD", "NAS100"}
    levels = t.groupby("instrument").avg_price.median()
    assert 0.9 < levels["EURUSD"] < 1.3 and 30000 < levels["BTCUSD"] < 120000
    assert (t.volume > 0).all() and t.pnl.notna().all()


def test_date_formats_are_explicit():
    assert parse_trade_date("04/03/2025") == pd.Timestamp("2025-03-04")   # day-first system
    assert parse_trade_date("03-04-2025") == pd.Timestamp("2025-03-04")   # month-first system
    assert parse_trade_date("2025-03-04") == pd.Timestamp("2025-03-04")
    for bad in ("2025-13-01", "31/02/2025", "", "n/a", "4/3/25"):
        assert pd.isna(parse_trade_date(bad))


def test_missing_columns_fail_loudly(tmp_path):
    p = tmp_path / "bad.csv"
    pd.DataFrame({"trade_id": ["T1"], "pnl": [1]}).to_csv(p, index=False)
    try:
        extract(p)
        assert False
    except ValueError as e:
        assert "missing required columns" in str(e)


def test_dashboard_executes_against_warehouse(tmp_path):
    """Run dashboard.py top to bottom with a stub `streamlit` module."""
    _, _, db, report = _build(tmp_path)
    sys.path.insert(0, str(TESTS))
    import streamlit_stub
    calls = streamlit_stub.install()
    os.environ["WAREHOUSE_DB"] = str(db)
    try:
        runpy.run_path(str(TESTS.parent / "src" / "dashboard.py"), run_name="dashboard_test")
    finally:
        del os.environ["WAREHOUSE_DB"]
    metrics = {c[1][0]: c[1][1] for c in calls if c[0] == "metric"}
    with sqlite3.connect(db) as c:
        n_default = c.execute("SELECT COUNT(*) FROM trades WHERE volume_outlier_flag = 0").fetchone()[0]
    assert metrics["Trades"] == f"{n_default:,}"
    assert metrics["Rows received"] == f"{report['rows_in']:,}"
    assert any(c[0] == "line_chart" for c in calls) and any(c[0] == "download_button" for c in calls)
