import numpy as np
import pandas as pd

from analyze import (benjamini_hochberg, bootstrap_ci, breakdown, daily_returns,
                     max_losing_streak, overall_stats, permutation_pvalue)
from generate_trade_log import draw_r, generate_trade_log
from loader import load_journal, session_from_time

MT5_CSV = """Time,Position,Symbol,Type,Volume,Price,S / L,T / P,Time,Price,Commission,Swap,Profit
2025.03.03 08:15:00,1,EURUSD,buy,1.0,1.0800,1.0780,1.0850,2025.03.03 10:00:00,1.0840,-7,0,400
2025.03.03 14:30:00,2,xauusd ,sell,0.5,2900.0,2910.0,2880.0,2025.03.03 15:00:00,2910.0,-3,0,-500
2025.03.04 02:00:00,3,GBPUSD,buy,1.0,1.2700,0,0,2025.03.04 03:00:00,1.2710,-7,0,100
"""


def test_mt5_loader(tmp_path):
    path = tmp_path / "mt5.csv"
    path.write_text(MT5_CSV)
    df = load_journal(path, fmt="mt5", account_balance=100_000)
    assert list(df.instrument) == ["EURUSD", "XAUUSD", "GBPUSD"]
    assert abs(df.r_multiple[0] - 2.0) < 1e-9          # +40 pips on a 20-pip stop
    assert abs(df.r_multiple[1] + 1.0) < 1e-9          # short stopped out
    assert np.isnan(df.r_multiple[2])                  # no stop -> no R, not guessed
    assert list(df.session) == ["London", "New York", "Asia"]
    assert abs(df.pnl_pct[0] - 0.393) < 1e-9           # (400 - 7) / 100k


def test_column_mapping_and_validation(tmp_path):
    path = tmp_path / "j.csv"
    pd.DataFrame({"Open": ["2025-01-06 09:00"], "Sym": ["eurusd"], "R": [1.5]}).to_csv(path, index=False)
    df = load_journal(path, mapping={"Open": "entry_time", "Sym": "instrument", "R": "r_multiple"})
    assert df.instrument[0] == "EURUSD" and df.session[0] == "London" and df.pnl_pct[0] == 1.5
    try:
        load_journal(path)
        assert False, "should reject a journal without required columns"
    except ValueError:
        pass


def test_session_boundaries():
    t = pd.Series(pd.to_datetime(["2025-01-06 06:59", "2025-01-06 07:00", "2025-01-06 13:00",
                                  "2025-01-06 21:00"]))
    assert list(session_from_time(t)) == ["Asia", "London", "New York", "Off-hours"]


def test_generator_expectancy_is_calibrated():
    rng = np.random.default_rng(0)
    for edge in (-0.2, 0.0, 0.25):
        assert abs(draw_r(np.full(400_000, edge), rng).mean() - edge) < 0.01


def test_sharpe_counts_flat_days():
    df = generate_trade_log(n_trades=50, seed=1)
    df["session"] = "x"
    daily = daily_returns(df)
    assert len(daily) == len(pd.bdate_range(daily.index.min(), daily.index.max()))
    assert (daily == 0).sum() > 0


def test_bootstrap_ci_covers_true_mean_about_95pct():
    rng = np.random.default_rng(3)
    cover = 0
    for i in range(200):
        lo, hi = bootstrap_ci(rng.normal(0.1, 1.0, 200), n_boot=1000, seed=i)
        cover += lo <= 0.1 <= hi
    assert 0.88 < cover / 200 < 0.99


def test_permutation_pvalue():
    rng = np.random.default_rng(4)
    assert permutation_pvalue(rng.normal(1, 1, 80), rng.normal(0, 1, 300)) < 0.01
    ps = [permutation_pvalue(rng.normal(0, 1, 50), rng.normal(0, 1, 200), n_perm=500, seed=i)
          for i in range(100)]
    assert 0.02 < np.mean(np.array(ps) < 0.05) < 0.12


def test_benjamini_hochberg_known_values():
    q = benjamini_hochberg(np.array([0.01, 0.04, 0.03, 0.5]))
    assert np.allclose(q, [0.04, 0.0533333, 0.0533333, 0.5], atol=1e-6)


def test_losing_streak():
    assert max_losing_streak(pd.Series([1, -1, -1, 2, -1, -1, -1, 0.5])) == 3


def test_breakdown_and_overall_run_on_bundled_journal():
    df = load_journal(__import__("analyze").DATA)
    s = overall_stats(df)
    assert s["expectancy_ci_low"] < s["expectancy_r"] < s["expectancy_ci_high"]
    b = breakdown(df, "session")
    assert set(b.index) == {"Asia", "London", "New York"}
    assert b.loc["Asia", "verdict"] == "weaker than rest"
