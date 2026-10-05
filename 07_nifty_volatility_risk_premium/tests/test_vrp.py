import numpy as np
import pandas as pd

from data import load
from options_data import (black76, black76_delta, build_iv30, expiry_snapshot, implied_vol, iv30)
from real_straddle import run_cycles
from straddle import Config, bs, run_cycle
from vrp import build, newey_west_ols, realised_vol


# ------------------------------------------------------------------ real data sanity
def test_bundled_data_matches_known_market_events():
    df = load()
    assert abs(df.loc["2020-03-23", "spot"] - 7610.25) < 0.01          # COVID low close
    assert abs(df.loc["2024-06-04", "spot"] - 21884.50) < 0.01         # election-result crash
    assert abs(df.loc["2020-03-24", "vix"] - 83.61) < 0.01             # India VIX record close
    assert not df.index.duplicated().any() and (df.spot > 0).all()


def test_parity_forward_matches_nifty_futures():
    ts, _ = build_iv30()
    fut = pd.read_csv(load.__globals__["DATA"] / "nifty_futures_eod.csv", parse_dates=["date", "expiry"])
    m = ts.merge(fut, on=["date", "expiry"])
    rel = ((m.forward - m.settle).abs() / m.settle)
    assert len(m) > 2000 and rel.median() < 5e-4                         # within 5 bps


# ------------------------------------------------------------------ realised vol and statistics
def test_realised_vol_alignment_has_no_look_ahead():
    r = pd.Series(np.r_[np.zeros(30), np.full(21, 0.01), np.zeros(30)])
    fwd = realised_vol(r, 21, forward=True)
    past = realised_vol(r, 21)
    # forward vol at t=29 covers returns 30..50 exactly: 1% daily -> 15.87% annualised
    assert abs(fwd.iloc[29] - 0.01 * np.sqrt(252) * 100) < 1e-9
    assert fwd.iloc[50] < fwd.iloc[29] and past.iloc[50] == fwd.iloc[29]


def test_newey_west_zero_lags_is_white_se():
    rng = np.random.default_rng(0)
    x = rng.normal(size=500)
    y = 1 + 2 * x + rng.normal(size=500) * (1 + np.abs(x))
    b, se, _ = newey_west_ols(y, x, lags=0)
    X = np.column_stack([np.ones(500), x])
    u = y - X @ b
    white = np.linalg.inv(X.T @ X) @ (X.T * u**2) @ X @ np.linalg.inv(X.T @ X)
    assert np.allclose(se, np.sqrt(np.diag(white)))


def test_newey_west_widens_errors_for_overlapping_windows():
    rng = np.random.default_rng(1)
    e = pd.Series(rng.normal(size=3000)).rolling(21).sum().dropna().to_numpy()   # overlapping sums
    _, se0, _ = newey_west_ols(e, np.empty((len(e), 0)), 0)
    _, se20, _ = newey_west_ols(e, np.empty((len(e), 0)), 20)
    assert se20[0] > 3 * se0[0]


def test_vrp_definitions():
    v = build(load())
    row = v.dropna().iloc[100]
    assert abs(row.vrp_vol - (row.vix - row.rv_fwd)) < 1e-12


# ------------------------------------------------------------------ option maths
def test_black76_parity_delta_and_iv_round_trip():
    F, K, T, s = 20000.0, 20100.0, 0.08, 0.15
    c, p = black76(F, K, T, s, kind="C"), black76(F, K, T, s, kind="P")
    assert abs((c - p) - np.exp(-0.065 * T) * (F - K)) < 1e-9
    h = 1e-3
    fd = (black76(F + h, K, T, s, kind="C") - black76(F - h, K, T, s, kind="C")) / (2 * h)
    assert abs(fd - black76_delta(F, K, T, s, kind="C")) < 1e-7
    assert abs(implied_vol(c, F, K, T, "C") - s) < 1e-7


def _synthetic_chain(F=20000.0, T=30 / 365, sig=0.16, date="2024-01-02"):
    strikes = np.arange(18000, 22001, 50.0)
    rows = []
    for k in strikes:
        for kind, t in (("C", "CE"), ("P", "PE")):
            rows.append({"date": pd.Timestamp(date), "expiry": pd.Timestamp(date) + pd.Timedelta(days=round(T * 365)),
                         "strike": k, "type": t, "close": black76(F, k, T, sig, kind=kind),
                         "contracts": 100, "T": T})
    return pd.DataFrame(rows)


def test_snapshot_recovers_forward_and_vol():
    s = expiry_snapshot(_synthetic_chain(F=20013.0, sig=0.18))
    assert abs(s["forward"] - 20013.0) < 0.5
    assert abs(s["atm_iv"] - 0.18) < 1e-4


def test_iv30_interpolates_total_variance():
    ts = pd.DataFrame({"date": pd.Timestamp("2024-01-02"), "atm_iv": [0.10, 0.20], "T": [20 / 365, 40 / 365]})
    expected = np.sqrt((0.5 * 0.10**2 * 20 + 0.5 * 0.20**2 * 40) / 30) * 100
    assert abs(iv30(ts).iloc[0] - expected) < 1e-9


# ------------------------------------------------------------------ straddle engines
def test_straddle_with_no_move_earns_premium():
    spot = np.full(22, 20000.0)
    vix = np.full(22, 15.0)
    res = run_cycle(spot, vix, Config(opt_cost=0.0, hedge_bps=0.0))
    assert abs(res["pnl_pct"] - res["premium_pct"]) < 1e-9


def test_model_straddle_breaks_even_when_realised_equals_implied():
    rng = np.random.default_rng(3)
    pnl = []
    for _ in range(3000):
        path = 20000 * np.exp(np.cumsum(np.r_[0, rng.normal(-0.5 * 0.16**2 / 252, 0.16 / np.sqrt(252), 21)]))
        pnl.append(run_cycle(path, np.full(22, 16.0), Config(opt_cost=0.0, hedge_bps=0.0, r=0.0))["pnl_pct"])
    pnl = np.array(pnl)
    assert abs(pnl.mean()) < 4 * pnl.std() / np.sqrt(len(pnl)) + 0.02


def test_attribution_explains_most_of_pnl():
    rng = np.random.default_rng(4)
    path = 20000 * np.exp(np.cumsum(np.r_[0, rng.normal(0, 0.012, 21)]))
    vix = 16 + np.cumsum(np.r_[0, rng.normal(0, 0.3, 21)])
    r = run_cycle(path, vix, Config(opt_cost=0.0, hedge_bps=0.0, r=0.0))
    assert abs(r["gamma_pct"] + r["theta_pct"] + r["vega_pct"] - r["pnl_pct"]) < 0.25 * abs(r["premium_pct"])


def test_real_backtest_on_synthetic_market_prices():
    """Build a fake chain from Black-Scholes on a GBM path; the real-price engine must
    earn roughly the implied-minus-realised premium and never lose its forward."""
    rng = np.random.default_rng(5)
    days = pd.bdate_range("2024-01-01", periods=45)
    exp1, exp2 = days[20], days[44]
    spot = pd.Series(20000 * np.exp(np.cumsum(np.r_[0, rng.normal(0, 0.10 / np.sqrt(252), 44)])), index=days)
    rows = []
    for d in days:
        for exp in (exp1, exp2):
            T = (exp - d).days / 365
            if T <= 0 or d > exp:
                continue
            F = spot[d] * np.exp(0.065 * T)
            for k in np.arange(18500, 21501, 50.0):
                for kind, t in (("C", "CE"), ("P", "PE")):
                    rows.append({"date": d, "expiry": exp, "strike": k, "type": t, "contracts": 10,
                                 "close": black76(F, k, T, 0.20, kind=kind), "T": T})
    chain = pd.DataFrame(rows)
    from options_data import atm_term_structure
    ts = atm_term_structure(chain)
    out = run_cycles(chain, ts, spot, opt_cost=0.0, hedge_bps=0.0)
    assert len(out) == 1 and out.missing_forward_days.iloc[0] == 0
    assert abs(out.iv_entry.iloc[0] - 20.0) < 0.1
    assert out.pnl_pct.iloc[0] > 0          # sold at 20% implied, realised about 10%
