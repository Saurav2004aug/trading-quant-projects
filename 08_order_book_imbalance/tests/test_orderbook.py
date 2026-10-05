import numpy as np
import pandas as pd

from features import clock_grid, event_features, next_mid_move, ofi_increments
from lobster import load, validate
from study import adverse_selection, contemporaneous, split, taker_strategy


def _book(rows):
    """rows: (time, ask_p1, ask_q1, bid_p1, bid_q1); deeper levels filled consistently."""
    out = []
    for t, a, qa, b, qb in rows:
        out.append({"time": t, "type": 1, "size": 1, "price": a, "direction": 1,
                    "ask_p1": a, "ask_q1": qa, "bid_p1": b, "bid_q1": qb,
                    "ask_p2": a + .01, "ask_q2": 100, "bid_p2": b - .01, "bid_q2": 100,
                    "ask_p3": a + .02, "ask_q3": 100, "bid_p3": b - .02, "bid_q3": 100})
    return pd.DataFrame(out)


# ------------------------------------------------------------------ data integrity
def test_real_book_integrity():
    v = validate(load())
    assert v["events"] == 400_391 and v["time_monotonic"] and v["crossed_books"] == 0
    assert v["ask_levels_sorted"] and v["bid_levels_sorted"] and v["prices_on_tick_grid"]
    assert v["event_types"] == {1: 191015, 2: 3260, 3: 171126, 4: 23658, 5: 11332}


# ------------------------------------------------------------------ features
def test_ofi_matches_hand_calculation():
    a = np.array([10.05, 10.05, 10.04, 10.04, 10.04])
    qa = np.array([100., 80., 50., 50., 50.])
    b = np.array([10.00, 10.00, 10.00, 10.01, 10.01])
    qb = np.array([200., 200., 200., 30., 60.])
    e = ofi_increments(a, qa, b, qb)
    # 1: ask queue shrinks by 20 at same price -> +20
    # 2: new better ask of 50 -> -50 ; 3: new better bid of 30 -> +30 ; 4: bid queue +30 -> +30
    assert np.allclose(e, [0, 20, -50, 30, 30])


def test_imbalance_and_microprice_ranges():
    ev = event_features(load().iloc[:50_000])
    assert ev.imb1.between(-1, 1).all() and ev.imb3.between(-1, 1).all()
    assert (ev.micro_minus_mid_ticks.abs() <= ev.spread_ticks / 2 + 1e-9).all()


def test_clock_grid_uses_only_past_events():
    ev = event_features(load())
    g1 = clock_grid(ev, horizons=(1,))
    cut = 45_000.0
    shocked = ev.copy()
    later = shocked.time > cut
    shocked.loc[later, ["imb1", "imb3", "micro_minus_mid_ticks", "ofi"]] *= -1
    g2 = clock_grid(shocked, horizons=(1,))
    cols = ["imb1", "imb3", "micro_minus_mid_ticks", "ofi_past"]
    early = g1.time <= cut
    pd.testing.assert_frame_equal(g1.loc[early, cols], g2.loc[early, cols])


def test_targets_are_future_mid_changes():
    raw = _book([(100.0, 10.10, 10, 10.00, 10), (100.5, 10.12, 10, 10.02, 10), (102.2, 10.20, 10, 10.10, 10)])
    ev = event_features(raw)
    g = clock_grid(ev, dt=1.0, horizons=(1, 2), start=100.0, end=103.0)
    # at t=100 mid=10.05; at t=101 the last event (100.5) gives 10.07; at t=102 still 10.07
    assert np.allclose(g.loc[0, "dmid_1s"], 2.0) and np.allclose(g.loc[0, "dmid_2s"], 2.0)


def test_next_mid_move_direction():
    raw = _book([(1, 10.10, 10, 10.00, 10), (2, 10.10, 20, 10.00, 10),
                 (3, 10.12, 10, 10.02, 10), (4, 10.08, 10, 9.98, 10)])
    nm = next_mid_move(event_features(raw))
    assert list(nm.next_move) == [1, 1, -1]


# ------------------------------------------------------------------ evaluation and economics
def test_split_is_chronological_and_disjoint():
    g = clock_grid(event_features(load()))
    tr, te = split(g)
    assert tr.time.max() < te.time.min() and len(tr) + len(te) == len(g)


def test_contemporaneous_ofi_relation_on_real_data():
    g = clock_grid(event_features(load()))
    c = contemporaneous(*split(g))
    assert (c.ticks_per_1000_shares > 0).all() and (c.r2_test > 0.3).all()


def test_taker_pays_the_spread():
    n = 30
    test = pd.DataFrame({"imb1": np.ones(n), "imb3": 0.0, "micro_minus_mid_ticks": 0.0, "ofi_past": 0.0,
                         "spread_ticks": 10.0, "dmid_1s": 0.0})
    beta = np.array([0.0, 1.0, 0.0, 0.0, 0.0])        # predicts +1 tick everywhere
    r = taker_strategy(test, beta, 1, threshold=0.5)
    assert r["trades"] == n - 1                          # last row has no exit quote
    assert abs(r["net_ticks"] - (-10.0 - 0.6)) < 1e-12
    none = taker_strategy(test, np.zeros(5), 1, threshold=0.5)
    assert none["trades"] == 0


def test_adverse_selection_sign():
    raw = _book([(1.0, 10.10, 10, 10.00, 10), (1.01, 10.12, 10, 10.02, 10), (5.0, 10.14, 10, 10.04, 10)])
    raw.loc[1, "type"] = 4
    raw.loc[1, "direction"] = -1                         # resting sell hit by a buyer
    adv = adverse_selection(event_features(raw), raw, lags=(1,))
    assert adv.mean_move_ticks.iloc[0] > 0
