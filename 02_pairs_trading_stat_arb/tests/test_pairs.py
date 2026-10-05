import numpy as np
import pandas as pd

from cointegration import adf_test, engle_granger_test, half_life
from data_gen import load_cointegrated_pair
from engine import Params, backtest, summarize
from research import load_synthetic, walk_forward


def _random_walks(rng, n=500):
    return (pd.Series(np.cumsum(rng.standard_normal(n))),
            pd.Series(np.cumsum(rng.standard_normal(n))))


# ------------------------------------------------------------ statistics
def test_adf_size_close_to_nominal():
    rng = np.random.default_rng(1)
    rejections = np.mean([adf_test(np.cumsum(rng.standard_normal(400)))["reject_5pct"]
                          for _ in range(400)])
    assert 0.015 < rejections < 0.09


def test_engle_granger_size_close_to_nominal():
    rng = np.random.default_rng(2)
    rej = np.mean([engle_granger_test(*_random_walks(rng))["is_cointegrated"] for _ in range(400)])
    assert 0.015 < rej < 0.09


def test_adf_has_power_against_stationary_series():
    rng = np.random.default_rng(3)
    hits = 0
    for _ in range(100):
        x = np.zeros(400)
        for t in range(1, 400):
            x[t] = 0.9 * x[t - 1] + rng.standard_normal()
        hits += adf_test(x)["reject_5pct"]
    assert hits > 90


def test_half_life_recovers_ar1():
    rng = np.random.default_rng(4)
    x = np.zeros(20000)
    for t in range(1, len(x)):
        x[t] = 0.95 * x[t - 1] + rng.standard_normal()
    assert abs(half_life(pd.Series(x)) - np.log(2) / -np.log(0.95)) < 1.5


# ------------------------------------------------------------ engine
def test_trade_pnl_reconciles_with_daily_pnl():
    data = load_cointegrated_pair()
    daily, trades = backtest(data, 0.0, 1.0, Params(cost_bps=10.0), start=375)
    assert len(trades) > 5
    assert abs(trades.net.sum() - daily.net_pnl.sum()) < 1e-12
    assert abs(trades.cost.sum() - daily.cost.sum()) < 1e-12


def test_position_pnl_equals_units_times_spread_change():
    data = load_cointegrated_pair()
    alpha, beta = 0.3, 1.02
    daily, _ = backtest(data, alpha, beta, Params(cost_bps=0.0), start=375)
    expected = daily.units_b.shift(1).fillna(0) * daily.spread.diff().fillna(0)
    assert np.allclose(daily.gross_pnl.values, expected.values, atol=1e-12)


def test_negative_beta_pair_is_traded_correctly():
    base = load_cointegrated_pair()
    spread = base.asset_b - base.asset_a
    flipped = pd.DataFrame({"asset_a": base.asset_a, "asset_b": 300 - base.asset_a + spread})
    eg = engle_granger_test(flipped.asset_a.iloc[:375], flipped.asset_b.iloc[:375])
    assert eg["beta"] < 0 and eg["is_cointegrated"]
    daily, trades = backtest(flipped, eg["alpha"], eg["beta"], Params(cost_bps=0.0), start=375)
    # Hedge holds the same sign in both legs for a negative beta.
    held = daily[daily.position != 0]
    assert (np.sign(held.units_a) == np.sign(held.units_b)).all()
    assert summarize(daily, trades)["sharpe"] > 1.0


def test_no_look_ahead():
    data = load_cointegrated_pair()
    d1, _ = backtest(data, 0.0, 1.0, Params(), start=375)
    shocked = data.copy()
    shocked.iloc[600:, :] *= 1.5                      # change only the future
    d2, _ = backtest(shocked, 0.0, 1.0, Params(), start=375)
    cut = data.index[599]
    assert (d1.loc[:cut, "signal"] == d2.loc[:cut, "signal"]).all()
    assert (d1.loc[:cut, "position"] == d2.loc[:cut, "position"]).all()


def test_execution_lag_delays_positions():
    data = load_cointegrated_pair()
    d0, _ = backtest(data, 0.0, 1.0, Params(execution_lag=0), start=375)
    d1, _ = backtest(data, 0.0, 1.0, Params(execution_lag=1), start=375)
    assert (d1.position.values[1:-1] == d0.signal.values[:-2]).all()


def test_stop_loss_blocks_reentry():
    idx = pd.bdate_range("2024-01-01", periods=80)
    a = pd.Series(100.0, index=idx)
    spread = np.r_[np.zeros(40), np.linspace(-1, -40, 20), np.full(20, -40.0)]
    spread[:40] = np.random.default_rng(0).normal(0, 1, 40)
    data = pd.DataFrame({"asset_a": a, "asset_b": 100 + spread})
    daily, trades = backtest(data, 0.0, 1.0, Params(lookback=20, stop_z=3.0, execution_lag=0))
    assert (trades.exit_reason == "stop").sum() >= 1
    assert len(trades) <= 2


def test_open_position_closed_at_end():
    data = load_cointegrated_pair()
    daily, trades = backtest(data, 0.0, 1.0, Params(), start=375)
    assert daily.position.iloc[-1] == 0
    assert trades.exit_date.notna().all()


# ------------------------------------------------------------ research protocol
def test_walk_forward_skips_non_cointegrated_pairs():
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2015-01-01", periods=1500)
    prices = pd.DataFrame({"A": 100 + np.cumsum(rng.standard_normal(1500)),
                           "B": 100 + np.cumsum(rng.standard_normal(1500))}, index=idx)
    prices = prices - prices.min() + 10
    daily, trades, blocks = walk_forward(prices, Params(), train_days=504, test_days=63)
    traded_share = blocks.traded.mean()
    assert traded_share < 0.25                      # rarely passes the screen
    assert (daily.loc[daily.n_pairs == 0, "net_pnl"] == 0).all()


def test_walk_forward_trades_only_after_training_window():
    prices = load_synthetic(n=1200)
    daily, trades, _ = walk_forward(prices, Params(), train_days=504, test_days=63)
    assert daily.index[0] == prices.index[504]
    if len(trades):
        assert trades.entry_date.min() >= prices.index[504]
