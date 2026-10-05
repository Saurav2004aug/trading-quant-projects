import numpy as np

from binomial import crr_price, early_exercise_premium
from black_scholes import delta, gamma, implied_vol, price, rho, theta, vega
from delta_hedge_sim import frequency_study, run_hedge_simulation, theoretical_mean_pnl

ARGS = dict(S=100.0, K=100.0, T=1.0, r=0.05, sigma=0.2)


def test_textbook_values():
    # Hull: ATM 1y call, r=5%, vol=20%
    assert abs(float(price(**ARGS)) - 10.4506) < 1e-4
    assert abs(float(delta(**ARGS)) - 0.6368) < 1e-4
    assert abs(float(price(**ARGS, option_type="put")) - 5.5735) < 1e-4


def test_put_call_parity_with_dividends():
    S, K, T, r, s, q = 105.0, 95.0, 0.7, 0.03, 0.3, 0.02
    c = price(S, K, T, r, s, "call", q)
    p = price(S, K, T, r, s, "put", q)
    assert abs(float(c - p - (S * np.exp(-q * T) - K * np.exp(-r * T)))) < 1e-10


def test_greeks_match_finite_differences():
    h = 1e-4
    S, K, T, r, s = 100.0, 110.0, 0.5, 0.04, 0.25
    for ot in ("call", "put"):
        fd_delta = (price(S + h, K, T, r, s, ot) - price(S - h, K, T, r, s, ot)) / (2 * h)
        fd_gamma = (price(S + h, K, T, r, s, ot) - 2 * price(S, K, T, r, s, ot)
                    + price(S - h, K, T, r, s, ot)) / h**2
        fd_vega = (price(S, K, T, r, s + h, ot) - price(S, K, T, r, s - h, ot)) / (2 * h)
        fd_theta = -(price(S, K, T + h, r, s, ot) - price(S, K, T - h, r, s, ot)) / (2 * h)
        fd_rho = (price(S, K, T, r + h, s, ot) - price(S, K, T, r - h, s, ot)) / (2 * h)
        assert abs(float(delta(S, K, T, r, s, ot) - fd_delta)) < 1e-6
        assert abs(float(gamma(S, K, T, r, s) - fd_gamma)) < 1e-4
        assert abs(float(vega(S, K, T, r, s) - fd_vega)) < 1e-5
        assert abs(float(theta(S, K, T, r, s, ot) - fd_theta)) < 1e-5
        assert abs(float(rho(S, K, T, r, s, ot) - fd_rho)) < 1e-5


def test_implied_vol_round_trip():
    for s in (0.05, 0.2, 0.8):
        p = float(price(100, 90, 0.25, 0.01, s, "put"))
        assert abs(implied_vol(p, 100, 90, 0.25, 0.01, "put") - s) < 1e-7


def test_binomial_converges_to_black_scholes():
    bs = float(price(**ARGS, option_type="put"))
    err = [abs(crr_price(**ARGS, option_type="put", american=False, n_steps=n) - bs)
           for n in (50, 200, 800)]
    assert err[2] < err[0] and err[2] < 5e-3


def test_american_exercise_rules():
    # American put is worth more than European; American call on non-dividend stock is not.
    assert early_exercise_premium(**ARGS, option_type="put") > 0.3
    assert abs(early_exercise_premium(**ARGS, option_type="call")) < 1e-8
    # With a large dividend, early exercise of a deep ITM call has value.
    assert early_exercise_premium(150, 100, 1, 0.05, 0.2, "call", q=0.08) > 0.1


def test_hedge_mean_pnl_matches_theory():
    for rv in (0.12, 0.2, 0.3):
        df = run_hedge_simulation(sigma_realised=rv, n_rebalances=240, n_paths=20000, seed=5)
        se = df.pnl.std() / np.sqrt(len(df))
        assert abs(df.pnl.mean() - theoretical_mean_pnl(sigma_realised=rv)) < 4 * se + 0.01


def test_hedging_error_shrinks_with_frequency():
    stds = [run_hedge_simulation(n_rebalances=n, n_paths=8000, seed=2).pnl.std()
            for n in (8, 32, 128)]
    # Roughly 1/sqrt(n): each 4x in frequency should about halve the error.
    assert stds[0] > stds[1] > stds[2]
    assert 0.35 < stds[1] / stds[0] < 0.65


def test_costs_create_interior_optimum():
    fs = frequency_study(cost_bps=5.0, n_paths=4000)
    best = fs.n_rebalances[fs.p05.idxmax()]
    assert fs.n_rebalances.min() < best < fs.n_rebalances.max()
    assert (fs.mean_cost.diff().dropna() > 0).all()
