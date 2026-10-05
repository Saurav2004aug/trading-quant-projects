import numpy as np
from scipy.optimize import minimize_scalar

from prop_firm import ChallengeRules, challenge_sweep, simulate_challenge
from risk_of_ruin import sweep
from simulate import (exact_median_final, kelly_fraction, log_growth, max_drawdown,
                      ruin_probability, simulate_equity_paths)


def test_kelly_maximises_log_growth():
    for p, b in [(0.45, 1.5), (0.55, 1.0), (0.3, 4.0)]:
        res = minimize_scalar(lambda f: -log_growth(f, p, b), bounds=(0, 0.99), method="bounded")
        assert abs(res.x - kelly_fraction(p, b)) < 1e-4


def test_kelly_zero_without_edge():
    assert kelly_fraction(0.4, 1.5) < 1e-12         # expectancy 0.4*1.5 - 0.6 = 0
    assert kelly_fraction(0.35, 1.5) == 0.0         # negative edge -> do not bet


def test_simulated_median_matches_exact_formula():
    for f in (0.01, 0.05, 0.12):
        paths = simulate_equity_paths(240, 20_000, 0.45, 1.5, f, seed=4)
        assert abs(np.median(paths[:, -1]) / exact_median_final(f, 0.45, 1.5, 240) - 1) < 1e-9


def test_over_betting_loses_money_despite_positive_edge():
    k = kelly_fraction(0.45, 1.5)
    assert exact_median_final(2.5 * k, 0.45, 1.5, 240) < 1.0


def test_ruin_and_drawdown_increase_with_size():
    df = sweep(n_paths=4000, risk_fracs=np.array([0.01, 0.03, 0.06, 0.12]))
    assert df.p_ruin_50pct.is_monotonic_increasing
    assert df.p_dd_ge_20pct.is_monotonic_increasing


def test_drawdown_and_ruin_helpers():
    path = np.array([[1.0, 1.2, 0.6, 0.9]])
    assert abs(max_drawdown(path)[0] - 0.5) < 1e-12
    assert ruin_probability(path, 0.6) == 1.0 and ruin_probability(path, 0.5) == 0.0


def test_empirical_bootstrap_uses_given_r():
    paths = simulate_equity_paths(10, 5, risk_frac=0.1, empirical_r=np.array([2.0]), seed=0)
    assert np.allclose(paths[:, -1], 1.2 ** 10)


def test_challenge_outcomes_sum_to_one():
    df = challenge_sweep(risks=[0.005, 0.01, 0.03], n_paths=3000)
    total = df[["pass", "fail_max_loss", "fail_daily_loss", "timeout"]].sum(axis=1)
    assert np.allclose(total, 1.0)


def test_challenge_certain_winner_always_passes():
    out = simulate_challenge(0.01, win_rate=1.0, payoff=1.0, n_paths=500, seed=0)
    assert out["pass"] == 1.0
    assert abs(out["avg_days_to_pass"] - 4) < 1e-9          # 10 wins of 1% at 3 per day


def test_daily_limit_cliff():
    # With 3 trades/day, three straight losses at >= 5%/3 breach the 5% daily limit.
    below = simulate_challenge(0.016, n_paths=20_000, seed=1)
    above = simulate_challenge(0.0175, n_paths=20_000, seed=1)
    assert below.fail_daily_loss == 0.0
    assert above.fail_daily_loss > 0.3


def test_trailing_rules_are_stricter():
    s = simulate_challenge(0.01, ChallengeRules(drawdown_type="static"), n_paths=20_000, seed=2)
    t = simulate_challenge(0.01, ChallengeRules(drawdown_type="trailing"), n_paths=20_000, seed=2)
    assert t["pass"] <= s["pass"] and t.fail_max_loss >= s.fail_max_loss
