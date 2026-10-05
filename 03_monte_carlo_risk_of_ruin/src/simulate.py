"""
Monte Carlo equity paths under fixed-fractional position sizing.

Each trade risks a fraction f of *current* equity and returns R * f,
where R is the trade's R-multiple (+b on a win, -1 on a loss in the
binary model, or drawn from an empirical distribution of real trades).

Closed-form reference (binary model)
------------------------------------
Log-growth per trade   g(f) = p*ln(1 + b*f) + q*ln(1 - f)
Kelly fraction         f*   = (b*p - q) / b          (maximises g)
Median final equity    = (1+b*f)^k * (1-f)^(n-k),  k = median of Binomial(n, p)
                         (final equity is increasing in the number of wins, so its
                         median is the value at the median win count; ~exp(n*g(f)))
The simulation is checked against these in tests/.
"""
from __future__ import annotations

import numpy as np


def expectancy_r(win_rate: float, payoff: float) -> float:
    return win_rate * payoff - (1 - win_rate)


def kelly_fraction(win_rate: float, payoff: float) -> float:
    return max((payoff * win_rate - (1 - win_rate)) / payoff, 0.0)


def log_growth(f, win_rate: float, payoff: float):
    """Expected log growth per trade, g(f). Vectorised over f."""
    f = np.asarray(f, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return win_rate * np.log1p(payoff * f) + (1 - win_rate) * np.log1p(-f)


def exact_median_final(f, win_rate: float, payoff: float, n_trades: int):
    """Exact median final equity for the binary model (see module docstring)."""
    from scipy.stats import binom
    k = binom.median(n_trades, win_rate)
    f = np.asarray(f, dtype=float)
    return (1 + payoff * f) ** k * np.clip(1 - f, 0, None) ** (n_trades - k)


def sample_r_multiples(n_paths: int, n_trades: int, rng: np.random.Generator,
                       win_rate: float = 0.45, payoff: float = 1.5,
                       empirical_r: np.ndarray | None = None) -> np.ndarray:
    """(n_paths, n_trades) R-multiples: binary model, or bootstrap of real trades."""
    if empirical_r is not None:
        return rng.choice(np.asarray(empirical_r, float), size=(n_paths, n_trades), replace=True)
    wins = rng.random((n_paths, n_trades)) < win_rate
    return np.where(wins, payoff, -1.0)


def simulate_equity_paths(n_trades: int = 240, n_paths: int = 10_000,
                          win_rate: float = 0.45, payoff: float = 1.5,
                          risk_frac: float = 0.01, empirical_r=None,
                          seed: int | None = None) -> np.ndarray:
    """(n_paths, n_trades + 1) equity curves starting at 1.0."""
    rng = np.random.default_rng(seed)
    r = sample_r_multiples(n_paths, n_trades, rng, win_rate, payoff, empirical_r)
    steps = np.maximum(1.0 + risk_frac * r, 0.0)          # cannot lose more than the account
    return np.hstack([np.ones((n_paths, 1)), np.cumprod(steps, axis=1)])


def ruin_probability(paths: np.ndarray, threshold: float = 0.5) -> float:
    """P(equity ever touches `threshold` x starting equity)."""
    return float((paths <= threshold * paths[:, :1]).any(axis=1).mean())


def max_drawdown(paths: np.ndarray) -> np.ndarray:
    """Per-path maximum peak-to-trough drawdown (fraction)."""
    peaks = np.maximum.accumulate(paths, axis=1)
    return ((peaks - paths) / peaks).max(axis=1)


if __name__ == "__main__":
    p, b, f = 0.45, 1.5, 0.01
    paths = simulate_equity_paths(win_rate=p, payoff=b, risk_frac=f, seed=42)
    print(f"Expectancy          {expectancy_r(p, b):+.3f} R per trade")
    print(f"Kelly fraction      {kelly_fraction(p, b):.2%}")
    print(f"P(hit 50% of start) {ruin_probability(paths):.2%}  at {f:.0%} risk, 240 trades")
    print(f"P(max DD >= 20%)    {(max_drawdown(paths) >= 0.2).mean():.2%}")
    print(f"Median final equity {np.median(paths[:, -1]):.3f}x "
          f"(exact {float(exact_median_final(f, p, b, 240)):.3f}x)")
