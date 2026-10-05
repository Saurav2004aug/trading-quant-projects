"""
Engle-Granger cointegration test implemented from scratch (numpy only).

Step 1  OLS:  B_t = alpha + beta * A_t + e_t          (hedge ratio beta)
Step 2  ADF on the residual e_t. Reject the unit-root null  =>  cointegrated.

Why two sets of critical values
-------------------------------
The residual in step 2 comes from a regression that was fitted to make it
as small (and as stationary-looking) as possible. Using ordinary ADF
critical values on it rejects the null far too often: two independent
random walks get flagged as "cointegrated" roughly 3x more often than
the nominal 5% rate. `tests/test_cointegration.py` measures this with a
Monte Carlo size test, and `plots/eg_size_test.png` shows it.

Critical values are MacKinnon (2010) response-surface values,
    cv(T) = b0 + b1 / T + b2 / T^2,
which include the finite-sample correction.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# MacKinnon (2010), constant, no trend. N = number of I(1) variables.
_MACKINNON = {
    1: {"1%": (-3.43035, -6.5393, -16.786), "5%": (-2.86154, -2.8903, -4.234),
        "10%": (-2.56677, -1.5384, -2.809)},
    2: {"1%": (-3.89644, -10.9519, -22.527), "5%": (-3.33613, -6.1101, -6.823),
        "10%": (-3.04445, -4.2412, -2.720)},
}


def critical_values(n_obs: int, n_vars: int = 1) -> dict[str, float]:
    """Finite-sample ADF (n_vars=1) or Engle-Granger (n_vars=2) critical values."""
    return {k: b0 + b1 / n_obs + b2 / n_obs**2 for k, (b0, b1, b2) in _MACKINNON[n_vars].items()}


def ols(y: np.ndarray, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    coef, *_ = np.linalg.lstsq(X, y, rcond=None)
    return coef, y - X @ coef


def estimate_hedge_ratio(asset_a: pd.Series, asset_b: pd.Series) -> dict:
    """Fit B = alpha + beta*A. Spread = B - alpha - beta*A."""
    X = np.column_stack([np.ones(len(asset_a)), np.asarray(asset_a, float)])
    (alpha, beta), resid = ols(np.asarray(asset_b, float), X)
    return {"alpha": float(alpha), "beta": float(beta),
            "spread": pd.Series(resid, index=asset_a.index, name="spread")}


def _adf_regression(y: np.ndarray, lags: int, start: int):
    """Regress dy_t on [1, y_{t-1}, dy_{t-1..t-lags}] for t >= start (common sample)."""
    dy = np.diff(y)
    rows = np.arange(start, len(dy))
    X = [np.ones(len(rows)), y[rows]]
    X += [dy[rows - k] for k in range(1, lags + 1)]
    X = np.column_stack(X)
    target = dy[rows]
    coef, resid = ols(target, X)
    n, k = X.shape
    sigma2 = resid @ resid / (n - k)
    se = np.sqrt(sigma2 * np.linalg.inv(X.T @ X)[1, 1])
    aic = n * np.log(resid @ resid / n) + 2 * k
    return coef[1] / se, aic, n


def adf_test(series, max_lag: int | None = None, autolag: str | None = "AIC",
             n_vars: int = 1) -> dict:
    """Augmented Dickey-Fuller test (constant, no trend).

    max_lag: default Schwert rule 12*(n/100)^(1/4).
    autolag: "AIC" picks the lag in [0, max_lag] minimising AIC on a common
             sample (the statsmodels convention); None uses max_lag directly.
    n_vars:  1 for a plain ADF test, 2 when testing Engle-Granger residuals.
    """
    y = np.asarray(series, dtype=float)
    if max_lag is None:
        max_lag = int(np.floor(12 * (len(y) / 100) ** 0.25))
    if autolag == "AIC":
        aics = [_adf_regression(y, p, max_lag)[1] for p in range(max_lag + 1)]
        lags = int(np.argmin(aics))
    else:
        lags = max_lag
    t_stat, _, n_obs = _adf_regression(y, lags, lags)
    cv = critical_values(n_obs, n_vars)
    return {"t_stat": float(t_stat), "lags_used": lags, "n_obs": n_obs,
            "critical_values": cv, "reject_5pct": bool(t_stat < cv["5%"])}


def half_life(spread: pd.Series) -> float:
    """Mean-reversion half-life (in bars) from an AR(1) fit of the spread."""
    s = np.asarray(spread, dtype=float)
    X = np.column_stack([np.ones(len(s) - 1), s[:-1]])
    (_, phi), _ = ols(np.diff(s), X)
    return float(-np.log(2) / np.log1p(phi)) if -1 < phi < 0 else float("inf")


def engle_granger_test(asset_a: pd.Series, asset_b: pd.Series,
                       significance: str = "5%") -> dict:
    hedge = estimate_hedge_ratio(asset_a, asset_b)
    adf = adf_test(hedge["spread"].values, n_vars=2)
    return {**hedge, "adf": adf,
            "is_cointegrated": adf["t_stat"] < adf["critical_values"][significance],
            "half_life": half_life(hedge["spread"])}


if __name__ == "__main__":
    from data_gen import load_cointegrated_pair, load_uncorrelated_pair

    for name, loader in [("Cointegrated pair", load_cointegrated_pair),
                         ("Independent random walks", load_uncorrelated_pair)]:
        df = loader()
        res = engle_granger_test(df["asset_a"], df["asset_b"])
        print(f"\n{name}")
        print(f"  beta               {res['beta']:.4f}")
        print(f"  ADF t-stat         {res['adf']['t_stat']:.3f}  (lags {res['adf']['lags_used']})")
        print(f"  EG 5% critical     {res['adf']['critical_values']['5%']:.3f}")
        print(f"  half-life (days)   {res['half_life']:.1f}")
        print(f"  cointegrated at 5% {res['is_cointegrated']}")
