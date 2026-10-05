"""
Variance risk premium (VRP) on NIFTY: is implied volatility (India VIX)
systematically higher than the volatility that is subsequently realised?

Definitions (daily data)
------------------------
realised vol, forward   RV_fwd(t) = sqrt(252/h * sum_{i=1..h} r_{t+i}^2)
                        with h = 21 trading days (about the 30 calendar days
                        India VIX looks ahead) and r = close-to-close log return
realised vol, trailing  RV_past(t) = same over r_{t-h+1..t}  (known at t)
VRP (vol points)        VIX(t) - RV_fwd(t)
VRP (variance)          VIX(t)^2 - RV_fwd(t)^2   (what a variance swap pays)

Forecast evaluation
-------------------
Mincer-Zarnowitz regressions  RV_fwd = a + b * forecast  for VIX and for the
trailing realised vol. The forward windows overlap, so standard errors use
Newey-West with h-1 lags; ordinary OLS errors would be far too small.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from data import TRADING_DAYS

H = 21


def realised_vol(ret: pd.Series, h: int = H, forward: bool = False) -> pd.Series:
    """Annualised realised vol in % from squared log returns (zero-mean)."""
    rv = np.sqrt(ret.pow(2).rolling(h).sum() * TRADING_DAYS / h) * 100
    return rv.shift(-h) if forward else rv


def build(df: pd.DataFrame, h: int = H) -> pd.DataFrame:
    out = df.copy()
    out["rv_fwd"] = realised_vol(df["ret"], h, forward=True)
    out["rv_past"] = realised_vol(df["ret"], h)
    out["vrp_vol"] = out["vix"] - out["rv_fwd"]
    out["vrp_var"] = (out["vix"] ** 2 - out["rv_fwd"] ** 2) / 100      # in vol^2 / 100 units
    out["vrp_exante"] = out["vix"] - out["rv_past"]                     # known at t
    return out


def newey_west_ols(y: np.ndarray, X: np.ndarray, lags: int):
    """OLS coefficients with Newey-West (Bartlett kernel) standard errors."""
    X = np.column_stack([np.ones(len(y)), X])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    u = y - X @ beta
    xu = X * u[:, None]
    S = xu.T @ xu
    for L in range(1, lags + 1):
        w = 1 - L / (lags + 1)
        G = xu[L:].T @ xu[:-L]
        S += w * (G + G.T)
    XtX_inv = np.linalg.inv(X.T @ X)
    cov = XtX_inv @ S @ XtX_inv
    r2 = 1 - (u @ u) / ((y - y.mean()) @ (y - y.mean()))
    return beta, np.sqrt(np.diag(cov)), r2


def summary(v: pd.DataFrame, h: int = H) -> dict:
    d = v.dropna(subset=["vix", "rv_fwd", "rv_past"])
    beta_v, se_v, r2_v = newey_west_ols(d.rv_fwd.values, d.vix.values, h - 1)
    beta_p, se_p, r2_p = newey_west_ols(d.rv_fwd.values, d.rv_past.values, h - 1)
    # Mean VRP with Newey-West standard error (regression on a constant).
    X0 = np.empty((len(d), 0))
    m, se_m, _ = newey_west_ols(d.vrp_vol.values, X0, h - 1)
    return {
        "start": d.index[0].date(), "end": d.index[-1].date(), "days": len(d),
        "mean_vix": d.vix.mean(), "mean_rv_fwd": d.rv_fwd.mean(),
        "mean_vrp_vol": m[0], "vrp_vol_t_nw": m[0] / se_m[0],
        "median_vrp_vol": d.vrp_vol.median(),
        "share_vix_above_rv": (d.vrp_vol > 0).mean(),
        "worst_vrp_vol": d.vrp_vol.min(), "worst_vrp_date": d.vrp_vol.idxmin().date(),
        "mz_vix_a": beta_v[0], "mz_vix_b": beta_v[1], "mz_vix_b_se": se_v[1], "mz_vix_r2": r2_v,
        "mz_past_a": beta_p[0], "mz_past_b": beta_p[1], "mz_past_b_se": se_p[1], "mz_past_r2": r2_p,
        "rmse_vix": np.sqrt(((d.vix - d.rv_fwd) ** 2).mean()),
        "rmse_past": np.sqrt(((d.rv_past - d.rv_fwd) ** 2).mean()),
    }


def by_year(v: pd.DataFrame) -> pd.DataFrame:
    d = v.dropna(subset=["vix", "rv_fwd"])
    g = d.groupby(d.index.year)
    return pd.DataFrame({"mean_vix": g.vix.mean(), "mean_rv_fwd": g.rv_fwd.mean(),
                         "mean_vrp": g.vrp_vol.mean(), "share_vix_above_rv": g.vrp_vol.apply(lambda s: (s > 0).mean()),
                         "days": g.size()})


if __name__ == "__main__":
    from data import load
    v = build(load())
    for k, val in summary(v).items():
        print(f"{k:>20}: {val:.3f}" if isinstance(val, float) else f"{k:>20}: {val}")
    print(by_year(v).round(2).to_string())
