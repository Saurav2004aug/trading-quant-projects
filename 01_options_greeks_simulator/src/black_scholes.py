"""
Black-Scholes-Merton pricing, Greeks and implied volatility.

European options on an underlying with continuous dividend yield q.
All functions are vectorised: any argument can be a numpy array.

Units
-----
- vega:  per 1.00 change in sigma (divide by 100 for "per vol point")
- theta: per year (divide by 365 for per calendar day)
- rho:   per 1.00 change in r (divide by 100 for "per 1%")
`all_greeks` returns the conventional trader units.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import brentq
from scipy.stats import norm

_EPS = 1e-12


def _check_type(option_type: str) -> None:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")


def _prep(S, K, T, sigma):
    S = np.asarray(S, dtype=float)
    K = np.asarray(K, dtype=float)
    T = np.maximum(np.asarray(T, dtype=float), _EPS)       # guard T -> 0
    sigma = np.maximum(np.asarray(sigma, dtype=float), _EPS)
    return S, K, T, sigma


def d1_d2(S, K, T, r, sigma, q=0.0):
    S, K, T, sigma = _prep(S, K, T, sigma)
    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    return d1, d1 - sigma * sqrt_t


def price(S, K, T, r, sigma, option_type="call", q=0.0):
    _check_type(option_type)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S_, K_, T_, _ = _prep(S, K, T, sigma)
    df_r, df_q = np.exp(-r * T_), np.exp(-q * T_)
    if option_type == "call":
        return S_ * df_q * norm.cdf(d1) - K_ * df_r * norm.cdf(d2)
    return K_ * df_r * norm.cdf(-d2) - S_ * df_q * norm.cdf(-d1)


def delta(S, K, T, r, sigma, option_type="call", q=0.0):
    _check_type(option_type)
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    _, _, T_, _ = _prep(S, K, T, sigma)
    df_q = np.exp(-q * T_)
    return df_q * norm.cdf(d1) if option_type == "call" else df_q * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    S_, _, T_, sig_ = _prep(S, K, T, sigma)
    return np.exp(-q * T_) * norm.pdf(d1) / (S_ * sig_ * np.sqrt(T_))


def vega(S, K, T, r, sigma, q=0.0):
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    S_, _, T_, _ = _prep(S, K, T, sigma)
    return S_ * np.exp(-q * T_) * norm.pdf(d1) * np.sqrt(T_)


def theta(S, K, T, r, sigma, option_type="call", q=0.0):
    _check_type(option_type)
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    S_, K_, T_, sig_ = _prep(S, K, T, sigma)
    df_r, df_q = np.exp(-r * T_), np.exp(-q * T_)
    decay = -(S_ * df_q * norm.pdf(d1) * sig_) / (2 * np.sqrt(T_))
    if option_type == "call":
        return decay - r * K_ * df_r * norm.cdf(d2) + q * S_ * df_q * norm.cdf(d1)
    return decay + r * K_ * df_r * norm.cdf(-d2) - q * S_ * df_q * norm.cdf(-d1)


def rho(S, K, T, r, sigma, option_type="call", q=0.0):
    _check_type(option_type)
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    _, K_, T_, _ = _prep(S, K, T, sigma)
    df_r = np.exp(-r * T_)
    if option_type == "call":
        return K_ * T_ * df_r * norm.cdf(d2)
    return -K_ * T_ * df_r * norm.cdf(-d2)


def all_greeks(S, K, T, r, sigma, option_type="call", q=0.0) -> dict:
    """Price and Greeks in trader units (vega per vol pt, theta per day, rho per 1%)."""
    return {
        "price": price(S, K, T, r, sigma, option_type, q),
        "delta": delta(S, K, T, r, sigma, option_type, q),
        "gamma": gamma(S, K, T, r, sigma, q),
        "vega": vega(S, K, T, r, sigma, q) / 100,
        "theta": theta(S, K, T, r, sigma, option_type, q) / 365,
        "rho": rho(S, K, T, r, sigma, option_type, q) / 100,
    }


def implied_vol(market_price, S, K, T, r, option_type="call", q=0.0,
                lo=1e-4, hi=5.0) -> float:
    """Invert Black-Scholes for sigma with Brent's method.

    Raises ValueError if the price is outside no-arbitrage bounds.
    """
    intrinsic_lo = price(S, K, T, r, lo, option_type, q)
    intrinsic_hi = price(S, K, T, r, hi, option_type, q)
    if not (intrinsic_lo - 1e-10 <= market_price <= intrinsic_hi + 1e-10):
        raise ValueError("price outside the range attainable by Black-Scholes")
    return brentq(lambda s: price(S, K, T, r, s, option_type, q) - market_price,
                  lo, hi, xtol=1e-10)


if __name__ == "__main__":
    # Hull, Options Futures & Other Derivatives: S=K=100, T=1, r=5%, sigma=20%
    g = all_greeks(S=100, K=100, T=1.0, r=0.05, sigma=0.2, option_type="call")
    for k, v in g.items():
        print(f"{k:>6}: {float(v):.4f}")
    print(f"implied vol round-trip: {implied_vol(float(g['price']), 100, 100, 1.0, 0.05):.6f}")
