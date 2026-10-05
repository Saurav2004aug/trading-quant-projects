"""
Cox-Ross-Rubinstein binomial tree for European and American options.

Used to (1) price American options, where early exercise has value
(puts, or calls on dividend-paying stock), and (2) cross-check the
closed-form Black-Scholes price: the European tree price converges to
Black-Scholes as the number of steps grows.
"""
from __future__ import annotations

import numpy as np


def crr_price(S: float, K: float, T: float, r: float, sigma: float,
              option_type: str = "put", american: bool = True,
              q: float = 0.0, n_steps: int = 500) -> float:
    if option_type not in ("call", "put"):
        raise ValueError("option_type must be 'call' or 'put'")
    dt = T / n_steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp((r - q) * dt) - d) / (u - d)
    if not 0.0 < p < 1.0:
        raise ValueError("risk-neutral probability outside (0,1); increase n_steps")
    disc = np.exp(-r * dt)

    # Terminal stock prices, highest first.
    j = np.arange(n_steps + 1)
    ST = S * u ** (n_steps - j) * d ** j
    sign = 1.0 if option_type == "call" else -1.0
    values = np.maximum(sign * (ST - K), 0.0)

    for step in range(n_steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1 - p) * values[1:])
        if american:
            j = np.arange(step + 1)
            St = S * u ** (step - j) * d ** j
            values = np.maximum(values, sign * (St - K))
    return float(values[0])


def early_exercise_premium(S, K, T, r, sigma, option_type="put", q=0.0, n_steps=500) -> float:
    """American minus European price from the same tree."""
    am = crr_price(S, K, T, r, sigma, option_type, True, q, n_steps)
    eu = crr_price(S, K, T, r, sigma, option_type, False, q, n_steps)
    return am - eu


if __name__ == "__main__":
    from black_scholes import price as bs
    for n in (50, 200, 1000):
        eu = crr_price(100, 100, 1, 0.05, 0.2, "put", american=False, n_steps=n)
        am = crr_price(100, 100, 1, 0.05, 0.2, "put", american=True, n_steps=n)
        print(f"n={n:5d}  EU put {eu:.4f} (BS {float(bs(100,100,1,0.05,0.2,'put')):.4f})  AM put {am:.4f}")
