"""
Discrete delta-hedging simulator.

A dealer sells one European option at the implied-vol price and
delta-hedges it by trading the underlying at `n_rebalances` evenly spaced
times. The underlying follows GBM with a *realised* volatility that may
differ from the implied vol used for pricing and for the hedge ratio.

What the simulation shows
-------------------------
1. Realised vs implied volatility (gamma P&L). With the hedge ratio taken
   at implied vol and drift equal to r, the expected hedged P&L is
       E[P&L] = e^{rT} * (C(sigma_implied) - C(sigma_realised)),
   so selling vol only pays if it realises below where it was sold.
   `theoretical_mean_pnl` gives this value and the tests check the
   simulation against it.
2. Discretisation error. Hedging at discrete times leaves residual risk
   whose standard deviation shrinks roughly like 1/sqrt(n_rebalances).
3. Transaction costs. Every rebalance pays a proportional cost, which
   grows roughly like sqrt(n_rebalances). Together with (2) this gives an
   interior optimal hedging frequency (see `frequency_study`).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from black_scholes import delta, price


def simulate_gbm_paths(S0: float, mu: float, sigma: float, T: float,
                       n_steps: int, n_paths: int, seed: int | None = None) -> np.ndarray:
    """GBM paths of shape (n_paths, n_steps + 1), exact log-normal stepping."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    log_ret = (mu - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.hstack([np.zeros((n_paths, 1)), np.cumsum(log_ret, axis=1)])
    return S0 * np.exp(log_paths)


def run_hedge_simulation(S0: float = 100.0, K: float = 100.0, T: float = 30 / 365,
                         r: float = 0.05, sigma_implied: float = 0.20,
                         sigma_realised: float = 0.20, n_rebalances: int = 30,
                         n_paths: int = 5000, option_type: str = "call",
                         cost_bps: float = 0.0, seed: int = 42) -> pd.DataFrame:
    """Short one option, delta-hedge at n_rebalances times, return per-path results.

    cost_bps: proportional cost charged on the traded notional |shares * S|
              at every rebalance, including the final unwind.
    Columns: S_T, pnl (net of costs), hedge_costs, n_shares_traded.
    """
    paths = simulate_gbm_paths(S0, r, sigma_realised, T, n_rebalances, n_paths, seed)
    dt = T / n_rebalances
    growth = np.exp(r * dt)
    cost_rate = cost_bps / 10_000

    cash = np.full(n_paths, float(price(S0, K, T, r, sigma_implied, option_type)))
    shares = np.zeros(n_paths)
    costs = np.zeros(n_paths)
    traded = np.zeros(n_paths)

    for i in range(n_rebalances):
        S_t = paths[:, i]
        tau = T - i * dt
        # A short option has delta -Delta, so the hedge holds +Delta shares.
        target = delta(S_t, K, tau, r, sigma_implied, option_type)
        trade = target - shares
        c = np.abs(trade) * S_t * cost_rate
        cash -= trade * S_t + c
        costs += c
        traded += np.abs(trade)
        shares = target
        cash *= growth                       # cash accrues at r until the next rebalance

    S_T = paths[:, -1]
    payoff = np.maximum(S_T - K, 0.0) if option_type == "call" else np.maximum(K - S_T, 0.0)
    unwind_cost = np.abs(shares) * S_T * cost_rate
    costs += unwind_cost
    pnl = cash + shares * S_T - unwind_cost - payoff
    return pd.DataFrame({"S_T": S_T, "pnl": pnl, "hedge_costs": costs,
                         "n_shares_traded": traded})


def theoretical_mean_pnl(S0=100.0, K=100.0, T=30 / 365, r=0.05, sigma_implied=0.20,
                         sigma_realised=0.20, option_type="call") -> float:
    """Expected hedged P&L (continuous hedging, zero costs, drift = r)."""
    c_imp = price(S0, K, T, r, sigma_implied, option_type)
    c_real = price(S0, K, T, r, sigma_realised, option_type)
    return float(np.exp(r * T) * (c_imp - c_real))


def frequency_study(rebalance_grid=(1, 2, 4, 8, 15, 30, 60, 120, 240, 480),
                    cost_bps: float = 5.0, n_paths: int = 6000, seed: int = 11,
                    **kwargs) -> pd.DataFrame:
    """Hedge P&L statistics as a function of rebalancing frequency.

    `p05` (5th-percentile P&L) is the risk-adjusted objective: it rewards
    lower hedging noise and penalises cost drag, and is maximised at an
    interior frequency when costs are non-zero.
    """
    rows = []
    for n in rebalance_grid:
        df = run_hedge_simulation(n_rebalances=n, n_paths=n_paths, cost_bps=cost_bps,
                                  seed=seed, **kwargs)
        rows.append({"n_rebalances": n,
                     "mean_pnl": df.pnl.mean(),
                     "std_pnl": df.pnl.std(),
                     "mean_cost": df.hedge_costs.mean(),
                     "p05": df.pnl.quantile(0.05)})
    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = run_hedge_simulation()
    print("Zero-cost daily hedge, sigma_realised = sigma_implied = 20%")
    print(df["pnl"].describe().round(4))
    print("\nFrequency study at 5 bps cost:")
    fs = frequency_study()
    print(fs.round(4).to_string(index=False))
    best = fs.loc[fs.p05.idxmax()]
    print(f"\nBest 5th-percentile P&L at {int(best.n_rebalances)} rebalances over 30 days")
