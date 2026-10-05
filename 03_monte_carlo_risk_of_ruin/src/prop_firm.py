"""
Prop-firm evaluation ("challenge") simulator.

Typical rules (configurable in `ChallengeRules`):
  - profit target:     +10% of the initial balance -> PASS
  - max loss:          equity may never fall below 90% of the initial balance
                       ("static"), or below the running peak minus 10%
                       ("trailing") -> FAIL
  - daily loss limit:  equity may not fall more than 5% of the initial
                       balance below the day's starting equity -> FAIL
  - time limit:        not reaching the target in `max_days` -> TIMEOUT
Risk per trade is a fixed percentage of the *initial* balance, which is
how most funded traders size (fixed dollar risk).

The point: pass probability is not monotonic in risk. Too small and you
time out; too large and you hit the loss limits. There is an interior
optimum that depends on the edge and the rules, and it is far below Kelly.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from simulate import sample_r_multiples


@dataclass
class ChallengeRules:
    profit_target: float = 0.10
    max_loss: float = 0.10
    daily_loss: float = 0.05
    drawdown_type: str = "static"      # "static" or "trailing"
    max_days: int = 30
    trades_per_day: int = 3


def simulate_challenge(risk_per_trade: float, rules: ChallengeRules = ChallengeRules(),
                       win_rate: float = 0.45, payoff: float = 1.5, empirical_r=None,
                       n_paths: int = 20_000, seed: int | None = None) -> pd.Series:
    """Fraction of paths ending in each outcome, plus average days to pass."""
    rng = np.random.default_rng(seed)
    n_trades = rules.max_days * rules.trades_per_day
    r = sample_r_multiples(n_paths, n_trades, rng, win_rate, payoff, empirical_r)
    pnl = r * risk_per_trade                              # in units of initial balance

    equity = np.ones(n_paths)
    peak = np.ones(n_paths)
    status = np.zeros(n_paths, dtype=int)                 # 0 active, 1 pass, 2 max-loss, 3 daily
    day_passed = np.full(n_paths, np.nan)

    for day in range(rules.max_days):
        day_start = equity.copy()
        for k in range(rules.trades_per_day):
            active = status == 0
            if not active.any():
                break
            equity = np.where(active, equity + pnl[:, day * rules.trades_per_day + k], equity)
            peak = np.maximum(peak, equity)
            floor = (1 - rules.max_loss) if rules.drawdown_type == "static" \
                else np.minimum(peak - rules.max_loss, 1.0)   # trailing stops trailing at start
            eps = 1e-12                                   # touching a limit counts as a breach
            hit_daily = active & (equity <= day_start - rules.daily_loss + eps)
            hit_max = active & (equity <= floor + eps) & ~hit_daily
            hit_target = active & (equity >= 1 + rules.profit_target - eps) & ~hit_daily & ~hit_max
            status[hit_daily] = 3
            status[hit_max] = 2
            status[hit_target] = 1
            day_passed[hit_target] = day + 1

    return pd.Series({
        "risk_per_trade": risk_per_trade,
        "pass": (status == 1).mean(),
        "fail_max_loss": (status == 2).mean(),
        "fail_daily_loss": (status == 3).mean(),
        "timeout": (status == 0).mean(),
        "avg_days_to_pass": np.nanmean(day_passed) if (status == 1).any() else np.nan,
    })


def challenge_sweep(risks=None, rules: ChallengeRules = ChallengeRules(), **kw) -> pd.DataFrame:
    if risks is None:
        risks = np.round(np.arange(0.00125, 0.05001, 0.00125), 5)
    return pd.DataFrame([simulate_challenge(r, rules, seed=i, **kw) for i, r in enumerate(risks)])


if __name__ == "__main__":
    pd.set_option("display.width", 140)
    for dd in ("static", "trailing"):
        rules = ChallengeRules(drawdown_type=dd)
        df = challenge_sweep(rules=rules)
        best = df.loc[df["pass"].idxmax()]
        print(f"\n{dd} drawdown, 45% win rate, 1.5R payoff, {rules.trades_per_day} trades/day, "
              f"{rules.max_days} days")
        print(df.round(3).to_string(index=False))
        print(f"-> best risk per trade {best.risk_per_trade:.2%}, pass rate {best['pass']:.1%}")
