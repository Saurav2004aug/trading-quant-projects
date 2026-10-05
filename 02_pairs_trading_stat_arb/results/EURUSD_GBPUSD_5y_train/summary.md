# Results: EURUSD / GBPUSD from fx_daily_fred.csv, 2006-01-03 to 2026-09-25

Default parameters: lookback 60, entry |z| 2.0, exit |z| 0.5, stop |z| 4.0, cost 2.0 bps per side, execution at next close.

| Metric | Walk-forward | Single 60/40 split |
|---|---|---|
| net_return | 2.1% | 0.0% |
| annual_return | 0.1% | 0.0% |
| annual_vol | 0.6% | 0.0% |
| sharpe | 0.21 | n/a |
| t_stat_mean | 0.85 | n/a |
| max_drawdown | 2.9% | 0.0% |
| trades | 13 | 0 |
| win_rate | 69.2% | n/a |
| profit_factor | 2.22 | n/a |
| avg_hold_days | 14.69 | n/a |
| stop_outs | 0 | 0 |
| costs | 0.5% | 0.0% |
| years | 15.62 | 8.25 |

Share of walk-forward blocks where at least one pair passed the cointegration screen: 13%

## Cost sensitivity (walk-forward)

| cost_bps | net_return | sharpe | trades |
|---|---|---|---|
| 0 | 0.027 | 0.267 | 13 |
| 1 | 0.024 | 0.241 | 13 |
| 2 | 0.021 | 0.214 | 13 |
| 5 | 0.014 | 0.135 | 13 |
| 10 | 0.001 | 0.006 | 13 |

## Parameter stability (walk-forward Sharpe)

| lookback | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|
| 20 | 0.24 | 0.29 | 0.32 | -0.25 |
| 30 | 0.28 | 0.42 | 0.45 | -0.08 |
| 60 | 0.1 | 0.21 | 0.2 | nan |
| 90 | -0.15 | -0.07 | 0.13 | 0.35 |

A t-stat below ~2 means the mean daily return is not statistically distinguishable from zero over this sample.