# Results: AUDUSD / NZDUSD from fx_daily_fred.csv, 2016-01-04 to 2026-09-25

Default parameters: lookback 30, entry |z| 2.0, exit |z| 0.5, stop |z| 4.0, cost 2.0 bps per side, execution at next close.

| Metric | Walk-forward | Single 60/40 split |
|---|---|---|
| net_return | 0.6% | -2.8% |
| annual_return | 0.1% | -0.7% |
| annual_vol | 0.2% | 1.7% |
| sharpe | 0.41 | -0.40 |
| t_stat_mean | 1.21 | -0.83 |
| max_drawdown | 0.1% | 3.7% |
| trades | 1 | 27 |
| win_rate | 100.0% | 63.0% |
| profit_factor | n/a | 0.61 |
| avg_hold_days | 4.00 | 16.44 |
| stop_outs | 0 | 0 |
| costs | 0.0% | 1.1% |
| years | 8.64 | 4.26 |

Share of walk-forward blocks where at least one pair passed the cointegration screen: 3%

## Cost sensitivity (walk-forward)

| cost_bps | net_return | sharpe | trades |
|---|---|---|---|
| 0 | 0.006 | 0.442 | 1 |
| 1 | 0.006 | 0.426 | 1 |
| 2 | 0.006 | 0.41 | 1 |
| 5 | 0.005 | 0.361 | 1 |
| 10 | 0.004 | 0.279 | 1 |

## Parameter stability (walk-forward Sharpe)

| lookback | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|
| 20 | 0.25 | 0.33 | 0.34 | nan |
| 30 | 0.46 | 0.41 | 0.41 | 0.41 |
| 60 | 0.41 | nan | nan | nan |
| 90 | 0.41 | nan | nan | nan |

A t-stat below ~2 means the mean daily return is not statistically distinguishable from zero over this sample.