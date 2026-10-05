# Results: EURUSD / GBPUSD / AUDUSD / NZDUSD / USDCAD from fx_daily_fred.csv, 2016-01-04 to 2026-09-25

Default parameters: lookback 30, entry |z| 2.0, exit |z| 0.5, stop |z| 4.0, cost 2.0 bps per side, execution at next close.

| Metric | Walk-forward | Single 60/40 split |
|---|---|---|
| net_return | -1.8% | 0.5% |
| annual_return | -0.2% | 0.1% |
| annual_vol | 1.1% | 1.4% |
| sharpe | -0.18 | 0.08 |
| t_stat_mean | -0.53 | 0.17 |
| max_drawdown | 4.5% | 2.5% |
| trades | 8 | 63 |
| win_rate | 62.5% | 61.9% |
| profit_factor | 0.55 | 1.08 |
| avg_hold_days | 13.25 | 12.30 |
| stop_outs | 0 | 1 |
| costs | 0.3% | 1.3% |
| years | 8.64 | 4.26 |

Share of walk-forward blocks where at least one pair passed the cointegration screen: 11%

## Cost sensitivity (walk-forward)

| cost_bps | net_return | sharpe | trades |
|---|---|---|---|
| 0 | -0.015 | -0.149 | 8 |
| 1 | -0.016 | -0.166 | 8 |
| 2 | -0.018 | -0.182 | 8 |
| 5 | -0.023 | -0.231 | 8 |
| 10 | -0.031 | -0.311 | 8 |

## Parameter stability (walk-forward Sharpe)

| lookback | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|
| 20 | -0.12 | -0.14 | -0.11 | 0.26 |
| 30 | -0.2 | -0.18 | -0.25 | 0.22 |
| 60 | -0.31 | -0.29 | -0.28 | -0.33 |
| 90 | -0.49 | -0.36 | -0.41 | -0.39 |

A t-stat below ~2 means the mean daily return is not statistically distinguishable from zero over this sample.