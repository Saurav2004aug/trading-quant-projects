# Results: EURUSD=X / GBPUSD=X (10y, Yahoo Finance)

Default parameters: lookback 30, entry |z| 2.0, exit |z| 0.5, stop |z| 4.0, cost 2.0 bps per side, execution at next close.

| Metric | Walk-forward | Single 60/40 split |
|---|---|---|
| net_return | 1.3% | 5.5% |
| annual_return | 0.2% | 1.3% |
| annual_vol | 0.5% | 1.4% |
| sharpe | 0.34 | 0.93 |
| t_stat_mean | 0.98 | 1.90 |
| max_drawdown | 0.7% | 1.0% |
| trades | 5 | 30 |
| win_rate | 80.0% | 76.7% |
| profit_factor | 17.44 | 4.97 |
| avg_hold_days | 9.20 | 10.63 |
| stop_outs | 0 | 0 |
| costs | 0.2% | 1.2% |
| years | 8.32 | 4.13 |

Share of walk-forward blocks where at least one pair passed the cointegration screen: 9%

## Cost sensitivity (walk-forward)

| cost_bps | net_return | sharpe | trades |
|---|---|---|---|
| 0 | 0.015 | 0.391 | 5 |
| 1 | 0.014 | 0.365 | 5 |
| 2 | 0.013 | 0.339 | 5 |
| 5 | 0.01 | 0.261 | 5 |
| 10 | 0.005 | 0.131 | 5 |

## Parameter stability (walk-forward Sharpe)

| lookback | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|
| 20 | 0.29 | 0.26 | 0.38 | 0.31 |
| 30 | 0.3 | 0.34 | 0.44 | nan |
| 60 | 0.62 | 0.49 | 0.89 | 0.67 |
| 90 | 0.5 | 0.34 | 0.61 | 0.61 |

A t-stat below ~2 means the mean daily return is not statistically distinguishable from zero over this sample.