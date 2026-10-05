# Results: synthetic regime-switching demo (not market data)

Default parameters: lookback 30, entry |z| 2.0, exit |z| 0.5, stop |z| 4.0, cost 2.0 bps per side, execution at next close.

| Metric | Walk-forward | Single 60/40 split |
|---|---|---|
| net_return | 2.0% | 4.2% |
| annual_return | 0.2% | 1.0% |
| annual_vol | 1.2% | 2.0% |
| sharpe | 0.21 | 0.52 |
| t_stat_mean | 0.60 | 1.03 |
| max_drawdown | 3.2% | 2.5% |
| trades | 13 | 31 |
| win_rate | 76.9% | 77.4% |
| profit_factor | 1.51 | 1.83 |
| avg_hold_days | 12.62 | 10.48 |
| stop_outs | 0 | 0 |
| costs | 0.5% | 1.2% |
| years | 7.92 | 3.97 |

Share of walk-forward blocks where at least one pair passed the cointegration screen: 22%

## Cost sensitivity (walk-forward)

| cost_bps | net_return | sharpe | trades |
|---|---|---|---|
| 0 | 0.025 | 0.271 | 13 |
| 1 | 0.022 | 0.243 | 13 |
| 2 | 0.02 | 0.215 | 13 |
| 5 | 0.012 | 0.13 | 13 |
| 10 | -0.001 | -0.011 | 13 |

## Parameter stability (walk-forward Sharpe)

| lookback | 1.5 | 2.0 | 2.5 | 3.0 |
|---|---|---|---|---|
| 20 | 0.34 | 0.1 | 0.33 | 0.73 |
| 30 | 0.1 | 0.21 | 0.54 | 0.63 |
| 60 | 0.19 | 0.04 | -0.21 | 0.05 |
| 90 | 0.04 | -0.03 | -0 | -0.06 |

A t-stat below ~2 means the mean daily return is not statistically distinguishable from zero over this sample.