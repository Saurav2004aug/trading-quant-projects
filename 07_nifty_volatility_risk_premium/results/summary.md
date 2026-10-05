# NIFTY volatility risk premium: results

## 1. India VIX vs subsequent realised volatility
2020-01-02 to 2026-03-10, 1515 days.

| | |
|---|---:|
| Mean VIX / mean realised vol (next 21 days) | 17.34 / 15.18 |
| Mean premium, vol points (Newey-West t) | 2.16 (2.65) |
| Days VIX above subsequent realised | 79% |
| Worst day | -64.50 on 2020-03-05 |
| Forecast R²: VIX / trailing realised | 0.36 / 0.26 |
| Forecast RMSE: VIX / trailing realised | 8.54 / 10.08 |

## 2. India VIX vs real 30-day ATM implied vol (NSE option prices)
1200 days. VIX - ATM IV30: mean 1.44, median 1.27, 10th-90th pct 0.56 to 2.43; correlation 0.990.
ATM IV30 - subsequent realised: mean 1.78, median 1.66, positive on 69% of days.

## 3. Short monthly ATM straddle at real NSE prices (per cycle, % of forward)

| | Delta-hedged | Unhedged |
|---|---:|---:|
| cycles | 58 | 58 |
| mean_pnl_pct | 0.25 | 0.01 |
| median_pnl_pct | 0.28 | 0.38 |
| t_stat | 2.07 | 0.04 |
| win_rate | 0.62 | 0.53 |
| worst_pct | -2.91 | -6.38 |
| best_pct | 2.46 | 5.97 |
| sharpe | 0.94 | 0.02 |
| skew | -0.63 | -0.47 |
| mean_premium_pct | 3.65 | 3.65 |
| mean_iv_entry | 16.30 | 16.30 |
| mean_rv_realised | 14.21 | 14.21 |
| mean_cost_pct | 0.05 | 0.02 |

Period 2020-05-04 to 2025-04-30. Worst hedged cycle entered 2022-01-28. Share of option marks from days a leg did not trade: 0.8%.

Cost sensitivity (delta-hedged):

|   opt_cost_pct_of_premium |   hedge_bps |   mean_pnl_pct |   t_stat |   sharpe |
|--------------------------:|------------:|---------------:|---------:|---------:|
|                      0.00 |        0.00 |           0.30 |     2.48 |     1.13 |
|                      0.50 |        1.00 |           0.25 |     2.07 |     0.94 |
|                      2.00 |        1.00 |           0.19 |     1.61 |     0.73 |
|                      5.00 |        3.00 |           0.02 |     0.16 |     0.07 |

## 4. Model on the longer 2020-2026 VIX sample (includes the March 2020 crash)

|                                            |   traded |   mean_pnl_pct |   win_rate |   worst_pct |   sharpe |   skew |
|:-------------------------------------------|---------:|---------------:|-----------:|------------:|---------:|-------:|
| VIX as IV                                  |  1531.00 |           0.55 |       0.76 |      -10.86 |     1.39 |  -2.04 |
| VIX - 1.44 (measured gap)                  |  1531.00 |           0.24 |       0.65 |      -11.04 |     0.62 |  -1.96 |
| VIX - 1.44, unhedged                       |  1531.00 |          -0.16 |       0.56 |      -34.57 |    -0.15 |  -3.28 |
| VIX - 1.44, sell only if VIX > trailing RV |  1298.00 |           0.17 |       0.66 |      -11.04 |     0.48 |  -2.97 |

Greeks attribution explains 90% of the variance of model cycle P&L.

|   atm_iv_below_vix_pts |   mean_pnl_pct |   win_rate |   sharpe |
|-----------------------:|---------------:|-----------:|---------:|
|                   0.00 |           0.55 |       0.76 |     1.39 |
|                   0.50 |           0.44 |       0.72 |     1.12 |
|                   1.00 |           0.34 |       0.69 |     0.85 |
|                   1.50 |           0.23 |       0.65 |     0.58 |
|                   2.00 |           0.12 |       0.61 |     0.31 |
|                   2.50 |           0.02 |       0.57 |     0.04 |
|                   3.00 |          -0.09 |       0.53 |    -0.23 |