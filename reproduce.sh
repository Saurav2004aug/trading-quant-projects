#!/usr/bin/env bash
# Regenerate every dataset, result table and figure, then run all tests.
# All real datasets are bundled; 08's raw files can be re-downloaded with data/fetch_raw.sh.
set -euo pipefail
cd "$(dirname "$0")"

python 01_options_greeks_simulator/src/generate_plots.py
python 02_pairs_trading_stat_arb/src/data_gen.py
python 02_pairs_trading_stat_arb/src/generate_plots.py
python 02_pairs_trading_stat_arb/src/research.py --synthetic
for pair in "EURUSD GBPUSD" "AUDUSD NZDUSD"; do
  python 02_pairs_trading_stat_arb/src/research.py --csv 02_pairs_trading_stat_arb/data/fx_daily_fred.csv \
    --cols $pair --start 2016-01-01 --name "${pair// /_}"
done
python 02_pairs_trading_stat_arb/src/research.py --csv 02_pairs_trading_stat_arb/data/fx_daily_fred.csv \
  --cols EURUSD GBPUSD AUDUSD NZDUSD USDCAD --start 2016-01-01 --name fx_universe
python 02_pairs_trading_stat_arb/src/research.py --csv 02_pairs_trading_stat_arb/data/fx_daily_fred.csv \
  --cols EURUSD GBPUSD --start 2006-01-01 --train-days 1260 --lookback 60 --name EURUSD_GBPUSD_5y_train
python 03_monte_carlo_risk_of_ruin/src/generate_plots.py
python 04_prop_firm_trade_journal/src/generate_trade_log.py
python 04_prop_firm_trade_journal/src/generate_plots.py
python 05_iot_predictive_maintenance/src/simulate_sensor_data.py
python 05_iot_predictive_maintenance/src/generate_plots.py
python 06_etl_pipeline_dashboard/src/generate_raw_data.py
python 06_etl_pipeline_dashboard/src/generate_plots.py
python 07_nifty_volatility_risk_premium/src/generate_plots.py
python 08_order_book_imbalance/src/generate_plots.py
python run_tests.py
