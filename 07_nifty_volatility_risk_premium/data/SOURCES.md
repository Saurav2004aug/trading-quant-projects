# Data sources

All market data here is real. The files are copies of public exchange data
that third parties committed to GitHub; each was checked against known
market events and against each other (see "Checks").

| File | Content | Origin |
|---|---|---|
| `nifty50_daily.csv` | NIFTY 50 index daily OHLC, 2007-09-17 to 2026-04-13 | github.com/kalilurrahman/NIFTY_50_STOCK_DATA, `NIFTY50_stock_history.csv`, commit 508787e1 (Yahoo Finance ^NSEI history) |
| `india_vix_daily.csv` | India VIX daily OHLC, 2020-01-02 to 2026-08-24 | github.com/tanya459/india-vix-market-data-analysis, `Dataset/IndiaVIX_Cleaned_Dataset.csv`, commit 7f81a403 |
| `nifty_monthly_options_eod.csv.gz` | NIFTY index options, monthly expiries, strikes within ±10% of the near future, 2020-04-13 to 2025-04-30: date, expiry, strike, type, close, contracts, open interest | NSE F&O bhavcopy files in github.com/sajal101agrawal/nse-options-last-5-years, `bhavcopy/extracted/fo*bhav.csv`, commit 3b50394c (1,246 daily files) |
| `nifty_futures_eod.csv` | NIFTY index futures, all expiries, same period | same bhavcopy files |
| `nifty_atm_iv.csv` | Derived: implied forward and ATM IV per date and expiry | computed by `src/options_data.py` (cache; delete to recompute) |

## Checks (automated in `tests/test_vrp.py`)
- NIFTY closed 7,610.25 on 2020-03-23 (COVID low) and 21,884.50 on 2024-06-04 (election result).
- India VIX closed at its record 83.61 on 2020-03-24.
- The forward implied by put-call parity from the option closes matches the NIFTY
  futures settlement price with a median gap of 1.7 bps over 2,781 date-expiry pairs.
- Every India VIX date is also a NIFTY trading date; no duplicate or non-positive prices.

## Replacing with official files
`src/data.py` reads any CSV with a date column and a close column, so the
historical-data downloads from niftyindices.com (NIFTY 50) and nseindia.com
(India VIX) can be dropped in directly. For options, NSE's daily F&O
bhavcopy archive is the primary source.
