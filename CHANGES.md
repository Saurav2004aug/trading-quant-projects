# Changes

## v3: real-data projects

**New: 07 NIFTY volatility risk premium.** Real NIFTY, India VIX and NSE F&O
bhavcopy data (2020–2026):
- implied vs realised volatility with Newey-West inference;
- a parity-implied forward (validated against futures at 1.7 bps);
- ATM IV, and a straddle backtest at real option prices;
- a model-vs-real reconciliation. It exposed a stale-forward hedging bug in
  the first version, which is now fixed and tested.

**New: 08 Order-book imbalance.** NASDAQ LOBSTER AAPL data:
- order-flow imbalance, depth imbalance and microprice;
- explain-vs-predict out of sample;
- event-time next-move probabilities;
- taker economics after the spread, and adverse selection after trades.

**02 Pairs trading on real data.**
- Walk-forward results on Federal Reserve FX rates (EUR/GBP, AUD/NZD, a
  5-currency universe, and a 20-year run): no tradable edge, reported as such.
- New `--cols` and `--start` options for the CSV loader.

**Repository.** Top-level README reorganised around the real-data projects;
CI and `reproduce.sh` extended to eight projects (83 tests).

# Changes in v2

A full review found correctness bugs, claims the results did not
support, and missing validation. Every item below has a test or a figure
that demonstrates the fix.

## Whole repository
- Added tests to every project (60 in total), a test runner that needs no
  extra packages, GitHub Actions CI and `reproduce.sh`.
- Scripts resolve paths from their own location, so they run from any directory.
- One colour-blind-checked plotting style across all figures.
- Removed `__pycache__`, `__MACOSX` and other build artefacts; added `.gitignore`.

## 01 Options
- Fixed terminology: realised-vs-implied volatility P&L was called "vol of vol".
- Added the closed-form expected hedged P&L and a test that the simulation matches it.
- Added transaction costs and a hedging-frequency study with an interior optimum.
- Added puts throughout, implied volatility, a CRR binomial tree for American options,
  and finite-difference tests for every Greek.

## 02 Pairs trading
- **Bug:** per-trade statistics excluded entry costs, overstating win rate and profit factor.
  Trade P&L now reconciles exactly with daily P&L (tested).
- **Bug:** sizing used |β|, so a negatively related pair was traded as the wrong combination.
  Fixed, and tested on a β < 0 pair.
- The backtest traded even when the cointegration test failed. Pairs are now traded only
  when they pass in their own training window.
- Removed the normal-approximation p-value, which is invalid for the ADF distribution.
- Critical values now include MacKinnon's finite-sample correction; ADF lags are chosen by AIC.
- Test size and power verified by Monte Carlo.
- Replaced two separate backtesters with one engine: next-close execution, stop-loss,
  and no-look-ahead tests.
- Added walk-forward re-screening and re-fitting, universe screening at the 1% level,
  a CSV loader and a generated `summary.md` per run.

## 03 Risk of ruin
- The claim that "the median trader does worse at higher risk" was false within the
  tested range: the sweep stopped at 10%, below the 17.5% Kelly fraction.
  The sweep now runs to 2.6× Kelly and shows the peak at Kelly and the collapse beyond it.
- Switched to a realistic edge (+0.125R instead of +0.35R). The old parameters produced
  a median of 170,000× in 500 trades.
- Simulated medians are checked against the exact binomial formula.
- Added drawdown probability and a prop-firm challenge simulator
  (target, static/trailing max loss, daily limit, time limit).

## 04 Trade journal
- The old generator built large edges in and the analysis then "found" them.
  The new generator has modest, noisy edges, session derived from timestamps,
  and realistic stop-outs.
- Sharpe was computed only on days with trades; it now includes flat days.
- Added bootstrap confidence intervals, permutation tests and Benjamini-Hochberg correction,
  plus a power / false-discovery study.
- Added an MT5 export parser and column-mapping loader.

## 05 Predictive maintenance
- Raw vibration alone scored AUC 0.992 against the model's 0.995, so the model added nothing.
  The fleet now has machine-to-machine variation, load confounding, abrupt failures,
  drift and missing data.
- Models are compared against a no-model baseline with 5-fold machine-grouped CV;
  results are reported as mean ± std across folds.
- Added machine-level alarm evaluation (caught / missed / false alarm, warning time)
  with a nested, cost-based threshold.
- **Firmware:** it sampled the accelerometer at 1 Hz, which cannot measure vibration.
  It now samples 1,024 readings at about 1 kHz with gravity removal, computes
  RMS / peak / crest / kurtosis, reads temperature without blocking, and reconnects safely.
  The feature maths is mirrored in Python and unit-tested.
- The README said degradation starts in the "final ~30% of life", but the code used 45%.
  The documentation now matches the code.

## 06 ETL
- The generator never produced the mixed-case names the README described,
  and prices were random from 1 to 2,000 for every instrument. Both fixed.
- The 5×-median outlier rule flagged 16 rows for 8 real outliers.
  It is replaced by a robust z-score; detection is now scored against injected ground truth.
- Rows are quarantined with a reason code instead of dropped.
- Added a primary key, CHECK constraints, a run log with source hash, idempotent loads
  and row-count reconciliation.
- The dashboard previously could not be tested. It now runs under a stub in the tests
  and has a data-quality tab.
