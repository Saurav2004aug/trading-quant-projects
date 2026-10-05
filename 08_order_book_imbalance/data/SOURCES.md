# Data source

**LOBSTER sample file: AAPL, NASDAQ, 21 June 2012, 09:30-16:00, 10 levels.**
LOBSTER (lobsterdata.com) reconstructs the full limit order book from
NASDAQ TotalView-ITCH messages. The AAPL sample is one of LOBSTER's free
sample files (https://lobsterdata.com/info/DataSamples.php).

- Raw files (not bundled, 110 MB): `AAPL_2012-06-21_34200000_57600000_message_10.csv`
  (SHA-256 6562394b...5411e5) and `..._orderbook_10.csv` (SHA-256 ed754500...6ff5d).
  Obtained from two independent GitHub mirrors with identical hashes:
  github.com/amaiti2/queue-aware-lob-alpha (commit 5dcfdb1b) and
  github.com/sohaibelkarmi/High-Frequency-Trading-Simulator (commit 780809e6).
  `fetch_raw.sh` downloads them and rebuilds the processed file (verified byte-identical).
- Bundled: `aapl_2012-06-21_events_L3.csv.gz`, all 400,391 events with the
  message fields and the top 3 book levels, prices in dollars (`src/lobster.py`).

Integrity checks (tested): time is monotonic, no crossed books, price
levels sorted on both sides, all prices on the $0.01 grid, event-type counts
match the raw file.
