"""
Load LOBSTER limit-order-book data (NASDAQ TotalView-ITCH reconstructions).

LOBSTER format
--------------
message file, one row per event:
    time (seconds after midnight), type, order id, size, price (x10000), direction
    type: 1 new limit order, 2 partial cancel, 3 full cancel,
          4 execution of a visible order, 5 execution of a hidden order,
          6 cross trade, 7 trading halt
    direction: +1 buy limit order, -1 sell limit order (for executions this is
               the side of the *resting* order, so -1 = a buyer hit the ask)
orderbook file, row i = the book *after* event i:
    ask price 1, ask size 1, bid price 1, bid size 1, ask price 2, ...
    empty levels hold dummy prices +-9999999999

Prices are converted to dollars; one tick is $0.01.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
PROCESSED = DATA / "aapl_2012-06-21_events_L3.csv.gz"
TICK = 0.01
MSG_COLS = ["time", "type", "order_id", "size", "price", "direction"]


def book_columns(levels: int) -> list[str]:
    return [f"{s}{i}" for i in range(1, levels + 1) for s in ("ask_p", "ask_q", "bid_p", "bid_q")]


def load_raw(message_csv: Path, orderbook_csv: Path, keep_levels: int = 3) -> pd.DataFrame:
    msg = pd.read_csv(message_csv, header=None, names=MSG_COLS)
    n_cols = pd.read_csv(orderbook_csv, header=None, nrows=1).shape[1]
    book = pd.read_csv(orderbook_csv, header=None, names=book_columns(n_cols // 4))
    if len(msg) != len(book):
        raise ValueError("message and orderbook files have different lengths")
    book = book[book_columns(keep_levels)].astype(float)
    for c in book.columns:
        if "_p" in c:
            book.loc[book[c].abs() >= 9_999_999_999, c] = np.nan
            book[c] = book[c] / 10_000
    df = pd.concat([msg.drop(columns="order_id"), book], axis=1)
    df["price"] = df["price"] / 10_000
    return df


def validate(df: pd.DataFrame) -> dict:
    """Integrity checks a reconstructed book must pass."""
    return {
        "events": len(df),
        "time_monotonic": bool(df["time"].is_monotonic_increasing),
        "crossed_books": int((df["ask_p1"] <= df["bid_p1"]).sum()),
        "ask_levels_sorted": bool((df["ask_p2"] > df["ask_p1"]).all()),
        "bid_levels_sorted": bool((df["bid_p2"] < df["bid_p1"]).all()),
        "prices_on_tick_grid": bool(np.allclose(np.round(df["ask_p1"] / TICK), df["ask_p1"] / TICK)),
        "event_types": df["type"].value_counts().sort_index().to_dict(),
    }


def load(path: Path = PROCESSED) -> pd.DataFrame:
    return pd.read_csv(path)


if __name__ == "__main__":
    import sys
    raw_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else DATA / "raw"
    msg = next(raw_dir.glob("*message_*.csv"))
    ob = next(raw_dir.glob("*orderbook_*.csv"))
    df = load_raw(msg, ob)
    print(validate(df))
    df.to_csv(PROCESSED, index=False, compression="gzip", float_format="%.6f")
    print(f"wrote {PROCESSED} ({PROCESSED.stat().st_size / 1e6:.1f} MB)")
