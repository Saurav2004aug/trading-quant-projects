"""
Load NIFTY 50 and India VIX daily closes and align them.

Bundled files (see data/SOURCES.md): data/nifty50_daily.csv and
data/india_vix_daily.csv. To use official NSE downloads instead, pass the
paths of the historical-data CSVs from niftyindices.com / nseindia.com;
column names are matched case-insensitively and date formats such as
"02-Jan-2020" or "2020-01-02" are both accepted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"
TRADING_DAYS = 252


def _read_close(path: Path) -> pd.Series:
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    date_col = next(c for c in df.columns if "date" in c)
    close_col = next(c for c in df.columns if c.startswith("close"))
    dates = pd.to_datetime(df[date_col].astype(str).str.strip(), format="mixed", dayfirst=False)
    s = pd.Series(pd.to_numeric(df[close_col].astype(str).str.replace(",", ""), errors="coerce").values,
                  index=dates).dropna().sort_index()
    s = s[~s.index.duplicated(keep="last")]
    if (s <= 0).any():
        raise ValueError(f"non-positive closes in {path}")
    return s


def load(nifty_path: Path = DATA / "nifty50_daily.csv",
         vix_path: Path = DATA / "india_vix_daily.csv") -> pd.DataFrame:
    """Daily frame indexed by date: spot, vix (in %), log return.

    Returns every NIFTY date (needed for realised volatility); `vix` is NaN
    on the few days without a VIX print.
    """
    spot = _read_close(nifty_path)
    vix = _read_close(vix_path)
    df = pd.DataFrame({"spot": spot})
    df["vix"] = vix.reindex(df.index)
    df["ret"] = np.log(df["spot"]).diff()
    return df
