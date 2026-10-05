"""
Streamlit dashboard over the warehouse built by etl.py.

    streamlit run dashboard.py

The dashboard only reads the clean `trades` table, the `quarantine` table
and the `etl_runs` log -- it never touches the raw file. Data quality is
shown next to the numbers so a viewer can see how much was excluded and why.
"""
import json
import os
import sqlite3
from pathlib import Path

import pandas as pd
import streamlit as st

DB_PATH = Path(os.environ.get("WAREHOUSE_DB",
                              Path(__file__).resolve().parents[1] / "data" / "warehouse.db"))

st.set_page_config(page_title="Trading Activity", layout="wide")


@st.cache_data(ttl=300)
def load_tables(db_path: str = str(DB_PATH)):
    with sqlite3.connect(db_path) as conn:
        trades = pd.read_sql("SELECT * FROM trades", conn, parse_dates=["trade_date"])
        quarantine = pd.read_sql("SELECT reason, COUNT(*) AS rows FROM quarantine GROUP BY reason "
                                 "ORDER BY rows DESC", conn)
        runs = pd.read_sql("SELECT * FROM etl_runs ORDER BY started_at DESC", conn)
    return trades, quarantine, runs


if not DB_PATH.exists():
    st.error("No warehouse found. Run `python etl.py` first.")
    st.stop()

trades, quarantine, runs = load_tables()
last_run = runs.iloc[0]
report = json.loads(last_run["report"])

st.title("Trading activity")
st.caption(f"Warehouse built by run `{last_run['run_id']}` at {last_run['started_at']} UTC "
           f"from `{last_run['source_file']}` (sha256 {last_run['source_sha256'][:10]}...)")

# ---------------- filters
st.sidebar.header("Filters")
instruments = sorted(trades["instrument"].unique())
chosen = st.sidebar.multiselect("Instrument", instruments, default=instruments)
d0, d1 = trades["trade_date"].min().date(), trades["trade_date"].max().date()
dates = st.sidebar.date_input("Date range", (d0, d1), min_value=d0, max_value=d1)
exclude_outliers = st.sidebar.checkbox("Exclude flagged volume outliers", value=True)
exclude_imputed = st.sidebar.checkbox("Exclude trades with imputed price", value=False)

mask = trades["instrument"].isin(chosen)
if isinstance(dates, (tuple, list)) and len(dates) == 2:
    mask &= trades["trade_date"].between(pd.Timestamp(dates[0]), pd.Timestamp(dates[1]))
if exclude_outliers:
    mask &= trades["volume_outlier_flag"] == 0
if exclude_imputed:
    mask &= trades["price_imputed"] == 0
view = trades[mask]

# ---------------- KPIs
c1, c2, c3, c4 = st.columns(4)
c1.metric("Trades", f"{len(view):,}")
c2.metric("Total P&L", f"{view['pnl'].sum():,.0f}")
c3.metric("Avg P&L per trade", f"{view['pnl'].mean():,.1f}" if len(view) else "-")
c4.metric("Win rate", f"{(view['pnl'] > 0).mean():.1%}" if len(view) else "-")

tab_perf, tab_quality, tab_data = st.tabs(["Performance", "Data quality", "Data"])

with tab_perf:
    left, right = st.columns(2)
    with left:
        st.subheader("Cumulative P&L by instrument")
        cum = (view.groupby(["trade_date", "instrument"])["pnl"].sum()
               .unstack(fill_value=0).cumsum())
        st.line_chart(cum)
    with right:
        st.subheader("Notional volume by instrument")
        notional = (view["volume"] * view["avg_price"]).groupby(view["instrument"]).sum()
        st.bar_chart(notional.sort_values(ascending=False))

with tab_quality:
    q1, q2, q3 = st.columns(3)
    q1.metric("Rows received", f"{report['rows_in']:,}")
    q2.metric("Loaded", f"{report['rows_loaded']:,}")
    q3.metric("Quarantined + duplicates",
              f"{report['rows_quarantined'] + report['exact_duplicates_dropped']:,}")
    st.subheader("Why rows were rejected")
    st.bar_chart(quarantine.set_index("reason")["rows"])
    st.write(f"Prices imputed: {report['price_imputed']}  |  "
             f"volume outliers flagged (kept): {report['volume_outliers_flagged']}  |  "
             f"row accounting reconciles: {report['reconciles']}")
    st.subheader("Run history")
    st.dataframe(runs[["run_id", "started_at", "source_file"]], use_container_width=True)

with tab_data:
    st.dataframe(view.sort_values("trade_date", ascending=False), use_container_width=True)
    st.download_button("Download filtered trades (CSV)", view.to_csv(index=False).encode(),
                       file_name="trades_filtered.csv", mime="text/csv")
