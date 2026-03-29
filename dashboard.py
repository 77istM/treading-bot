import os
import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st


DB_PATH = os.getenv("TRADING_DB_PATH", "trading_bot.db")
DAILY_MIN_TRADES = int(os.getenv("DAILY_MIN_TRADES", "20"))
DAILY_MAX_TRADES = int(os.getenv("DAILY_MAX_TRADES", "100"))

st.set_page_config(page_title="Trading Bot Dashboard", layout="wide")
st.title("Trading Bot Dashboard")
st.caption("Connected to SQLite database and rendering live trade telemetry.")


@st.cache_data(ttl=10)
def load_trades(db_path: str) -> pd.DataFrame:
    """Load trades from SQLite. Returns an empty DataFrame when missing/unavailable."""
    if not os.path.exists(db_path):
        return pd.DataFrame()

    try:
        with sqlite3.connect(db_path) as conn:
            trades = pd.read_sql_query("SELECT rowid AS trade_rowid, * FROM trades", conn)
        return trades
    except sqlite3.Error:
        return pd.DataFrame()


def build_pnl_frame(trades: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Build cumulative PnL series from available columns, else a signed-exposure proxy."""
    frame = trades.copy()

    # Prefer explicit PnL-like columns if present.
    pnl_candidates = ["pnl", "profit", "profit_loss", "realized_pnl", "realised_pnl"]
    pnl_col = next((col for col in pnl_candidates if col in frame.columns), None)

    if pnl_col is not None:
        frame[pnl_col] = pd.to_numeric(frame[pnl_col], errors="coerce").fillna(0.0)
        frame["cumulative_pnl"] = frame[pnl_col].cumsum()
        return frame, f"Cumulative PnL ({pnl_col})"

    if "side" in frame.columns:
        qty = pd.to_numeric(frame.get("qty", 1.0), errors="coerce").fillna(1.0)
        direction = frame["side"].astype(str).str.upper().map(
            {
                "BULLISH": 1,
                "BUY": 1,
                "LONG": 1,
                "BEARISH": -1,
                "SELL": -1,
                "SHORT": -1,
            }
        ).fillna(0)
        frame["cumulative_pnl"] = (direction * qty).cumsum()
        return frame, "Cumulative Exposure Proxy (side x qty)"

    frame["cumulative_pnl"] = 0.0
    return frame, "Cumulative PnL (insufficient trade fields)"


trades_df = load_trades(DB_PATH)

if trades_df.empty:
    st.warning(f"No trades found at '{DB_PATH}'. Run the bot first to populate the database.")
    st.stop()

if "trade_rowid" in trades_df.columns:
    trades_df = trades_df.sort_values("trade_rowid").reset_index(drop=True)
else:
    trades_df = trades_df.reset_index(drop=True)

pnl_df, pnl_label = build_pnl_frame(trades_df)

total_trades = len(trades_df)
buy_trades = 0
sell_trades = 0
daily_trades = 0
if "side" in trades_df.columns:
    side_normalized = trades_df["side"].astype(str).str.upper()
    buy_trades = side_normalized.isin(["BULLISH", "BUY", "LONG"]).sum()
    sell_trades = side_normalized.isin(["BEARISH", "SELL", "SHORT"]).sum()

if "created_at" in trades_df.columns:
    today_str = date.today().isoformat()
    daily_trades = trades_df[
        trades_df["created_at"].astype(str).str.startswith(today_str)
    ].shape[0]

col1, col2, col3, col4, col5 = st.columns(5)
col1.metric("Total Trades", total_trades)
col2.metric("Buy Trades", int(buy_trades))
col3.metric("Sell Trades", int(sell_trades))
col4.metric("Net", int(buy_trades - sell_trades))
col5.metric(
    f"Trades Today ({date.today().isoformat()})",
    f"{daily_trades} / {DAILY_MAX_TRADES}",
    delta=f"min target: {DAILY_MIN_TRADES}",
    delta_color="off",
)

st.subheader("PnL Curve")
st.line_chart(pnl_df.set_index("trade_rowid" if "trade_rowid" in pnl_df.columns else pnl_df.index)["cumulative_pnl"])
st.caption(pnl_label)

st.subheader("Trade History")
display_df = trades_df.sort_values("trade_rowid", ascending=False) if "trade_rowid" in trades_df.columns else trades_df.iloc[::-1]
st.dataframe(display_df, use_container_width=True)

if st.button("Refresh"):
    st.cache_data.clear()
    st.rerun()

st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
