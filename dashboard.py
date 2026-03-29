import logging
import os
import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st

from db.queries import read_setting as _db_read_setting, write_setting as _db_write_setting

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("TRADING_DB_PATH", "trading_bot.db")
DAILY_MAX_TRADES = int(os.getenv("DAILY_MAX_TRADES", "1000"))
DEFAULT_STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))
DEFAULT_TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))

st.set_page_config(page_title="Hedge Fund Trading Bot", layout="wide")
st.title("🏦 Hedge Fund Trading Bot Dashboard")
st.caption("Real-time trade telemetry with multi-signal AI analysis (sentiment · technicals · geopolitics · Fed rate · fear/VIX).")


# --- Settings helpers (read/write via DB settings table) ---
def _get_conn():
    if not os.path.exists(DB_PATH):
        return None
    try:
        return sqlite3.connect(DB_PATH)
    except sqlite3.Error as exc:
        logger.warning("Failed to open database at '%s': %s", DB_PATH, exc)
        return None


def write_setting(key: str, value: float) -> None:
    conn = _get_conn()
    if conn is None:
        return
    try:
        _db_write_setting(conn, key, value)
    finally:
        conn.close()


def read_setting(key: str, default: float) -> float:
    conn = _get_conn()
    if conn is None:
        return default
    try:
        return _db_read_setting(conn, key, default)
    finally:
        conn.close()


# --- Sidebar: Risk Configuration ---
with st.sidebar:
    st.header("⚙️ Risk Configuration")
    st.caption("Override stop loss and take profit levels. Changes take effect on the next bot run.")

    current_stop = read_setting("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)
    current_take = read_setting("take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)

    stop_pct = st.slider(
        "Stop Loss %",
        min_value=0.5,
        max_value=10.0,
        value=float(current_stop * 100),
        step=0.5,
        format="%.1f%%",
        help="Close position when loss exceeds this % (default 3 %).",
    ) / 100.0

    take_pct = st.slider(
        "Take Profit %",
        min_value=1.0,
        max_value=20.0,
        value=float(current_take * 100),
        step=0.5,
        format="%.1f%%",
        help="Close position when profit reaches this % (default 5 %).",
    ) / 100.0

    if st.button("💾 Save Risk Settings"):
        write_setting("stop_loss_pct", stop_pct)
        write_setting("take_profit_pct", take_pct)
        st.success(f"Saved: Stop = {stop_pct * 100:.1f}%  |  Take = {take_pct * 100:.1f}%")

    st.divider()
    st.markdown(
        f"**Ring Fence:** < 3 % per trade  \n"
        f"**Daily Max:** {DAILY_MAX_TRADES} trades  \n"
        f"**Directions:** Long (BUY) & Short (SELL)"
    )


# --- Data Loading ---
@st.cache_data(ttl=10)
def load_trades(db_path: str) -> pd.DataFrame:
    """Load trades from SQLite. Returns an empty DataFrame when missing/unavailable."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            trades = pd.read_sql_query("SELECT rowid AS trade_rowid, * FROM trades", conn)
        return trades
    except sqlite3.Error as exc:
        logger.warning("Failed to load trades from '%s': %s", db_path, exc)
        return pd.DataFrame()


def build_pnl_frame(trades: pd.DataFrame) -> tuple[pd.DataFrame, str]:
    """Build cumulative PnL series from available columns, else a signed-exposure proxy."""
    frame = trades.copy()
    pnl_candidates = ["pnl", "profit", "profit_loss", "realized_pnl", "realised_pnl"]
    pnl_col = next((col for col in pnl_candidates if col in frame.columns), None)

    if pnl_col is not None:
        frame[pnl_col] = pd.to_numeric(frame[pnl_col], errors="coerce").fillna(0.0)
        frame["cumulative_pnl"] = frame[pnl_col].cumsum()
        return frame, f"Cumulative PnL ({pnl_col})"

    if "side" in frame.columns:
        qty = pd.to_numeric(frame.get("qty", 1.0), errors="coerce").fillna(1.0)
        direction = frame["side"].astype(str).str.upper().map(
            {"BULLISH": 1, "BUY": 1, "LONG": 1, "BEARISH": -1, "SELL": -1, "SHORT": -1}
        ).fillna(0)
        frame["cumulative_pnl"] = (direction * qty).cumsum()
        return frame, "Cumulative Exposure Proxy (side × qty)"

    frame["cumulative_pnl"] = 0.0
    return frame, "Cumulative PnL (insufficient trade fields)"


# --- Load Data ---
trades_df = load_trades(DB_PATH)

if trades_df.empty:
    st.warning(f"No trades found at '{DB_PATH}'. Run the bot first to populate the database.")
    st.stop()

if "trade_rowid" in trades_df.columns:
    trades_df = trades_df.sort_values("trade_rowid").reset_index(drop=True)
else:
    trades_df = trades_df.reset_index(drop=True)

pnl_df, pnl_label = build_pnl_frame(trades_df)

# --- Metrics ---
total_trades = len(trades_df)
long_trades = 0
short_trades = 0
daily_trades = 0
net_pnl = 0.0

if "side" in trades_df.columns:
    side_norm = trades_df["side"].astype(str).str.upper()
    long_trades = int(side_norm.isin(["BULLISH", "BUY", "LONG"]).sum())
    short_trades = int(side_norm.isin(["BEARISH", "SELL", "SHORT"]).sum())

if "created_at" in trades_df.columns:
    today_str = date.today().isoformat()
    daily_trades = int(trades_df[
        trades_df["created_at"].astype(str).str.startswith(today_str)
    ].shape[0])

if "realized_pnl" in trades_df.columns:
    net_pnl = float(pd.to_numeric(trades_df["realized_pnl"], errors="coerce").fillna(0).sum())

col1, col2, col3, col4, col5, col6 = st.columns(6)
col1.metric("Total Trades", total_trades)
col2.metric("Long (Buy)", long_trades)
col3.metric("Short (Sell)", short_trades)
col4.metric("Net Position", long_trades - short_trades)
col5.metric(f"Today ({date.today().isoformat()})", f"{daily_trades} / {DAILY_MAX_TRADES}")
col6.metric("Realized PnL", f"${net_pnl:,.2f}")

# --- PnL Curve ---
st.subheader("📈 PnL Curve")
idx_col = "trade_rowid" if "trade_rowid" in pnl_df.columns else pnl_df.index
st.line_chart(pnl_df.set_index(idx_col)["cumulative_pnl"])
st.caption(pnl_label)

# --- Trade History ---
st.subheader("📋 Trade History")
desired_cols = [
    "trade_rowid", "created_at", "ticker", "side", "qty", "price",
    "stop_loss_price", "take_profit_price",
    "sentiment", "technical_signal", "geopolitics", "fed_sentiment", "fear_level",
    "realized_pnl", "reason",
]
display_cols = [c for c in desired_cols if c in trades_df.columns]
display_df = trades_df[display_cols].copy() if display_cols else trades_df.copy()

for col in ("price", "stop_loss_price", "take_profit_price", "realized_pnl"):
    if col in display_df.columns:
        display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(4)

if "trade_rowid" in display_df.columns:
    display_df = display_df.sort_values("trade_rowid", ascending=False)

st.dataframe(display_df, use_container_width=True)

# --- AI Pre-Trade Analysis Viewer ---
if "trade_analysis" in trades_df.columns:
    st.subheader("🧠 Pre-Trade AI Analysis (last 5 trades)")
    recent = (
        trades_df[
            trades_df["trade_analysis"].notna()
            & (trades_df["trade_analysis"].astype(str).str.strip() != "")
        ]
        .tail(5)
    )
    if recent.empty:
        st.info("No AI analysis recorded yet.")
    else:
        for _, row in recent.iterrows():
            label = (
                f"[{row.get('ticker', 'N/A')}]  "
                f"{row.get('created_at', '')}  —  {row.get('side', '')}"
            )
            with st.expander(label):
                st.text(row.get("trade_analysis", ""))

if st.button("🔄 Refresh"):
    st.cache_data.clear()
    st.rerun()

st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

