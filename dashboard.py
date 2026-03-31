import logging
import os
import sqlite3
from datetime import date, datetime

import pandas as pd
import streamlit as st

from db.queries import (
    read_bool_setting as _db_read_bool_setting,
    read_setting as _db_read_setting,
    write_setting as _db_write_setting,
)
from pnl.attribution import (
    benchmark_cumulative_returns,
    build_closed_trades_frame,
    compute_core_metrics,
    compute_signal_accuracy,
    compute_signal_outcome_breakdown,
    compute_signal_pnl_breakdown,
    compute_strategy_pnl_breakdown,
)

logger = logging.getLogger(__name__)

DB_PATH = os.getenv("TRADING_DB_PATH", "trading_bot.db")
DAILY_MAX_TRADES = int(os.getenv("DAILY_MAX_TRADES", "1000"))
DEFAULT_STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))
DEFAULT_TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))
DEFAULT_ALLOW_STOCK_SHORTS = os.getenv("ALLOW_STOCK_SHORTS", "true").strip().lower() in {
    "1", "true", "yes", "y", "on"
}
DEFAULT_ALLOW_CRYPTO_SHORTS = os.getenv("ALLOW_CRYPTO_SHORTS", "false").strip().lower() in {
    "1", "true", "yes", "y", "on"
}
CRYPTO_TICKERS = os.getenv("CRYPTO_TICKERS", "BTC/USD,ETH/USD")
BOT_LOG_PATH = os.getenv("BOT_LOG_PATH", "bot.log")

st.set_page_config(page_title="Hedge Fund Trading Bot", layout="wide")
st.title("🏦 Hedge Fund Trading Bot Dashboard")
st.caption(
    "Real-time trade telemetry · multi-signal AI analysis · "
    "post-trade reflection · portfolio risk controller"
)


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


def read_bool_setting(key: str, default: bool) -> bool:
    conn = _get_conn()
    if conn is None:
        return default
    try:
        return _db_read_bool_setting(conn, key, default)
    finally:
        conn.close()


# --- Sidebar: Risk Configuration ---
with st.sidebar:
    st.header("⚙️ Risk Configuration")
    st.caption("Override stop loss and take profit levels. Changes take effect on the next bot run.")

    current_stop = read_setting("stop_loss_pct", DEFAULT_STOP_LOSS_PCT)
    current_take = read_setting("take_profit_pct", DEFAULT_TAKE_PROFIT_PCT)
    current_stock_shorts = read_bool_setting("allow_stock_shorts", DEFAULT_ALLOW_STOCK_SHORTS)
    current_crypto_shorts = read_bool_setting("allow_crypto_shorts", DEFAULT_ALLOW_CRYPTO_SHORTS)

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

    allow_stock_shorts = st.checkbox(
        "Allow Stock Shorts",
        value=current_stock_shorts,
        help="Enable SELL orders for non-crypto symbols.",
    )

    allow_crypto_shorts = st.checkbox(
        "Allow Crypto Shorts",
        value=current_crypto_shorts,
        help="Enable SELL orders for crypto symbols.",
    )

    if st.button("💾 Save Risk Settings"):
        write_setting("stop_loss_pct", stop_pct)
        write_setting("take_profit_pct", take_pct)
        write_setting("allow_stock_shorts", allow_stock_shorts)
        write_setting("allow_crypto_shorts", allow_crypto_shorts)
        st.success(
            "Saved risk controls and direction toggles. "
            f"Stop = {stop_pct * 100:.1f}% | Take = {take_pct * 100:.1f}%"
        )

    # Keep caption in sync even before pressing save.
    stock_direction = "Long + Short" if allow_stock_shorts else "Long-only (BUY)"
    crypto_direction = "Long + Short" if allow_crypto_shorts else "Long-only (BUY)"

    st.divider()
    st.markdown(
        f"**Ring Fence:** < 3 % per trade  \n"
        f"**Daily Max:** {DAILY_MAX_TRADES} trades  \n"
        f"**Crypto Universe:** {CRYPTO_TICKERS}  \n"
        f"**Stock Direction:** {stock_direction}  \n"
        f"**Crypto Direction:** {crypto_direction}"
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


@st.cache_data(ttl=10)
def load_reflections(db_path: str) -> pd.DataFrame:
    """Load the reflections table. Returns an empty DataFrame when unavailable."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM reflections ORDER BY id DESC", conn
            )
    except sqlite3.Error:
        return pd.DataFrame()


@st.cache_data(ttl=10)
def load_risk_snapshots(db_path: str) -> pd.DataFrame:
    """Load the risk_snapshots table. Returns an empty DataFrame when unavailable."""
    if not os.path.exists(db_path):
        return pd.DataFrame()
    try:
        with sqlite3.connect(db_path) as conn:
            return pd.read_sql_query(
                "SELECT * FROM risk_snapshots ORDER BY id DESC LIMIT 200", conn
            )
    except sqlite3.Error:
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


def _tail_file(path: str, lines: int = 40) -> list[str]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="ignore") as f:
            all_lines = f.readlines()
        return [ln.rstrip("\n") for ln in all_lines[-lines:]]
    except OSError:
        return []


def _bot_activity_hint(db_path: str) -> tuple[bool, str]:
    if not os.path.exists(db_path):
        return False, f"Database file '{db_path}' does not exist yet."
    try:
        with sqlite3.connect(db_path) as conn:
            row = conn.execute("SELECT COUNT(*) FROM risk_snapshots").fetchone()
            snapshots = int(row[0]) if row else 0
        if snapshots > 0:
            return True, f"Bot appears active: {snapshots} risk snapshots recorded."
        return False, "Database exists, but no risk snapshots yet."
    except sqlite3.Error as exc:
        return False, f"Could not inspect DB activity: {exc}"


# --- Load Data ---
trades_df = load_trades(DB_PATH)

if trades_df.empty:
    st.warning(f"No trades found at '{DB_PATH}'.")
    active, detail = _bot_activity_hint(DB_PATH)
    if active:
        stock_shorts_enabled = read_bool_setting("allow_stock_shorts", DEFAULT_ALLOW_STOCK_SHORTS)
        crypto_shorts_enabled = read_bool_setting("allow_crypto_shorts", DEFAULT_ALLOW_CRYPTO_SHORTS)
        stock_mode = "Long + Short" if stock_shorts_enabled else "Long-only"
        crypto_mode = "Long + Short" if crypto_shorts_enabled else "Long-only"
        st.info(
            "Bot is running, but no orders have been executed yet. "
            f"Current direction filters: Stocks {stock_mode}, Crypto {crypto_mode}."
        )
    st.caption(detail)

    tail = _tail_file(BOT_LOG_PATH, lines=100)
    if tail:
        interesting = [
            ln for ln in tail
            if ("Strategy Decision" in ln or "Skipping" in ln or "Cycle complete" in ln)
        ]
        if interesting:
            st.subheader("Recent Bot Decisions")
            st.code("\n".join(interesting[-25:]), language="text")
    st.stop()

if "trade_rowid" in trades_df.columns:
    trades_df = trades_df.sort_values("trade_rowid").reset_index(drop=True)
else:
    trades_df = trades_df.reset_index(drop=True)

pnl_df, pnl_label = build_pnl_frame(trades_df)
closed_trades_df = build_closed_trades_frame(trades_df)
core_metrics = compute_core_metrics(closed_trades_df)

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

st.subheader("🎯 Phase 6 Performance Attribution")
p1, p2, p3, p4 = st.columns(4)
p1.metric("Closed Trades", int(core_metrics["closed_trades"]))
p2.metric("Win Rate", f"{core_metrics['win_rate'] * 100:.1f}%")
p3.metric("Sharpe (trade-level)", f"{core_metrics['sharpe']:.2f}")
p4.metric("Max Drawdown", f"{core_metrics['max_drawdown_pct'] * 100:.2f}%")
st.caption(f"Max drawdown (absolute): ${core_metrics['max_drawdown']:,.2f}")

# --- PnL Curve ---
st.subheader("📈 PnL Curve")
idx_col = "trade_rowid" if "trade_rowid" in pnl_df.columns else pnl_df.index
st.line_chart(pnl_df.set_index(idx_col)["cumulative_pnl"])
st.caption(pnl_label)

# --- Benchmark Comparison ---
st.subheader("🌍 Benchmark Comparison")
if "created_at" in closed_trades_df.columns and not closed_trades_df.empty:
    local_closed = closed_trades_df.copy()
    local_closed["created_at"] = pd.to_datetime(local_closed["created_at"], errors="coerce")
    local_closed = local_closed.dropna(subset=["created_at"])

    qty = pd.to_numeric(local_closed.get("qty"), errors="coerce").fillna(0.0).abs()
    entry_px = pd.to_numeric(local_closed.get("entry_reference_price"), errors="coerce")
    fallback_px = pd.to_numeric(local_closed.get("price"), errors="coerce")
    notionals = (qty * entry_px.fillna(fallback_px)).replace(0, pd.NA)
    local_closed["trade_return"] = (
        pd.to_numeric(local_closed.get("realized_pnl"), errors="coerce").fillna(0.0) / notionals
    ).fillna(0.0)
    local_closed["trade_day"] = local_closed["created_at"].dt.date
    bot_daily_returns = local_closed.groupby("trade_day")["trade_return"].mean().sort_index()

    if not bot_daily_returns.empty:
        bot_cum = (1.0 + bot_daily_returns).cumprod() - 1.0
        start_ts = pd.Timestamp(bot_daily_returns.index.min()).to_pydatetime()
        end_ts = pd.Timestamp(bot_daily_returns.index.max()).to_pydatetime()
        bench = benchmark_cumulative_returns(start_ts, end_ts)

        comparison = pd.DataFrame({"BOT": bot_cum.values}, index=pd.to_datetime(bot_cum.index))
        if not bench.empty:
            comparison = comparison.join(bench, how="outer").sort_index().ffill().fillna(0.0)

        st.line_chart(comparison)

        if not bench.empty:
            bot_final = float(comparison["BOT"].iloc[-1])
            alpha_rows = []
            for col in bench.columns:
                bm_final = float(comparison[col].iloc[-1])
                alpha_rows.append(
                    {
                        "benchmark": col,
                        "benchmark_return": bm_final,
                        "bot_return": bot_final,
                        "alpha": bot_final - bm_final,
                    }
                )
            alpha_df = pd.DataFrame(alpha_rows)
            alpha_df["benchmark_return"] = (alpha_df["benchmark_return"] * 100).round(2)
            alpha_df["bot_return"] = (alpha_df["bot_return"] * 100).round(2)
            alpha_df["alpha"] = (alpha_df["alpha"] * 100).round(2)
            st.dataframe(alpha_df, use_container_width=True)
            st.caption("Alpha vs Benchmark (%)")
            st.bar_chart(alpha_df.set_index("benchmark")[["alpha"]])
        else:
            st.info("Benchmark data unavailable. Ensure Alpaca data credentials are configured.")
    else:
        st.info("Not enough closed-trade return data yet for benchmark comparison.")
else:
    st.info("No closed trades with timestamps available yet for benchmark comparison.")

# --- Signal Accuracy Scoreboard ---
st.subheader("✅ Signal Accuracy Scoreboard")
accuracy_df = compute_signal_accuracy(closed_trades_df)
if accuracy_df.empty:
    st.info("No directional signal outcomes yet. Accuracy appears after closing trades with price-move data.")
else:
    show_acc = accuracy_df.copy()
    show_acc["accuracy"] = (show_acc["accuracy"] * 100).round(2)
    st.dataframe(show_acc, use_container_width=True)
    accuracy_chart = (
        show_acc[["signal", "accuracy"]]
        .set_index("signal")
        .sort_values("accuracy", ascending=False)
    )
    st.bar_chart(accuracy_chart)

# --- Signal Win/Loss Attribution ---
st.subheader("⚖️ Signal Win/Loss Attribution")
signal_outcomes_df = compute_signal_outcome_breakdown(closed_trades_df)
if signal_outcomes_df.empty:
    st.info("No closed-trade outcomes to attribute by signal state yet.")
else:
    signal_outcome_filter = st.selectbox(
        "Attribution Signal Family",
        options=["ALL"] + sorted(signal_outcomes_df["signal"].dropna().unique().tolist()),
        index=0,
    )
    show_outcomes = signal_outcomes_df.copy()
    if signal_outcome_filter != "ALL":
        show_outcomes = show_outcomes[show_outcomes["signal"] == signal_outcome_filter]
    show_outcomes["win_rate"] = (pd.to_numeric(show_outcomes["win_rate"], errors="coerce") * 100).round(2)
    for col in ("total_pnl", "avg_pnl"):
        show_outcomes[col] = pd.to_numeric(show_outcomes[col], errors="coerce").round(2)
    st.dataframe(show_outcomes, use_container_width=True)

    if not show_outcomes.empty:
        outcome_charts = show_outcomes[["signal_state", "win_rate", "total_pnl"]].copy()
        outcome_charts["signal_state"] = outcome_charts["signal_state"].astype(str)
        oc1, oc2 = st.columns(2)
        with oc1:
            st.caption("Win Rate by Signal State (%)")
            st.bar_chart(outcome_charts.set_index("signal_state")[["win_rate"]])
        with oc2:
            st.caption("Total PnL by Signal State ($)")
            st.bar_chart(outcome_charts.set_index("signal_state")[["total_pnl"]])

# --- Per-Strategy PnL ---
st.subheader("🧠 Per-Strategy PnL")
strategy_pnl_df = compute_strategy_pnl_breakdown(closed_trades_df)
if strategy_pnl_df.empty:
    st.info("No strategy-attributed closed trades yet.")
else:
    show_strategy = strategy_pnl_df.copy()
    show_strategy["total_pnl"] = pd.to_numeric(show_strategy["total_pnl"], errors="coerce").round(2)
    show_strategy["avg_pnl"] = pd.to_numeric(show_strategy["avg_pnl"], errors="coerce").round(2)
    st.dataframe(show_strategy, use_container_width=True)

    strategy_series = (
        show_strategy.groupby("strategy_name", dropna=False)["total_pnl"]
        .sum()
        .sort_values(ascending=False)
    )
    st.bar_chart(strategy_series)

    strategy_regime = (
        show_strategy.pivot_table(
            index="strategy_name",
            columns="strategy_regime",
            values="total_pnl",
            aggfunc="sum",
            fill_value=0.0,
        )
        .sort_index()
    )
    if not strategy_regime.empty:
        st.caption("Strategy PnL by Regime")
        st.bar_chart(strategy_regime)

# --- Per-Signal PnL Contribution ---
st.subheader("🧩 Per-Signal PnL Contribution")
signal_pnl_df = compute_signal_pnl_breakdown(closed_trades_df)
if signal_pnl_df.empty:
    st.info("No attributable closed trades yet.")
else:
    signal_filter = st.selectbox(
        "Signal Family",
        options=["ALL"] + sorted(signal_pnl_df["signal"].dropna().unique().tolist()),
        index=0,
    )
    table_df = signal_pnl_df.copy()
    if signal_filter != "ALL":
        table_df = table_df[table_df["signal"] == signal_filter]

    for col in ("total_pnl", "avg_pnl"):
        table_df[col] = pd.to_numeric(table_df[col], errors="coerce").round(2)

    st.dataframe(table_df, use_container_width=True)

    if not table_df.empty:
        contribution_chart = (
            table_df[["signal_state", "total_pnl"]]
            .copy()
            .set_index("signal_state")
            .sort_values("total_pnl", ascending=False)
        )
        st.caption("Signal-State PnL Contribution ($)")
        st.bar_chart(contribution_chart)

# --- Trade History ---
st.subheader("📋 Trade History")
desired_cols = [
    "trade_rowid", "created_at", "ticker", "side", "qty", "price",
    "stop_loss_price", "take_profit_price",
    "strategy_name", "strategy_regime",
    # Phase 1 signals
    "sentiment", "geopolitics", "fed_sentiment", "fear_level",
    # Phase 2 signals
    "technical_signal", "rsi_signal", "macd_signal", "bbands_signal", "volume_signal",
    "earnings_flag", "momentum_score",
    # Phase 6 attribution fields
    "is_closing_trade", "entry_reference_price", "price_move_pct",
    "realized_pnl", "reason",
]
display_cols = [c for c in desired_cols if c in trades_df.columns]
display_df = trades_df[display_cols].copy() if display_cols else trades_df.copy()

for col in (
    "price", "stop_loss_price", "take_profit_price", "entry_reference_price",
    "price_move_pct", "realized_pnl", "momentum_score",
):
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

# --- Post-Trade Reflections ---
st.divider()
st.subheader("🧠 Post-Trade Reflections & Lessons")
reflections_df = load_reflections(DB_PATH)
if reflections_df.empty:
    st.info("No reflections recorded yet. They appear after stop-loss events and each market close.")
else:
    tab_stop, tab_eod, tab_all = st.tabs(["🔴 Stop-Loss Reflections", "🌙 End-of-Day Reviews", "📚 All Lessons"])

    with tab_stop:
        sl_df = reflections_df[reflections_df["reflection_type"] == "stop_loss"].head(20)
        if sl_df.empty:
            st.info("No stop-loss reflections yet.")
        else:
            for _, row in sl_df.iterrows():
                label = (
                    f"[{row.get('ticker', 'N/A')}]  {row.get('created_at', '')}  "
                    f"PnL: ${float(row.get('pnl') or 0):.2f}"
                )
                with st.expander(label):
                    st.markdown(f"**Lesson:** {row.get('lesson', '')}")
                    st.text(row.get("raw_analysis", ""))

    with tab_eod:
        eod_df = reflections_df[reflections_df["reflection_type"] == "end_of_day"].head(10)
        if eod_df.empty:
            st.info("No end-of-day reflections yet. They run automatically after market close.")
        else:
            for _, row in eod_df.iterrows():
                label = f"EOD Review — {row.get('created_at', '')}  PnL: ${float(row.get('pnl') or 0):.2f}"
                with st.expander(label):
                    st.markdown(f"**Summary:** {row.get('lesson', '')}")
                    st.text(row.get("raw_analysis", ""))

    with tab_all:
        display_refl = reflections_df[["created_at", "reflection_type", "ticker",
                                       "outcome", "pnl", "lesson"]].copy()
        display_refl["pnl"] = pd.to_numeric(display_refl["pnl"], errors="coerce").round(2)
        st.dataframe(display_refl, use_container_width=True)

# --- Portfolio Risk Status ---
st.divider()
st.subheader("🛡️ Portfolio Risk Controller")
risk_df = load_risk_snapshots(DB_PATH)
if risk_df.empty:
    st.info("No risk snapshots yet. They are recorded each trading cycle.")
else:
    latest = risk_df.iloc[0]
    rc1, rc2, rc3, rc4 = st.columns(4)
    rc1.metric("Portfolio Value", f"${float(latest.get('portfolio_value') or 0):,.0f}")
    rc2.metric("Day-Start Value", f"${float(latest.get('day_start_value') or 0):,.0f}")
    drawdown = float(latest.get('drawdown_pct') or 0)
    rc3.metric("Daily Drawdown", f"{drawdown * 100:.2f}%", delta_color="inverse")
    heat = float(latest.get('total_heat_pct') or 0)
    rc4.metric("Portfolio Heat", f"{heat * 100:.1f}%")

    halted = int(latest.get("trading_halted") or 0)
    if halted:
        st.error(f"⛔ Trading HALTED — {latest.get('halt_reason', '')}")
    else:
        st.success("✅ Trading ACTIVE — all risk limits within bounds.")

    with st.expander("Risk Snapshot History"):
        display_risk = risk_df[["created_at", "portfolio_value", "drawdown_pct",
                                 "total_heat_pct", "open_positions",
                                 "trading_halted", "halt_reason"]].copy()
        for col in ("portfolio_value", "drawdown_pct", "total_heat_pct"):
            display_risk[col] = pd.to_numeric(display_risk[col], errors="coerce").round(4)
        st.dataframe(display_risk, use_container_width=True)

st.caption("Last updated: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

