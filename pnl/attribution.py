import logging
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from config import BENCHMARK_TICKERS, StockBarsRequest, TimeFrame, data_client

logger = logging.getLogger(__name__)

_SIGNAL_COLUMNS = [
    "sentiment",
    "technical_signal",
    "rsi_signal",
    "macd_signal",
    "bbands_signal",
    "volume_signal",
    "earnings_flag",
    "geopolitics",
    "fed_sentiment",
    "fear_level",
]

_DIRECTIONAL_SIGNAL_SPECS = {
    "sentiment": {"BULLISH": 1, "BEARISH": -1},
    "technical_signal": {"BULLISH": 1, "BEARISH": -1},
    "rsi_signal": {"BULLISH": 1, "BEARISH": -1},
    "macd_signal": {"BULLISH": 1, "BEARISH": -1},
    "bbands_signal": {"BULLISH": 1, "BEARISH": -1},
    "volume_signal": {"SPIKE_UP": 1, "SPIKE_DOWN": -1},
}


def build_closed_trades_frame(trades: pd.DataFrame) -> pd.DataFrame:
    """Return trades representing realized outcomes (closed positions)."""
    if trades.empty:
        return pd.DataFrame()

    frame = trades.copy()
    frame["realized_pnl"] = pd.to_numeric(frame.get("realized_pnl"), errors="coerce").fillna(0.0)

    if "is_closing_trade" in frame.columns:
        close_mask = pd.to_numeric(frame["is_closing_trade"], errors="coerce").fillna(0).astype(int) == 1
        closed = frame[close_mask].copy()
        if not closed.empty:
            return closed

    closed = frame[frame["realized_pnl"] != 0].copy()
    return closed


def _trade_return_series(closed_trades: pd.DataFrame) -> pd.Series:
    qty = pd.to_numeric(closed_trades.get("qty"), errors="coerce").fillna(0.0).abs()
    px = pd.to_numeric(closed_trades.get("entry_reference_price"), errors="coerce")
    if px.isna().all():
        px = pd.to_numeric(closed_trades.get("price"), errors="coerce")

    notionals = (qty * px).replace(0, np.nan)
    pnl = pd.to_numeric(closed_trades.get("realized_pnl"), errors="coerce").fillna(0.0)
    returns = (pnl / notionals).replace([np.inf, -np.inf], np.nan).dropna()
    return returns


def compute_core_metrics(closed_trades: pd.DataFrame) -> dict:
    """Compute win rate, Sharpe ratio, and max drawdown from closed trades."""
    if closed_trades.empty:
        return {
            "closed_trades": 0,
            "wins": 0,
            "losses": 0,
            "win_rate": 0.0,
            "sharpe": 0.0,
            "max_drawdown": 0.0,
            "max_drawdown_pct": 0.0,
            "net_realized_pnl": 0.0,
        }

    pnl = pd.to_numeric(closed_trades.get("realized_pnl"), errors="coerce").fillna(0.0)
    wins = int((pnl > 0).sum())
    losses = int((pnl < 0).sum())
    total = int(len(closed_trades))

    returns = _trade_return_series(closed_trades)
    if returns.empty or float(returns.std(ddof=1) or 0.0) == 0.0:
        sharpe = 0.0
    else:
        sharpe = float((returns.mean() / returns.std(ddof=1)) * np.sqrt(252))

    equity_curve = pnl.cumsum()
    rolling_peak = equity_curve.cummax()
    drawdowns = equity_curve - rolling_peak
    max_drawdown = float(drawdowns.min()) if not drawdowns.empty else 0.0

    cumulative_returns = (1.0 + returns).cumprod() if not returns.empty else pd.Series(dtype=float)
    if cumulative_returns.empty:
        max_drawdown_pct = 0.0
    else:
        peak = cumulative_returns.cummax()
        dd_pct = (cumulative_returns / peak) - 1.0
        max_drawdown_pct = float(dd_pct.min()) if not dd_pct.empty else 0.0

    return {
        "closed_trades": total,
        "wins": wins,
        "losses": losses,
        "win_rate": (wins / total) if total else 0.0,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "max_drawdown_pct": max_drawdown_pct,
        "net_realized_pnl": float(pnl.sum()),
    }


def compute_signal_accuracy(closed_trades: pd.DataFrame) -> pd.DataFrame:
    """Compute directional accuracy for all directional categorical signals."""
    if closed_trades.empty:
        return pd.DataFrame()

    frame = closed_trades.copy()
    if "price_move_pct" not in frame.columns:
        return pd.DataFrame()

    move = pd.to_numeric(frame["price_move_pct"], errors="coerce")
    actual = np.sign(move)

    rows = []
    for col, direction_map in _DIRECTIONAL_SIGNAL_SPECS.items():
        if col not in frame.columns:
            continue
        signal = frame[col].astype(str).str.upper()
        expected = signal.map(direction_map)
        mask = expected.notna() & actual.notna() & (actual != 0)
        if not mask.any():
            continue

        correct = (expected[mask] == actual[mask]).sum()
        total = int(mask.sum())
        rows.append(
            {
                "signal": col,
                "samples": total,
                "correct": int(correct),
                "accuracy": float(correct / total) if total else 0.0,
            }
        )

    return pd.DataFrame(rows).sort_values("accuracy", ascending=False).reset_index(drop=True)


def compute_signal_outcome_breakdown(closed_trades: pd.DataFrame) -> pd.DataFrame:
    """Break down win/loss attribution by signal family and signal state."""
    if closed_trades.empty:
        return pd.DataFrame()

    frame = closed_trades.copy()
    frame["realized_pnl"] = pd.to_numeric(frame.get("realized_pnl"), errors="coerce").fillna(0.0)

    parts = []
    for signal_col in _SIGNAL_COLUMNS:
        if signal_col not in frame.columns:
            continue

        local = frame[[signal_col, "realized_pnl"]].copy()
        local["signal_state"] = local[signal_col].astype(str).str.upper().replace("", "UNKNOWN")
        local["signal"] = signal_col
        local["wins"] = (local["realized_pnl"] > 0).astype(int)
        local["losses"] = (local["realized_pnl"] < 0).astype(int)
        local["flat"] = (local["realized_pnl"] == 0).astype(int)

        grouped = (
            local.groupby(["signal", "signal_state"], dropna=False)
            .agg(
                trades=("realized_pnl", "count"),
                wins=("wins", "sum"),
                losses=("losses", "sum"),
                flat=("flat", "sum"),
                total_pnl=("realized_pnl", "sum"),
                avg_pnl=("realized_pnl", "mean"),
            )
            .reset_index()
        )
        grouped["win_rate"] = grouped["wins"] / grouped["trades"].replace(0, np.nan)
        grouped["win_rate"] = grouped["win_rate"].fillna(0.0)
        parts.append(grouped)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out = out.sort_values(["signal", "total_pnl"], ascending=[True, False]).reset_index(drop=True)
    return out


def compute_signal_pnl_breakdown(closed_trades: pd.DataFrame) -> pd.DataFrame:
    """Return per-signal-state realized PnL contribution breakdown."""
    if closed_trades.empty:
        return pd.DataFrame()

    parts = []
    frame = closed_trades.copy()
    frame["realized_pnl"] = pd.to_numeric(frame.get("realized_pnl"), errors="coerce").fillna(0.0)

    for signal_col in _SIGNAL_COLUMNS:
        if signal_col not in frame.columns:
            continue
        grouped = (
            frame.groupby(frame[signal_col].astype(str), dropna=False)["realized_pnl"]
            .agg(["count", "sum", "mean"])
            .reset_index()
            .rename(
                columns={
                    signal_col: "signal_state",
                    "count": "trades",
                    "sum": "total_pnl",
                    "mean": "avg_pnl",
                }
            )
        )
        grouped["signal"] = signal_col
        parts.append(grouped)

    if not parts:
        return pd.DataFrame()

    out = pd.concat(parts, ignore_index=True)
    out["abs_contribution"] = out["total_pnl"].abs()
    out = out.sort_values(["signal", "abs_contribution"], ascending=[True, False])
    return out.drop(columns=["abs_contribution"]).reset_index(drop=True)


def compute_strategy_pnl_breakdown(closed_trades: pd.DataFrame) -> pd.DataFrame:
    """Aggregate realized PnL by strategy and market regime."""
    if closed_trades.empty:
        return pd.DataFrame()
    if "strategy_name" not in closed_trades.columns:
        return pd.DataFrame()

    frame = closed_trades.copy()
    frame["strategy_name"] = frame["strategy_name"].astype(str).fillna("unknown")
    frame["strategy_regime"] = frame.get("strategy_regime", "UNKNOWN").astype(str)
    frame["realized_pnl"] = pd.to_numeric(frame.get("realized_pnl"), errors="coerce").fillna(0.0)

    grouped = (
        frame.groupby(["strategy_name", "strategy_regime"], dropna=False)["realized_pnl"]
        .agg(["count", "sum", "mean"])
        .reset_index()
        .rename(columns={"count": "trades", "sum": "total_pnl", "mean": "avg_pnl"})
        .sort_values("total_pnl", ascending=False)
        .reset_index(drop=True)
    )
    return grouped


def benchmark_cumulative_returns(
    start_at: datetime | None = None,
    end_at: datetime | None = None,
    tickers: list[str] | None = None,
) -> pd.DataFrame:
    """Fetch benchmark close series from Alpaca and return cumulative returns."""
    if data_client is None or StockBarsRequest is None or TimeFrame is None:
        return pd.DataFrame()

    benchmarks = tickers or BENCHMARK_TICKERS
    if not benchmarks:
        return pd.DataFrame()

    end_time = end_at or datetime.now(timezone.utc)
    start_time = start_at or (end_time - timedelta(days=365))

    try:
        req = StockBarsRequest(
            symbol_or_symbols=benchmarks,
            timeframe=TimeFrame.Day,
            start=start_time,
            end=end_time,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return pd.DataFrame()

        if hasattr(df.index, "levels"):
            closes = (
                df.reset_index()
                .pivot(index="timestamp", columns="symbol", values="close")
                .sort_index()
            )
        else:
            closes = pd.DataFrame({benchmarks[0]: df["close"]})

        closes = closes.reindex(columns=benchmarks)
        closes = closes.dropna(how="all")
        if closes.empty:
            return pd.DataFrame()

        returns = closes.pct_change().fillna(0.0)
        cumulative = (1.0 + returns).cumprod() - 1.0
        return cumulative
    except Exception as exc:
        logger.warning("Failed benchmark fetch: %s", exc)
        return pd.DataFrame()
