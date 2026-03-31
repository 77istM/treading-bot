"""Model Context Protocol (MCP) server for the trading bot.

Exposes the bot's data and analysis capabilities as MCP tools and resources so
that AI coding assistants (Claude Code, Codex, etc.) can query portfolio status,
trade history, signals and risk metrics without direct database access.

Run standalone (stdio transport – for Claude Desktop / claude CLI):
    python mcp_server.py

Run as HTTP SSE server (for web-based integrations):
    python mcp_server.py --transport sse

Add to Claude Desktop config (~/.config/claude/claude_desktop_config.json):
    {
      "mcpServers": {
        "trading-bot": {
          "command": "python",
          "args": ["/path/to/trading-bot/mcp_server.py"],
          "env": {}
        }
      }
    }
"""

from __future__ import annotations

import argparse
import json
import logging
import sqlite3
from datetime import date, datetime, timezone
from typing import Any

from mcp.server.fastmcp import FastMCP

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# MCP server instance
# ---------------------------------------------------------------------------

mcp = FastMCP(
    name="trading-bot",
    instructions=(
        "Trading-bot MCP server. "
        "Use the tools to query portfolio status, trade history, "
        "technical/sentiment signals, reflections and risk metrics."
    ),
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _Config:
    """Mutable server configuration (database path, etc.)."""

    db_path: str = "trading_bot.db"


_config = _Config()


def _get_conn() -> sqlite3.Connection:
    """Open a read-only connection to the trading bot database."""
    conn = sqlite3.connect(f"file:{_config.db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _get_rw_conn() -> sqlite3.Connection:
    """Open a read-write connection to the trading bot database."""
    conn = sqlite3.connect(_config.db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _row_to_dict(row: sqlite3.Row) -> dict[str, Any]:
    return dict(zip(row.keys(), tuple(row)))


# ---------------------------------------------------------------------------
# Resources
# ---------------------------------------------------------------------------


@mcp.resource("trades://recent")
def resource_recent_trades() -> str:
    """The 20 most recent trades recorded by the bot."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT id, ticker, side, qty, price, realized_pnl,
                      strategy_name, strategy_regime, reason, created_at
               FROM trades
               ORDER BY id DESC
               LIMIT 20"""
        ).fetchall()
        trades = [_row_to_dict(r) for r in rows]
        return json.dumps(trades, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("portfolio://status")
def resource_portfolio_status() -> str:
    """Current portfolio status from the latest risk snapshot."""
    try:
        conn = _get_conn()
        row = conn.execute(
            """SELECT portfolio_value, day_start_value, drawdown_pct,
                      open_positions, total_heat_pct,
                      trading_halted, halt_reason, created_at
               FROM risk_snapshots
               ORDER BY id DESC
               LIMIT 1"""
        ).fetchone()
        if row is None:
            return json.dumps({"status": "no snapshot available yet"})
        return json.dumps(_row_to_dict(row), indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("reflections://recent")
def resource_recent_reflections() -> str:
    """The 10 most recent LLM-generated trade reflections and lessons."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT id, reflection_type, ticker, outcome, pnl, lesson, created_at
               FROM reflections
               ORDER BY id DESC
               LIMIT 10"""
        ).fetchall()
        reflections = [_row_to_dict(r) for r in rows]
        return json.dumps(reflections, indent=2)
    except Exception as exc:
        return json.dumps({"error": str(exc)})


@mcp.resource("risk://snapshot")
def resource_risk_snapshot() -> str:
    """Current risk snapshot with portfolio heat and drawdown metrics."""
    return resource_portfolio_status()


# ---------------------------------------------------------------------------
# Tools
# ---------------------------------------------------------------------------


@mcp.tool(description="Get the current portfolio status and latest risk metrics.")
def get_portfolio_status() -> dict[str, Any]:
    """Return the latest risk snapshot: portfolio value, drawdown, heat, halt status."""
    try:
        conn = _get_conn()
        row = conn.execute(
            """SELECT portfolio_value, day_start_value, drawdown_pct,
                      open_positions, total_heat_pct,
                      trading_halted, halt_reason, created_at
               FROM risk_snapshots
               ORDER BY id DESC
               LIMIT 1"""
        ).fetchone()
        if row is None:
            return {"status": "no snapshot available yet"}
        return _row_to_dict(row)
    except Exception as exc:
        logger.warning("get_portfolio_status error: %s", exc)
        return {"error": str(exc)}


@mcp.tool(description="Get the N most recent trades (default 20, max 100).")
def get_recent_trades(limit: int = 20) -> list[dict[str, Any]]:
    """Return recent trade records including ticker, side, price, PnL, strategy and reason."""
    limit = max(1, min(limit, 100))
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT id, ticker, side, qty, price, realized_pnl, stop_loss_price,
                      take_profit_price, strategy_name, strategy_regime,
                      sentiment, technical_signal, reason, created_at
               FROM trades
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.warning("get_recent_trades error: %s", exc)
        return [{"error": str(exc)}]


@mcp.tool(
    description=(
        "Get trades for a specific ticker symbol. "
        "Returns up to 50 most recent trades for that ticker."
    )
)
def get_trades_for_ticker(ticker: str) -> list[dict[str, Any]]:
    """Return all recorded trades for the given ticker symbol."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT id, ticker, side, qty, price, realized_pnl, stop_loss_price,
                      take_profit_price, strategy_name, strategy_regime,
                      sentiment, technical_signal, reason, created_at
               FROM trades
               WHERE ticker = ?
               ORDER BY id DESC
               LIMIT 50""",
            (ticker.strip().upper(),),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.warning("get_trades_for_ticker error: %s", exc)
        return [{"error": str(exc)}]


@mcp.tool(
    description=(
        "Get the most recent signal snapshot for a ticker. "
        "Returns sentiment, technical indicators, strategy, and regime."
    )
)
def get_signals_for_ticker(ticker: str) -> dict[str, Any]:
    """Return the latest recorded signal values for a given ticker."""
    defaults: dict[str, Any] = {
        "sentiment": "NEUTRAL",
        "technical": "NEUTRAL",
        "geopolitics": "MEDIUM_RISK",
        "fed_rate": "NEUTRAL",
        "fear_level": "MEDIUM",
        "rsi": "NEUTRAL",
        "macd": "NEUTRAL",
        "bbands": "NEUTRAL",
        "volume": "NORMAL",
        "earnings": "UNKNOWN",
        "momentum_score": 0.0,
        "strategy_name": "unknown",
        "strategy_regime": "UNKNOWN",
    }
    try:
        conn = _get_conn()
        row = conn.execute(
            """SELECT sentiment, technical_signal, geopolitics, fed_sentiment, fear_level,
                      rsi_signal, macd_signal, bbands_signal, volume_signal,
                      earnings_flag, momentum_score, strategy_name, strategy_regime,
                      created_at
               FROM trades
               WHERE ticker = ?
               ORDER BY id DESC
               LIMIT 1""",
            (ticker.strip().upper(),),
        ).fetchone()
        if row is None:
            return {**defaults, "note": "no trade history for this ticker"}
        d = _row_to_dict(row)
        return {
            "ticker": ticker.strip().upper(),
            "sentiment": d.get("sentiment") or defaults["sentiment"],
            "technical": d.get("technical_signal") or defaults["technical"],
            "geopolitics": d.get("geopolitics") or defaults["geopolitics"],
            "fed_rate": d.get("fed_sentiment") or defaults["fed_rate"],
            "fear_level": d.get("fear_level") or defaults["fear_level"],
            "rsi": d.get("rsi_signal") or defaults["rsi"],
            "macd": d.get("macd_signal") or defaults["macd"],
            "bbands": d.get("bbands_signal") or defaults["bbands"],
            "volume": d.get("volume_signal") or defaults["volume"],
            "earnings": d.get("earnings_flag") or defaults["earnings"],
            "momentum_score": float(d.get("momentum_score") or 0.0),
            "strategy_name": d.get("strategy_name") or defaults["strategy_name"],
            "strategy_regime": d.get("strategy_regime") or defaults["strategy_regime"],
            "as_of": d.get("created_at"),
        }
    except Exception as exc:
        logger.warning("get_signals_for_ticker error: %s", exc)
        return {"error": str(exc)}


@mcp.tool(description="Get recent LLM-generated trade reflections and lessons.")
def get_reflections(limit: int = 10) -> list[dict[str, Any]]:
    """Return recent reflections from stop-loss events and EOD reviews."""
    limit = max(1, min(limit, 50))
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT id, reflection_type, trade_id, ticker, outcome,
                      pnl, lesson, created_at
               FROM reflections
               ORDER BY id DESC
               LIMIT ?""",
            (limit,),
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.warning("get_reflections error: %s", exc)
        return [{"error": str(exc)}]


@mcp.tool(description="Get the daily trade count for today.")
def get_daily_trade_count() -> dict[str, Any]:
    """Return how many trades the bot has executed today vs the configured limit."""
    try:
        conn = _get_conn()
        today = date.today().isoformat()
        row = conn.execute(
            "SELECT COUNT(*) FROM trades WHERE DATE(created_at) = ?",
            (today,),
        ).fetchone()
        count = row[0] if row else 0

        # Read limit from settings or use DB default
        limit_row = conn.execute(
            "SELECT value FROM settings WHERE key = 'daily_max_trades'",
        ).fetchone()
        limit = int(limit_row[0]) if limit_row else None

        return {
            "today": today,
            "count": count,
            "limit": limit,
            "remaining": (limit - count) if limit is not None else "unlimited",
        }
    except Exception as exc:
        logger.warning("get_daily_trade_count error: %s", exc)
        return {"error": str(exc)}


@mcp.tool(description="Get all current bot settings from the database.")
def get_settings() -> dict[str, Any]:
    """Return all key-value settings currently stored in the bot's settings table."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            "SELECT key, value, updated_at FROM settings ORDER BY key"
        ).fetchall()
        return {r["key"]: {"value": r["value"], "updated_at": r["updated_at"]} for r in rows}
    except Exception as exc:
        logger.warning("get_settings error: %s", exc)
        return {"error": str(exc)}


@mcp.tool(
    description=(
        "Update a bot setting. "
        "Allowed keys: stop_loss_pct, take_profit_pct, allow_stock_shorts, "
        "allow_crypto_shorts. Values must be numeric or boolean strings."
    )
)
def update_setting(key: str, value: str) -> dict[str, Any]:
    """Persist a setting value to the bot's settings table.

    Only the following keys are permitted to avoid accidental misconfiguration:
    stop_loss_pct, take_profit_pct, allow_stock_shorts, allow_crypto_shorts.
    """
    _ALLOWED_KEYS = {
        "stop_loss_pct",
        "take_profit_pct",
        "allow_stock_shorts",
        "allow_crypto_shorts",
    }
    key = key.strip().lower()
    if key not in _ALLOWED_KEYS:
        return {
            "success": False,
            "error": f"Key '{key}' is not in the allowed set: {sorted(_ALLOWED_KEYS)}",
        }
    try:
        conn = _get_rw_conn()
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
        return {"success": True, "key": key, "value": value}
    except Exception as exc:
        logger.warning("update_setting error: %s", exc)
        return {"success": False, "error": str(exc)}


@mcp.tool(
    description=(
        "Get a summary of today's P&L across all closed trades. "
        "Returns total realized PnL, win/loss counts and best/worst trade."
    )
)
def get_daily_pnl_summary() -> dict[str, Any]:
    """Summarise realized P&L for all trades closed today."""
    try:
        conn = _get_conn()
        today = date.today().isoformat()
        rows = conn.execute(
            """SELECT ticker, side, realized_pnl, price, created_at
               FROM trades
               WHERE DATE(created_at) = ? AND is_closing_trade = 1""",
            (today,),
        ).fetchall()
        if not rows:
            return {"today": today, "total_realized_pnl": 0.0, "closed_trades": 0}

        total = sum(r["realized_pnl"] or 0.0 for r in rows)
        wins = sum(1 for r in rows if (r["realized_pnl"] or 0.0) > 0)
        losses = sum(1 for r in rows if (r["realized_pnl"] or 0.0) < 0)
        sorted_by_pnl = sorted(rows, key=lambda r: r["realized_pnl"] or 0.0)

        return {
            "today": today,
            "total_realized_pnl": round(total, 4),
            "closed_trades": len(rows),
            "wins": wins,
            "losses": losses,
            "best_trade": _row_to_dict(sorted_by_pnl[-1]) if sorted_by_pnl else None,
            "worst_trade": _row_to_dict(sorted_by_pnl[0]) if sorted_by_pnl else None,
        }
    except Exception as exc:
        logger.warning("get_daily_pnl_summary error: %s", exc)
        return {"error": str(exc)}


@mcp.tool(
    description=(
        "Get a list of all distinct tickers that have been traded, "
        "along with their trade count and latest trade date."
    )
)
def get_traded_tickers() -> list[dict[str, Any]]:
    """Return a summary of all tickers the bot has traded."""
    try:
        conn = _get_conn()
        rows = conn.execute(
            """SELECT ticker,
                      COUNT(*) AS trade_count,
                      MAX(created_at) AS last_traded,
                      SUM(realized_pnl) AS total_realized_pnl
               FROM trades
               GROUP BY ticker
               ORDER BY trade_count DESC"""
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    except Exception as exc:
        logger.warning("get_traded_tickers error: %s", exc)
        return [{"error": str(exc)}]


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------


@mcp.prompt(
    name="trade-analysis-template",
    description="Template for requesting a trade analysis of a given ticker.",
)
def prompt_trade_analysis(ticker: str) -> str:
    """Return a structured prompt that asks the LLM to analyse a ticker trade opportunity."""
    return (
        f"Analyse the current trading opportunity for {ticker.upper()}.\n\n"
        "Use the following MCP tools to gather data:\n"
        "1. get_signals_for_ticker – retrieve the latest technical and sentiment signals\n"
        "2. get_recent_trades – review recent trades for context\n"
        "3. get_portfolio_status – check current risk exposure before recommending\n\n"
        "Based on the data, provide:\n"
        "- A BUY / SELL / HOLD recommendation with confidence (0–1)\n"
        "- Key supporting signals\n"
        "- Risk considerations (stop-loss level, position sizing)\n"
        "- One-line summary suitable for a trade log"
    )


@mcp.prompt(
    name="portfolio-review-template",
    description="Template for a full portfolio review and risk assessment.",
)
def prompt_portfolio_review() -> str:
    """Return a structured prompt for a comprehensive portfolio review."""
    return (
        "Perform a comprehensive portfolio review using the trading-bot MCP tools.\n\n"
        "Steps:\n"
        "1. Call get_portfolio_status – current value, drawdown, heat\n"
        "2. Call get_recent_trades (limit=50) – recent activity\n"
        "3. Call get_daily_pnl_summary – today's P&L\n"
        "4. Call get_reflections (limit=5) – recent LLM lessons\n\n"
        "Provide:\n"
        "- Portfolio health summary (drawdown, heat vs limits)\n"
        "- Today's performance highlights\n"
        "- Key lessons from recent reflections\n"
        "- Recommendations for risk adjustments if needed"
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Trading-bot MCP server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse"],
        default="stdio",
        help="MCP transport to use (default: stdio for local AI assistants)",
    )
    parser.add_argument(
        "--db",
        default="trading_bot.db",
        help="Path to the trading bot SQLite database (default: trading_bot.db)",
    )
    args = parser.parse_args()

    _config.db_path = args.db

    logging.basicConfig(
        level=logging.WARNING,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    mcp.run(transport=args.transport)


if __name__ == "__main__":
    main()
