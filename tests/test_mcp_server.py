"""Unit tests for the MCP server tools.

Tests run against an in-memory SQLite database so no real Alpaca / Ollama
connections are required.
"""

import sqlite3
import unittest
from datetime import date
from unittest.mock import patch


def _make_in_memory_db() -> sqlite3.Connection:
    """Create a fully-initialised in-memory trading_bot database."""
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    conn.executescript(
        """
        CREATE TABLE trades (
            id INTEGER PRIMARY KEY,
            ticker TEXT,
            side TEXT,
            qty REAL,
            price REAL,
            stop_loss_price REAL,
            take_profit_price REAL,
            is_closing_trade INTEGER DEFAULT 0,
            entry_reference_price REAL,
            price_move_pct REAL,
            strategy_name TEXT,
            strategy_regime TEXT,
            sentiment TEXT,
            technical_signal TEXT,
            geopolitics TEXT,
            fed_sentiment TEXT,
            fear_level TEXT,
            trade_analysis TEXT,
            realized_pnl REAL DEFAULT 0,
            reason TEXT,
            rsi_signal TEXT,
            macd_signal TEXT,
            bbands_signal TEXT,
            volume_signal TEXT,
            earnings_flag TEXT,
            momentum_score REAL,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE settings (
            key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE reflections (
            id INTEGER PRIMARY KEY,
            reflection_type TEXT,
            trade_id INTEGER,
            ticker TEXT,
            outcome TEXT,
            pnl REAL,
            lesson TEXT,
            raw_analysis TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        CREATE TABLE risk_snapshots (
            id INTEGER PRIMARY KEY,
            portfolio_value REAL,
            day_start_value REAL,
            drawdown_pct REAL,
            open_positions INTEGER,
            total_heat_pct REAL,
            trading_halted INTEGER DEFAULT 0,
            halt_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP
        );
        """
    )
    today = date.today().isoformat()
    _trade_cols = (
        "ticker, side, qty, price, realized_pnl, is_closing_trade, "
        "strategy_name, strategy_regime, sentiment, technical_signal, "
        "rsi_signal, macd_signal, bbands_signal, volume_signal, earnings_flag, "
        "momentum_score, reason, created_at"
    )
    _trade_placeholders = ", ".join(["?"] * 18)
    conn.executemany(
        f"INSERT INTO trades ({_trade_cols}) VALUES ({_trade_placeholders})",
        [
            # AAPL open long
            ("AAPL", "BUY", 10, 175.0, 0.0, 0,
             "momentum", "BULL", "BULLISH", "BULLISH",
             "OVERSOLD", "BULLISH", "LOWER", "HIGH", "SAFE", 0.7,
             "Signal buy", today),
            # AAPL closing trade (take-profit)
            ("AAPL", "SELL", 10, 180.0, 50.0, 1,
             "momentum", "BULL", "NEUTRAL", "BULLISH",
             "NEUTRAL", "BULLISH", "UPPER", "NORMAL", "SAFE", 0.5,
             "Take profit", today),
            # MSFT open long
            ("MSFT", "BUY", 5, 300.0, 0.0, 0,
             "mean_reversion", "SIDEWAYS", "NEUTRAL", "NEUTRAL",
             "NEUTRAL", "NEUTRAL", "MIDDLE", "NORMAL", "UNKNOWN", 0.0,
             "Mean rev buy", today),
        ],
    )
    conn.execute(
        "INSERT INTO risk_snapshots (portfolio_value, day_start_value, drawdown_pct, "
        "open_positions, total_heat_pct, trading_halted, created_at) VALUES (?,?,?,?,?,?,?)",
        (100_000.0, 102_000.0, -0.02, 2, 0.06, 0, today),
    )
    conn.execute(
        "INSERT INTO reflections (reflection_type, ticker, outcome, pnl, lesson, created_at) "
        "VALUES (?,?,?,?,?,?)",
        ("stop_loss", "AAPL", "LOSS", -150.0, "Do not chase momentum in red markets.", today),
    )
    conn.execute(
        "INSERT INTO settings (key, value, updated_at) VALUES (?,?,?)",
        ("stop_loss_pct", "0.03", today),
    )
    conn.commit()
    return conn


class TestMcpServerTools(unittest.TestCase):
    """Test each MCP tool function against an in-memory database."""

    def setUp(self) -> None:
        """Patch _get_conn and _get_rw_conn to return the in-memory DB."""
        self.db = _make_in_memory_db()

        import mcp_server

        self._patch_ro = patch.object(mcp_server, "_get_conn", return_value=self.db)
        self._patch_rw = patch.object(mcp_server, "_get_rw_conn", return_value=self.db)
        self._patch_ro.start()
        self._patch_rw.start()
        self.mcp_server = mcp_server

    def tearDown(self) -> None:
        self._patch_ro.stop()
        self._patch_rw.stop()
        self.db.close()

    # ------------------------------------------------------------------
    # get_portfolio_status
    # ------------------------------------------------------------------
    def test_get_portfolio_status_returns_snapshot(self) -> None:
        result = self.mcp_server.get_portfolio_status()
        self.assertIsInstance(result, dict)
        self.assertAlmostEqual(result["portfolio_value"], 100_000.0)
        self.assertAlmostEqual(result["drawdown_pct"], -0.02)
        self.assertEqual(result["open_positions"], 2)
        self.assertEqual(result["trading_halted"], 0)

    # ------------------------------------------------------------------
    # get_recent_trades
    # ------------------------------------------------------------------
    def test_get_recent_trades_default_limit(self) -> None:
        result = self.mcp_server.get_recent_trades()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 3)

    def test_get_recent_trades_with_limit(self) -> None:
        result = self.mcp_server.get_recent_trades(limit=2)
        self.assertEqual(len(result), 2)

    def test_get_recent_trades_limit_capped_at_100(self) -> None:
        # Even if limit=9999 is requested, the function caps it at 100
        result = self.mcp_server.get_recent_trades(limit=9999)
        self.assertLessEqual(len(result), 100)

    # ------------------------------------------------------------------
    # get_trades_for_ticker
    # ------------------------------------------------------------------
    def test_get_trades_for_ticker_returns_matching_rows(self) -> None:
        result = self.mcp_server.get_trades_for_ticker("AAPL")
        self.assertEqual(len(result), 2)
        for row in result:
            self.assertEqual(row["ticker"], "AAPL")

    def test_get_trades_for_ticker_case_insensitive(self) -> None:
        result = self.mcp_server.get_trades_for_ticker("aapl")
        self.assertEqual(len(result), 2)

    def test_get_trades_for_ticker_no_results(self) -> None:
        result = self.mcp_server.get_trades_for_ticker("UNKNOWN_TICKER")
        self.assertEqual(result, [])

    # ------------------------------------------------------------------
    # get_signals_for_ticker
    # ------------------------------------------------------------------
    def test_get_signals_for_ticker_returns_signals(self) -> None:
        result = self.mcp_server.get_signals_for_ticker("AAPL")
        self.assertIsInstance(result, dict)
        self.assertEqual(result["ticker"], "AAPL")
        self.assertIn("sentiment", result)
        self.assertIn("momentum_score", result)

    def test_get_signals_for_ticker_unknown_returns_defaults(self) -> None:
        result = self.mcp_server.get_signals_for_ticker("ZZZZ")
        self.assertIn("note", result)
        self.assertEqual(result["sentiment"], "NEUTRAL")

    # ------------------------------------------------------------------
    # get_reflections
    # ------------------------------------------------------------------
    def test_get_reflections_returns_list(self) -> None:
        result = self.mcp_server.get_reflections()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]["ticker"], "AAPL")

    def test_get_reflections_limit_capped_at_50(self) -> None:
        result = self.mcp_server.get_reflections(limit=999)
        self.assertLessEqual(len(result), 50)

    # ------------------------------------------------------------------
    # get_daily_trade_count
    # ------------------------------------------------------------------
    def test_get_daily_trade_count_today(self) -> None:
        result = self.mcp_server.get_daily_trade_count()
        self.assertIsInstance(result, dict)
        self.assertIn("count", result)
        self.assertEqual(result["count"], 3)

    # ------------------------------------------------------------------
    # get_settings
    # ------------------------------------------------------------------
    def test_get_settings_returns_dict(self) -> None:
        result = self.mcp_server.get_settings()
        self.assertIsInstance(result, dict)
        self.assertIn("stop_loss_pct", result)
        self.assertEqual(result["stop_loss_pct"]["value"], "0.03")

    # ------------------------------------------------------------------
    # update_setting
    # ------------------------------------------------------------------
    def test_update_setting_allowed_key(self) -> None:
        result = self.mcp_server.update_setting("stop_loss_pct", "0.05")
        self.assertTrue(result["success"])
        self.assertEqual(result["key"], "stop_loss_pct")

    def test_update_setting_disallowed_key_returns_error(self) -> None:
        result = self.mcp_server.update_setting("daily_max_trades", "50")
        self.assertFalse(result["success"])
        self.assertIn("error", result)

    # ------------------------------------------------------------------
    # get_daily_pnl_summary
    # ------------------------------------------------------------------
    def test_get_daily_pnl_summary_sums_closing_trades(self) -> None:
        result = self.mcp_server.get_daily_pnl_summary()
        self.assertIsInstance(result, dict)
        # The AAPL SELL (is_closing_trade=1) has realized_pnl=50.0
        self.assertAlmostEqual(result["total_realized_pnl"], 50.0)
        self.assertEqual(result["closed_trades"], 1)
        self.assertEqual(result["wins"], 1)
        self.assertEqual(result["losses"], 0)

    # ------------------------------------------------------------------
    # get_traded_tickers
    # ------------------------------------------------------------------
    def test_get_traded_tickers_returns_all_tickers(self) -> None:
        result = self.mcp_server.get_traded_tickers()
        tickers = [r["ticker"] for r in result]
        self.assertIn("AAPL", tickers)
        self.assertIn("MSFT", tickers)

    def test_get_traded_tickers_counts_are_correct(self) -> None:
        result = self.mcp_server.get_traded_tickers()
        by_ticker = {r["ticker"]: r for r in result}
        self.assertEqual(by_ticker["AAPL"]["trade_count"], 2)
        self.assertEqual(by_ticker["MSFT"]["trade_count"], 1)


class TestMcpServerResources(unittest.TestCase):
    """Test that MCP resource functions return valid JSON strings."""

    def setUp(self) -> None:
        self.db = _make_in_memory_db()

        import mcp_server

        self._patch_ro = patch.object(mcp_server, "_get_conn", return_value=self.db)
        self._patch_ro.start()
        self.mcp_server = mcp_server

    def tearDown(self) -> None:
        self._patch_ro.stop()
        self.db.close()

    def test_resource_recent_trades_is_valid_json(self) -> None:
        import json

        raw = self.mcp_server.resource_recent_trades()
        data = json.loads(raw)
        self.assertIsInstance(data, list)
        self.assertGreater(len(data), 0)

    def test_resource_portfolio_status_is_valid_json(self) -> None:
        import json

        raw = self.mcp_server.resource_portfolio_status()
        data = json.loads(raw)
        self.assertIsInstance(data, dict)
        self.assertIn("portfolio_value", data)

    def test_resource_reflections_is_valid_json(self) -> None:
        import json

        raw = self.mcp_server.resource_recent_reflections()
        data = json.loads(raw)
        self.assertIsInstance(data, list)


if __name__ == "__main__":
    unittest.main()
