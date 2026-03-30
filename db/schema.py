import logging
import sqlite3

logger = logging.getLogger(__name__)


def init_db() -> sqlite3.Connection:
    """Create (or open) the SQLite database and ensure the schema is up to date."""
    conn = sqlite3.connect("trading_bot.db")
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS trades
           (id INTEGER PRIMARY KEY,
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
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
    )

    cursor.execute("PRAGMA table_info(trades)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns = [
        ("price", "REAL"),
        ("stop_loss_price", "REAL"),
        ("take_profit_price", "REAL"),
        ("realized_pnl", "REAL DEFAULT 0"),
        ("created_at", "TEXT DEFAULT CURRENT_TIMESTAMP"),
        ("is_closing_trade", "INTEGER DEFAULT 0"),
        ("entry_reference_price", "REAL"),
        ("price_move_pct", "REAL"),
        ("strategy_name", "TEXT"),
        ("strategy_regime", "TEXT"),
        ("sentiment", "TEXT"),
        ("technical_signal", "TEXT"),
        ("geopolitics", "TEXT"),
        ("fed_sentiment", "TEXT"),
        ("fear_level", "TEXT"),
        ("trade_analysis", "TEXT"),
        # Phase 2 — Richer Signals
        ("macd_signal", "TEXT"),
        ("rsi_signal", "TEXT"),
        ("bbands_signal", "TEXT"),
        ("volume_signal", "TEXT"),
        ("earnings_flag", "TEXT"),
        ("momentum_score", "REAL"),
    ]
    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")

    cursor.execute(
        """CREATE TABLE IF NOT EXISTS settings
           (key TEXT PRIMARY KEY,
            value TEXT,
            updated_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
    )

    # Reflections table — stores LLM-generated lessons after stop-losses and EOD reviews
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS reflections
           (id INTEGER PRIMARY KEY,
            reflection_type TEXT,
            trade_id INTEGER,
            ticker TEXT,
            outcome TEXT,
            pnl REAL,
            lesson TEXT,
            raw_analysis TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
    )

    # Risk snapshots — one row per trading cycle capturing portfolio health
    cursor.execute(
        """CREATE TABLE IF NOT EXISTS risk_snapshots
           (id INTEGER PRIMARY KEY,
            portfolio_value REAL,
            day_start_value REAL,
            drawdown_pct REAL,
            open_positions INTEGER,
            total_heat_pct REAL,
            trading_halted INTEGER DEFAULT 0,
            halt_reason TEXT,
            created_at TEXT DEFAULT CURRENT_TIMESTAMP)"""
    )

    conn.commit()
    return conn
