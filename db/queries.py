import logging
import sqlite3
from datetime import date, datetime, timezone

logger = logging.getLogger(__name__)


def read_setting(conn: sqlite3.Connection, key: str, default: float) -> float:
    """Read a numeric setting from the settings table, falling back to default."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return float(row[0])
    except Exception as exc:
        logger.warning(
            "Could not read setting '%s' from DB: %s. Using default %s.", key, exc, default
        )
    return default


def write_setting(conn: sqlite3.Connection, key: str, value: float) -> None:
    """Persist a numeric setting to the settings table."""
    try:
        conn.execute(
            "CREATE TABLE IF NOT EXISTS settings "
            "(key TEXT PRIMARY KEY, value TEXT, updated_at TEXT)"
        )
        conn.execute(
            "INSERT OR REPLACE INTO settings (key, value, updated_at) VALUES (?, ?, ?)",
            (key, str(value), datetime.now(timezone.utc).isoformat()),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to write setting '%s'=%s to DB: %s", key, value, exc)


def get_daily_trade_count(conn: sqlite3.Connection) -> int:
    """Return the number of trades executed today (calendar day, UTC)."""
    today = date.today().isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM trades WHERE DATE(created_at) = ?",
        (today,),
    )
    row = cursor.fetchone()
    return row[0] if row else 0


def get_latest_signal_snapshot(conn: sqlite3.Connection, ticker: str) -> dict:
    """Return the most recent signal snapshot recorded for *ticker*.

    This is used to carry signal context onto position-closing rows so
    realized PnL can be attributed by signal quality.
    """
    defaults = {
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
        row = conn.execute(
            """SELECT sentiment, technical_signal, geopolitics, fed_sentiment, fear_level,
                      rsi_signal, macd_signal, bbands_signal, volume_signal,
                     earnings_flag, momentum_score, strategy_name, strategy_regime
               FROM trades
               WHERE ticker = ?
               ORDER BY id DESC
               LIMIT 1""",
            (ticker,),
        ).fetchone()
        if not row:
            return defaults

        return {
            "sentiment": row[0] or defaults["sentiment"],
            "technical": row[1] or defaults["technical"],
            "geopolitics": row[2] or defaults["geopolitics"],
            "fed_rate": row[3] or defaults["fed_rate"],
            "fear_level": row[4] or defaults["fear_level"],
            "rsi": row[5] or defaults["rsi"],
            "macd": row[6] or defaults["macd"],
            "bbands": row[7] or defaults["bbands"],
            "volume": row[8] or defaults["volume"],
            "earnings": row[9] or defaults["earnings"],
            "momentum_score": float(row[10] or defaults["momentum_score"]),
            "strategy_name": row[11] or defaults["strategy_name"],
            "strategy_regime": row[12] or defaults["strategy_regime"],
        }
    except Exception as exc:
        logger.warning("Failed to get latest signal snapshot for %s: %s", ticker, exc)
        return defaults
