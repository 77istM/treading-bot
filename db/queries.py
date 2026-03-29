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
