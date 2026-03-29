import logging
import sqlite3

from config import trading_client
from trading.execution import _close_position

logger = logging.getLogger(__name__)


def monitor_positions(conn: sqlite3.Connection, stop_pct: float, take_pct: float) -> None:
    """Check all open positions and close any that have hit stop loss or take profit."""
    try:
        positions = trading_client.get_all_positions()
    except Exception as exc:
        logger.warning("Could not fetch positions: %s", exc)
        return

    for pos in positions:
        ticker = pos.symbol
        qty = abs(float(pos.qty))
        avg_entry = float(pos.avg_entry_price)
        current_price = float(pos.current_price)
        pos_side = str(pos.side).lower()

        if pos_side == "long":
            stop_price = avg_entry * (1 - stop_pct)
            take_price = avg_entry * (1 + take_pct)
            if current_price <= stop_price:
                reason = (
                    f"Stop loss triggered: price {current_price:.2f} <= "
                    f"{stop_price:.2f} ({stop_pct * 100:.1f}% down)"
                )
                logger.warning("[STOP LOSS]   LONG %s: %s", ticker, reason)
                _close_position(conn, ticker, qty, "SELL", current_price, reason)
            elif current_price >= take_price:
                reason = (
                    f"Take profit triggered: price {current_price:.2f} >= "
                    f"{take_price:.2f} ({take_pct * 100:.1f}% up)"
                )
                logger.info("[TAKE PROFIT] LONG %s: %s", ticker, reason)
                _close_position(conn, ticker, qty, "SELL", current_price, reason)

        elif pos_side == "short":
            stop_price = avg_entry * (1 + stop_pct)
            take_price = avg_entry * (1 - take_pct)
            if current_price >= stop_price:
                reason = (
                    f"Stop loss triggered: price {current_price:.2f} >= "
                    f"{stop_price:.2f} ({stop_pct * 100:.1f}% up)"
                )
                logger.warning("[STOP LOSS]   SHORT %s: %s", ticker, reason)
                _close_position(conn, ticker, qty, "BUY", current_price, reason)
            elif current_price <= take_price:
                reason = (
                    f"Take profit triggered: price {current_price:.2f} <= "
                    f"{take_price:.2f} ({take_pct * 100:.1f}% down)"
                )
                logger.info("[TAKE PROFIT] SHORT %s: %s", ticker, reason)
                _close_position(conn, ticker, qty, "BUY", current_price, reason)
