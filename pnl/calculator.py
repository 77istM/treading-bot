import logging
import sqlite3

logger = logging.getLogger(__name__)


def _side_to_direction(side) -> int:
    normalized = str(side).upper()
    if normalized in {"BULLISH", "BUY", "LONG"}:
        return 1
    if normalized in {"BEARISH", "SELL", "SHORT"}:
        return -1
    return 0


def _apply_trade_to_position(position_qty, entry_price, side, qty, price):
    """Apply one trade to a running position and return new state + realized PnL."""
    direction = _side_to_direction(side)
    if direction == 0 or qty <= 0 or price is None:
        return position_qty, entry_price, 0.0

    trade_qty = direction * qty
    realized = 0.0

    if position_qty == 0 or (position_qty > 0 and trade_qty > 0) or (position_qty < 0 and trade_qty < 0):
        new_qty = position_qty + trade_qty
        weighted_cost = (abs(position_qty) * entry_price) + (abs(trade_qty) * price)
        new_entry = weighted_cost / abs(new_qty) if new_qty != 0 else 0.0
        return new_qty, new_entry, realized

    close_qty = min(abs(position_qty), abs(trade_qty))
    if position_qty > 0 and trade_qty < 0:
        realized = (price - entry_price) * close_qty
    elif position_qty < 0 and trade_qty > 0:
        realized = (entry_price - price) * close_qty

    new_qty = position_qty + trade_qty
    if new_qty == 0:
        return 0.0, 0.0, realized
    if (position_qty > 0 > new_qty) or (position_qty < 0 < new_qty):
        return new_qty, price, realized
    return new_qty, entry_price, realized


def calculate_realized_pnl(
    conn: sqlite3.Connection, ticker: str, side, qty, price
) -> float:
    """Calculate realized PnL impact of the next trade against stored trade history."""
    if price is None:
        return 0.0

    cursor = conn.cursor()
    cursor.execute(
        "SELECT side, qty, price FROM trades WHERE ticker = ? ORDER BY id ASC",
        (ticker,),
    )
    historical_trades = cursor.fetchall()

    position_qty = 0.0
    entry_price = 0.0
    for hist_side, hist_qty, hist_price in historical_trades:
        hist_qty = float(hist_qty) if hist_qty is not None else 0.0
        hist_price = float(hist_price) if hist_price is not None else None
        position_qty, entry_price, _ = _apply_trade_to_position(
            position_qty, entry_price, hist_side, hist_qty, hist_price,
        )

    _, _, realized = _apply_trade_to_position(position_qty, entry_price, side, qty, price)
    return float(realized)
