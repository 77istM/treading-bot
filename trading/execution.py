import logging
import sqlite3

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config import trading_client, MAX_ANALYSIS_LENGTH
from pnl.calculator import calculate_realized_pnl
from trading.sizing import get_current_price, get_portfolio_value, calculate_position_size

logger = logging.getLogger(__name__)


def execute_trade(
    conn: sqlite3.Connection,
    ticker: str,
    direction: str,
    reason: str,
    full_analysis: str,
    signals: dict,
    stop_pct: float,
    take_pct: float,
) -> int | None:
    """Submit a market order (long or short) with ring-fence position sizing.

    Calculates and stores stop-loss and take-profit price levels.

    Returns
    -------
    int | None
        The row id of the inserted trade record, or ``None`` on failure.
    """
    side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL

    current_price = get_current_price(ticker)
    if current_price is None:
        logger.warning(
            "Cannot fetch current price for %s – skipping trade to preserve ring fence.", ticker
        )
        return
    portfolio_value = get_portfolio_value()
    qty = calculate_position_size(current_price, portfolio_value)

    if current_price:
        if direction == "BUY":
            stop_price = round(current_price * (1 - stop_pct), 2)
            take_price = round(current_price * (1 + take_pct), 2)
        else:
            stop_price = round(current_price * (1 + stop_pct), 2)
            take_price = round(current_price * (1 - take_pct), 2)
    else:
        stop_price = None
        take_price = None

    market_order_data = MarketOrderRequest(
        symbol=ticker,
        qty=qty,
        side=side,
        time_in_force=TimeInForce.GTC,
    )
    order = trading_client.submit_order(order_data=market_order_data)

    filled_price = None
    raw_price = getattr(order, "filled_avg_price", None)
    if raw_price is not None:
        try:
            filled_price = float(raw_price)
        except (TypeError, ValueError):
            filled_price = current_price
    else:
        filled_price = current_price

    # Recalculate stop/take using actual fill price when available
    if filled_price:
        if direction == "BUY":
            stop_price = round(filled_price * (1 - stop_pct), 2)
            take_price = round(filled_price * (1 + take_pct), 2)
        else:
            stop_price = round(filled_price * (1 + stop_pct), 2)
            take_price = round(filled_price * (1 - take_pct), 2)

    realized_pnl = calculate_realized_pnl(conn, ticker, direction, qty, filled_price)

    price_str = f"{filled_price:.4f}" if filled_price else "pending"
    stop_str = f"{stop_price:.2f}" if stop_price else "N/A"
    take_str = f"{take_price:.2f}" if take_price else "N/A"
    logger.info(
        "→ %s %dx %s @ %s | Stop: %s | Target: %s | Reason: %s",
        direction, qty, ticker, price_str, stop_str, take_str, reason,
    )

    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO trades
           (ticker, side, qty, price, stop_loss_price, take_profit_price,
            sentiment, technical_signal, geopolitics, fed_sentiment, fear_level,
            macd_signal, bbands_signal, volume_signal, earnings_flag, momentum_score,
            trade_analysis, realized_pnl, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker, direction, float(qty), filled_price, stop_price, take_price,
            signals.get("sentiment", "NEUTRAL"),
            signals.get("technical", "NEUTRAL"),
            signals.get("geopolitics", "MEDIUM_RISK"),
            signals.get("fed_rate", "NEUTRAL"),
            signals.get("fear_level", "MEDIUM"),
            signals.get("macd", "NEUTRAL"),
            signals.get("bbands", "NEUTRAL"),
            signals.get("volume", "NORMAL"),
            signals.get("earnings", "UNKNOWN"),
            signals.get("momentum_score", 0.0),
            full_analysis[:MAX_ANALYSIS_LENGTH] if full_analysis else reason,
            realized_pnl, reason,
        ),
    )
    conn.commit()
    return cursor.lastrowid


def _close_position(
    conn: sqlite3.Connection,
    ticker: str,
    qty: float,
    close_side: str,
    current_price: float,
    reason: str,
) -> int | None:
    """Submit a closing market order, log it, and return the new trade row id."""
    side = OrderSide.BUY if close_side == "BUY" else OrderSide.SELL
    try:
        order_data = MarketOrderRequest(
            symbol=ticker,
            qty=qty,
            side=side,
            time_in_force=TimeInForce.GTC,
        )
        order = trading_client.submit_order(order_data=order_data)

        filled_price = None
        raw_price = getattr(order, "filled_avg_price", None)
        if raw_price is not None:
            try:
                filled_price = float(raw_price)
            except (TypeError, ValueError):
                filled_price = current_price
        else:
            filled_price = current_price

        realized_pnl = calculate_realized_pnl(conn, ticker, close_side, qty, filled_price)

        cursor = conn.cursor()
        cursor.execute(
            "INSERT INTO trades (ticker, side, qty, price, realized_pnl, reason) VALUES (?, ?, ?, ?, ?, ?)",
            (ticker, close_side, qty, filled_price, realized_pnl, reason),
        )
        conn.commit()
        trade_id = cursor.lastrowid
        logger.info(
            "→ Closed %s: %s %s @ %s | PnL: %.2f",
            ticker, close_side, qty, filled_price, realized_pnl,
        )
        return trade_id
    except Exception as exc:
        logger.error("Failed to close position %s: %s", ticker, exc)
        return None
