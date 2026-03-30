import logging
import sqlite3

from alpaca.trading.enums import OrderSide, TimeInForce
from alpaca.trading.requests import MarketOrderRequest

from config import (
    ALLOW_CRYPTO_SHORTS,
    CRYPTO_MAX_POSITION_PCT,
    MAX_ANALYSIS_LENGTH,
    MAX_POSITION_PCT,
    trading_client,
    is_crypto_symbol,
)
from db.queries import get_latest_signal_snapshot
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
    strategy_name: str = "unknown",
    strategy_regime: str = "UNKNOWN",
) -> int | None:
    """Submit a market order (long or short) with ring-fence position sizing.

    Calculates and stores stop-loss and take-profit price levels.

    Returns
    -------
    int | None
        The row id of the inserted trade record, or ``None`` on failure.
    """
    symbol = ticker.strip().upper().replace("-", "/")
    crypto = is_crypto_symbol(symbol)
    if crypto and direction == "SELL" and not ALLOW_CRYPTO_SHORTS:
        logger.info("Skipping crypto short for %s because ALLOW_CRYPTO_SHORTS=false.", symbol)
        return None

    side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL

    current_price = get_current_price(symbol)
    if current_price is None:
        logger.warning(
            "Cannot fetch current price for %s – skipping trade to preserve ring fence.", symbol
        )
        return
    portfolio_value = get_portfolio_value()
    qty = calculate_position_size(
        current_price,
        portfolio_value,
        max_position_pct=CRYPTO_MAX_POSITION_PCT if crypto else MAX_POSITION_PCT,
        allow_fractional=crypto,
    )

    price_decimals = 6 if crypto else 2

    if current_price:
        if direction == "BUY":
            stop_price = round(current_price * (1 - stop_pct), price_decimals)
            take_price = round(current_price * (1 + take_pct), price_decimals)
        else:
            stop_price = round(current_price * (1 + stop_pct), price_decimals)
            take_price = round(current_price * (1 - take_pct), price_decimals)
    else:
        stop_price = None
        take_price = None

    market_order_data = MarketOrderRequest(
        symbol=symbol,
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
            stop_price = round(filled_price * (1 - stop_pct), price_decimals)
            take_price = round(filled_price * (1 + take_pct), price_decimals)
        else:
            stop_price = round(filled_price * (1 + stop_pct), price_decimals)
            take_price = round(filled_price * (1 - take_pct), price_decimals)

    realized_pnl = calculate_realized_pnl(conn, symbol, direction, qty, filled_price)

    price_str = f"{filled_price:.4f}" if filled_price else "pending"
    stop_str = f"{stop_price:.{price_decimals}f}" if stop_price else "N/A"
    take_str = f"{take_price:.{price_decimals}f}" if take_price else "N/A"
    logger.info(
        "→ %s %sx %s @ %s | Stop: %s | Target: %s | Reason: %s",
        direction, qty, symbol, price_str, stop_str, take_str, reason,
    )

    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO trades
           (ticker, side, qty, price, stop_loss_price, take_profit_price,
            strategy_name, strategy_regime,
            sentiment, technical_signal, geopolitics, fed_sentiment, fear_level,
            rsi_signal, macd_signal, bbands_signal, volume_signal, earnings_flag, momentum_score,
            trade_analysis, realized_pnl, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            symbol, direction, float(qty), filled_price, stop_price, take_price,
            strategy_name, strategy_regime,
            signals.get("sentiment", "NEUTRAL"),
            signals.get("technical", "NEUTRAL"),
            signals.get("geopolitics", "MEDIUM_RISK"),
            signals.get("fed_rate", "NEUTRAL"),
            signals.get("fear_level", "MEDIUM"),
            signals.get("rsi", "NEUTRAL"),
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
    signal_snapshot: dict | None = None,
    entry_reference_price: float | None = None,
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
        signal_data = signal_snapshot or get_latest_signal_snapshot(conn, ticker)
        price_move_pct = None
        if entry_reference_price and entry_reference_price > 0 and filled_price is not None:
            price_move_pct = (filled_price - entry_reference_price) / entry_reference_price

        cursor = conn.cursor()
        cursor.execute(
            """INSERT INTO trades
               (ticker, side, qty, price, realized_pnl, reason,
                is_closing_trade, entry_reference_price, price_move_pct,
                strategy_name, strategy_regime,
                sentiment, technical_signal, geopolitics, fed_sentiment, fear_level,
                rsi_signal, macd_signal, bbands_signal, volume_signal, earnings_flag, momentum_score)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                ticker, close_side, qty, filled_price, realized_pnl, reason,
                1, entry_reference_price, price_move_pct,
                signal_data.get("strategy_name", "unknown"),
                signal_data.get("strategy_regime", "UNKNOWN"),
                signal_data.get("sentiment", "NEUTRAL"),
                signal_data.get("technical", "NEUTRAL"),
                signal_data.get("geopolitics", "MEDIUM_RISK"),
                signal_data.get("fed_rate", "NEUTRAL"),
                signal_data.get("fear_level", "MEDIUM"),
                signal_data.get("rsi", "NEUTRAL"),
                signal_data.get("macd", "NEUTRAL"),
                signal_data.get("bbands", "NEUTRAL"),
                signal_data.get("volume", "NORMAL"),
                signal_data.get("earnings", "UNKNOWN"),
                signal_data.get("momentum_score", 0.0),
            ),
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
