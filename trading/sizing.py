import logging
import math

from config import (
    CryptoLatestTradeRequest,
    MAX_POSITION_PCT,
    StockLatestTradeRequest,
    crypto_data_client,
    data_client,
    is_crypto_symbol,
    trading_client,
)

logger = logging.getLogger(__name__)


def get_portfolio_value() -> float:
    """Fetch current portfolio value from Alpaca; falls back to $100 000."""
    try:
        account = trading_client.get_account()
        return float(account.portfolio_value)
    except Exception as exc:
        logger.warning(
            "Could not fetch portfolio value from Alpaca: %s. Defaulting to $100,000.", exc
        )
        return 100_000.0


def get_current_price(ticker: str) -> float | None:
    """Fetch the latest market price for a ticker."""
    normalized = ticker.strip().upper().replace("-", "/")
    try:
        if is_crypto_symbol(normalized):
            if crypto_data_client is not None and CryptoLatestTradeRequest is not None:
                req = CryptoLatestTradeRequest(symbol_or_symbols=normalized)
                trade = crypto_data_client.get_crypto_latest_trade(req)
                return float(trade[normalized].price)
        elif data_client is not None and StockLatestTradeRequest is not None:
            req = StockLatestTradeRequest(symbol_or_symbols=normalized)
            trade = data_client.get_stock_latest_trade(req)
            return float(trade[normalized].price)
    except Exception as exc:
        logger.debug("Live trade price fetch failed for %s: %s.", normalized, exc)
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            if str(pos.symbol).upper() == normalized:
                return float(pos.current_price)
    except Exception as exc:
        logger.warning(
            "Could not get current price for %s from positions: %s.", normalized, exc
        )
    return None


def calculate_position_size(
    price: float | None,
    portfolio_value: float,
    max_position_pct: float = MAX_POSITION_PCT,
    allow_fractional: bool = False,
) -> float:
    """Ring fence: allocate at most MAX_POSITION_PCT of portfolio per trade."""
    if price is None or price <= 0:
        logger.warning(
            "Invalid price (%s) for position sizing – defaulting to minimum quantity.", price
        )
        return 0.001 if allow_fractional else 1.0
    max_dollars = portfolio_value * max_position_pct
    raw_qty = max_dollars / price
    if allow_fractional:
        # Crypto supports fractional quantity; floor to millesimal precision.
        floored_qty = math.floor(raw_qty * 1000.0) / 1000.0
        return max(0.001, floored_qty)
    return float(max(1, int(raw_qty)))
