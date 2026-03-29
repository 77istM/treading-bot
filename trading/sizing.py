import logging

from config import data_client, trading_client, MAX_POSITION_PCT, StockLatestTradeRequest

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
    try:
        if data_client is not None:
            req = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trade = data_client.get_stock_latest_trade(req)
            return float(trade[ticker].price)
    except Exception as exc:
        logger.debug("Live trade price fetch failed for %s: %s.", ticker, exc)
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            if pos.symbol == ticker:
                return float(pos.current_price)
    except Exception as exc:
        logger.warning(
            "Could not get current price for %s from positions: %s.", ticker, exc
        )
    return None


def calculate_position_size(price: float, portfolio_value: float) -> int:
    """Ring fence: allocate at most MAX_POSITION_PCT of portfolio per trade."""
    if price is None or price <= 0:
        logger.warning(
            "Invalid price (%s) for position sizing – defaulting to 1 share.", price
        )
        return 1
    max_dollars = portfolio_value * MAX_POSITION_PCT
    return max(1, int(max_dollars / price))
