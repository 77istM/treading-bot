import logging
from datetime import datetime, timedelta

import talib

from config import data_client, StockBarsRequest, TimeFrame

logger = logging.getLogger(__name__)


def get_technical_signal(ticker: str) -> str:
    """Calculate RSI-based technical signal, using live Alpaca bars when available."""
    current_rsi = None
    try:
        if data_client is not None:
            end = datetime.utcnow()
            start = end - timedelta(days=60)
            req = StockBarsRequest(
                symbol_or_symbols=ticker,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = data_client.get_stock_bars(req)
            df = bars.df
            if hasattr(df.index, "levels"):
                lvl0 = df.index.get_level_values(0)
                if ticker in lvl0:
                    df = df.xs(ticker, level=0)
            if not df.empty and len(df) >= 15:
                close = df["close"].values.astype(float)
                rsi_arr = talib.RSI(close, timeperiod=14)
                current_rsi = float(rsi_arr[-1])
    except Exception as exc:
        logger.warning(
            "Live OHLCV fetch failed for %s: %s. Returning NEUTRAL technical signal.", ticker, exc
        )

    if current_rsi is None:
        logger.warning("No RSI data available for %s – returning NEUTRAL technical signal.", ticker)
        return "NEUTRAL"

    if current_rsi < 30:
        return "BULLISH"
    if current_rsi > 70:
        return "BEARISH"
    return "NEUTRAL"
