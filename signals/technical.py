"""Technical analysis signals: RSI, MACD, Bollinger Bands, Volume spike, Momentum score."""
import logging
from datetime import datetime, timedelta

import numpy as np
import talib

from config import (
    CryptoBarsRequest,
    StockBarsRequest,
    TimeFrame,
    crypto_data_client,
    data_client,
    is_crypto_symbol,
)

logger = logging.getLogger(__name__)

# Minimum bars needed for all indicators (26 MACD slow + 9 signal + 20 BB window)
_MIN_BARS = 40


# ---------------------------------------------------------------------------
# Data fetching
# ---------------------------------------------------------------------------

def _fetch_bars(ticker: str) -> dict | None:
    """Fetch 90 calendar days of daily OHLCV bars via Alpaca.

    Returns a dict with numpy arrays ``close``, ``high``, ``low``, ``volume``
    or ``None`` when data is unavailable / insufficient.
    """
    symbol = ticker.strip().upper().replace("-", "/")
    crypto = is_crypto_symbol(symbol)

    if crypto and crypto_data_client is None:
        return None
    if not crypto and data_client is None:
        return None
    try:
        end = datetime.utcnow()
        start = end - timedelta(days=90)  # ~63 trading days
        if crypto:
            if CryptoBarsRequest is None:
                return None
            req = CryptoBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = crypto_data_client.get_crypto_bars(req)
        else:
            req = StockBarsRequest(
                symbol_or_symbols=symbol,
                timeframe=TimeFrame.Day,
                start=start,
                end=end,
            )
            bars = data_client.get_stock_bars(req)
        df = bars.df
        if hasattr(df.index, "levels"):
            lvl0 = df.index.get_level_values(0)
            if symbol in lvl0:
                df = df.xs(symbol, level=0)
        if df.empty or len(df) < _MIN_BARS:
            logger.warning(
                "Insufficient bars for %s (%d < %d). Technical signals will be NEUTRAL.",
                symbol, len(df), _MIN_BARS,
            )
            return None
        return {
            "close":  df["close"].values.astype(float),
            "high":   df["high"].values.astype(float),
            "low":    df["low"].values.astype(float),
            "volume": df["volume"].values.astype(float),
        }
    except Exception as exc:
        logger.warning("OHLCV fetch failed for %s: %s. Returning NEUTRAL signals.", symbol, exc)
        return None


# ---------------------------------------------------------------------------
# Individual indicator helpers
# ---------------------------------------------------------------------------

def _rsi_signal(close: np.ndarray) -> str:
    """RSI(14): < 30 → BULLISH (oversold), > 70 → BEARISH (overbought)."""
    try:
        rsi = talib.RSI(close, timeperiod=14)
        val = float(rsi[-1])
        if np.isnan(val):
            return "NEUTRAL"
        if val < 30:
            return "BULLISH"
        if val > 70:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _macd_signal(close: np.ndarray) -> str:
    """MACD(12,26,9): BULLISH when MACD ≥ signal line, BEARISH otherwise.

    A fresh crossover (confirmed by the previous bar) is weighted most highly,
    but the overall histogram direction also counts.
    """
    try:
        macd, signal, _ = talib.MACD(close, fastperiod=12, slowperiod=26, signalperiod=9)
        m, s = float(macd[-1]), float(signal[-1])
        m_prev, s_prev = float(macd[-2]), float(signal[-2])
        if any(np.isnan(v) for v in (m, s, m_prev, s_prev)):
            return "NEUTRAL"
        # Confirmed crossover in last bar
        if m > s and m_prev <= s_prev:
            return "BULLISH"
        if m < s and m_prev >= s_prev:
            return "BEARISH"
        # No fresh crossover — use current histogram direction
        if m > s:
            return "BULLISH"
        if m < s:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _bbands_signal(close: np.ndarray) -> str:
    """Bollinger Bands(20, 2σ): price below lower → BULLISH, above upper → BEARISH."""
    try:
        upper, _middle, lower = talib.BBANDS(close, timeperiod=20, nbdevup=2, nbdevdn=2)
        price = close[-1]
        u, lo = float(upper[-1]), float(lower[-1])
        if np.isnan(u) or np.isnan(lo):
            return "NEUTRAL"
        if price < lo:
            return "BULLISH"
        if price > u:
            return "BEARISH"
        return "NEUTRAL"
    except Exception:
        return "NEUTRAL"


def _volume_signal(close: np.ndarray, volume: np.ndarray) -> str:
    """Volume spike: current volume ≥ 2× the 20-day average.

    Returns SPIKE_UP when paired with a positive price move,
    SPIKE_DOWN when paired with a negative price move, else NORMAL.
    """
    try:
        if len(volume) < 22:
            return "NORMAL"
        avg_vol = float(np.mean(volume[-21:-1]))  # 20-day average excl. today
        if avg_vol <= 0:
            return "NORMAL"
        ratio = float(volume[-1]) / avg_vol
        if ratio >= 2.0:
            price_chg = close[-1] - close[-2]
            return "SPIKE_UP" if price_chg > 0 else "SPIKE_DOWN"
        return "NORMAL"
    except Exception:
        return "NORMAL"


def _momentum_score(rsi: str, macd: str, bbands: str, volume: str) -> float:
    """Composite momentum score on a −3 to +3 scale.

    Each of RSI, MACD, BBands contributes ±1; volume spike contributes ±0.5.
    """
    score = 0.0
    score += {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(rsi, 0.0)
    score += {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(macd, 0.0)
    score += {"BULLISH": 1.0, "BEARISH": -1.0, "NEUTRAL": 0.0}.get(bbands, 0.0)
    score += {"SPIKE_UP": 0.5, "SPIKE_DOWN": -0.5, "NORMAL": 0.0}.get(volume, 0.0)
    return round(score, 1)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def get_technical_signals(ticker: str) -> dict:
    """Compute a full suite of technical indicators for *ticker*.

    Returns
    -------
    dict with keys:
        ``rsi``            — BULLISH / BEARISH / NEUTRAL
        ``macd``           — BULLISH / BEARISH / NEUTRAL
        ``bbands``         — BULLISH / BEARISH / NEUTRAL
        ``volume``         — SPIKE_UP / SPIKE_DOWN / NORMAL
        ``momentum_score`` — float from −3.0 to +3.0
        ``summary``        — compact human-readable string for logging / DB
    """
    bars = _fetch_bars(ticker)
    if bars is None:
        return {
            "rsi": "NEUTRAL",
            "macd": "NEUTRAL",
            "bbands": "NEUTRAL",
            "volume": "NORMAL",
            "momentum_score": 0.0,
            "summary": "RSI=NEUTRAL MACD=NEUTRAL BB=NEUTRAL VOL=NORMAL SCORE=+0.0",
        }

    close = bars["close"]
    volume = bars["volume"]

    rsi = _rsi_signal(close)
    macd = _macd_signal(close)
    bbands = _bbands_signal(close)
    vol = _volume_signal(close, volume)
    score = _momentum_score(rsi, macd, bbands, vol)

    summary = f"RSI={rsi} MACD={macd} BB={bbands} VOL={vol} SCORE={score:+.1f}"
    logger.debug("[%s] Technical: %s", ticker, summary)

    return {
        "rsi": rsi,
        "macd": macd,
        "bbands": bbands,
        "volume": vol,
        "momentum_score": score,
        "summary": summary,
    }


def get_technical_signal(ticker: str) -> str:
    """Backward-compatible wrapper — returns the composite summary string."""
    return get_technical_signals(ticker)["summary"]
