import os
import sqlite3
from datetime import date

import pandas as pd
import talib
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# --- Configuration & Setup ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# Support multiple tickers via the TICKERS env var (comma-separated) or fall back to defaults.
_tickers_env = os.getenv("TICKERS", "AAPL,MSFT,GOOGL,AMZN,TSLA")
TICKERS = [t.strip().upper() for t in _tickers_env.split(",") if t.strip()]

# Daily trade limits (risk assessment ruleset).
# Between 20 and 100 trades are allowed per calendar day; the counters reset at midnight
# and do NOT carry over to the next day.
DAILY_MIN_TRADES = 20   # Below this count the confluence requirement is relaxed.
DAILY_MAX_TRADES = 100  # Hard cap – no new trades are submitted once this is reached.

# Initialise Clients
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET, paper=True)
llm = Ollama(model="llama3.2:3b", base_url="http://localhost:11434")

# Initialise Database
def init_db():
    conn = sqlite3.connect('trading_bot.db')
    cursor = conn.cursor()

    # Keep table compatible with old runs while adding columns needed for real PnL tracking.
    cursor.execute('''CREATE TABLE IF NOT EXISTS trades 
                      (id INTEGER PRIMARY KEY,
                       ticker TEXT,
                       side TEXT,
                       qty REAL,
                       price REAL,
                       realized_pnl REAL DEFAULT 0,
                       reason TEXT,
                       created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    cursor.execute("PRAGMA table_info(trades)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    if 'price' not in existing_columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN price REAL")
    if 'realized_pnl' not in existing_columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN realized_pnl REAL DEFAULT 0")
    if 'created_at' not in existing_columns:
        cursor.execute("ALTER TABLE trades ADD COLUMN created_at TEXT DEFAULT CURRENT_TIMESTAMP")

    conn.commit()
    return conn


def get_daily_trade_count(conn) -> int:
    """Return the number of trades executed today (calendar day, UTC).

    The count is derived purely from the ``created_at`` column in the trades
    table, so it resets automatically at midnight with no carry-over to the
    next day.
    """
    today = date.today().isoformat()  # e.g. "2026-03-29"
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM trades WHERE DATE(created_at) = ?",
        (today,),
    )
    row = cursor.fetchone()
    return row[0] if row else 0


def _side_to_direction(side):
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


def calculate_realized_pnl(conn, ticker, side, qty, price):
    """Calculate realized PnL impact of the next trade against stored trade history."""
    if price is None:
        return 0.0

    cursor = conn.cursor()
    cursor.execute(
        "SELECT side, qty, price FROM trades WHERE ticker = ? ORDER BY id ASC",
        (ticker,)
    )
    historical_trades = cursor.fetchall()

    position_qty = 0.0
    entry_price = 0.0

    for hist_side, hist_qty, hist_price in historical_trades:
        hist_qty = float(hist_qty) if hist_qty is not None else 0.0
        hist_price = float(hist_price) if hist_price is not None else None
        position_qty, entry_price, _ = _apply_trade_to_position(
            position_qty,
            entry_price,
            hist_side,
            hist_qty,
            hist_price,
        )

    _, _, realized = _apply_trade_to_position(position_qty, entry_price, side, qty, price)
    return float(realized)

# --- The "Brain" (Sentiment Analysis) ---
def analyze_sentiment(ticker):
    # 1. Fetch News (Mock URL for structure)
    url = f"https://newsapi.org/v2/everything?q={ticker}&apiKey={NEWS_API_KEY}"
    response = requests.get(url).json()
    headlines = [article['title'] for article in response.get('articles', [])[:3]]
    news_text = " | ".join(headlines)

    # 2. LangChain Prompt
    template = """
    You are an expert financial analyst. Read the following news headlines about {ticker}:
    {news}
    Deduce the market sentiment. Reply ONLY with one word: BULLISH, BEARISH, or NEUTRAL.
    """
    prompt = PromptTemplate(template=template, input_variables=["ticker", "news"])
    chain = prompt | llm
    
    sentiment = chain.invoke({"ticker": ticker, "news": news_text}).strip().upper()
    return sentiment

# --- The Technical Engine ---
def get_technical_signal(ticker):
    # In a live environment, you fetch OHLCV data from Alpaca here into a DataFrame.
    # For demonstration, we use a mock DataFrame structure.
    data = {'close': [150.0, 151.0, 152.0, 149.0, 148.0, 145.0, 143.0, 140.0, 138.0, 135.0, 130.0, 128.0, 125.0, 124.0, 123.0]}
    df = pd.DataFrame(data)
    
    # Calculate RSI using TA-Lib
    df['RSI'] = talib.RSI(df['close'], timeperiod=14)
    current_rsi = df['RSI'].iloc[-1]
    
    if current_rsi < 30:
        return "BULLISH" # Oversold
    elif current_rsi > 70:
        return "BEARISH" # Overbought
    return "NEUTRAL"

# --- Execution Engine (long-only) ---
def execute_trade(ticker, reason, conn):
    """Submit a long (BUY) market order. Short and inverse trades are prohibited."""
    market_order_data = MarketOrderRequest(
        symbol=ticker,
        qty=1,
        side=OrderSide.BUY,
        time_in_force=TimeInForce.GTC,
    )

    order = trading_client.submit_order(order_data=market_order_data)

    filled_price = None
    raw_price = getattr(order, "filled_avg_price", None)
    if raw_price is not None:
        try:
            filled_price = float(raw_price)
        except (TypeError, ValueError):
            filled_price = None

    qty = 1.0
    realized_pnl = calculate_realized_pnl(conn, ticker, "BUY", qty, filled_price)

    if filled_price is not None:
        print(f"  → BUY {ticker} @ {filled_price:.4f}. Reason: {reason}")
    else:
        print(f"  → BUY {ticker}. Fill price pending. Reason: {reason}")

    # Log to SQLite
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO trades (ticker, side, qty, price, realized_pnl, reason) VALUES (?, ?, ?, ?, ?, ?)",
        (ticker, "BUY", qty, filled_price, realized_pnl, reason),
    )
    conn.commit()


# --- Risk Assessment ---
def assess_risk(daily_count: int, sentiment: str, technical_signal: str) -> tuple:
    """Apply risk rules and decide whether to execute a BUY trade.

    Rules
    -----
    1. Hard cap: no trades when ``daily_count >= DAILY_MAX_TRADES``.
    2. Long-only policy: BEARISH-only signals never trigger a trade.
       Short and inverse trades are strictly prohibited.
    3. Confluence mode (``daily_count >= DAILY_MIN_TRADES``): both the
       sentiment *and* technical signals must be BULLISH.
    4. Minimum-reach mode (``daily_count < DAILY_MIN_TRADES``): a single
       BULLISH signal is sufficient to help meet the daily minimum.

    Returns
    -------
    (should_trade: bool, reason: str)
    """
    if daily_count >= DAILY_MAX_TRADES:
        return False, (
            f"Daily maximum of {DAILY_MAX_TRADES} trades reached – halting for today."
        )

    both_bullish = sentiment == "BULLISH" and technical_signal == "BULLISH"
    either_bullish = sentiment == "BULLISH" or technical_signal == "BULLISH"
    bearish_only = not either_bullish and (
        sentiment == "BEARISH" or technical_signal == "BEARISH"
    )

    # Never go short or inverse – bearish-only signals are skipped.
    if bearish_only:
        return False, (
            "Bearish signal detected – holding (long-only policy; short/inverse trades prohibited)."
        )

    if both_bullish:
        return True, "Confluence: both sentiment and technical signals are BULLISH."

    # Below minimum threshold: accept a single BULLISH signal to aid minimum activity.
    if daily_count < DAILY_MIN_TRADES and either_bullish:
        return True, (
            f"Single BULLISH signal accepted (daily count {daily_count} is below "
            f"the minimum target of {DAILY_MIN_TRADES} trades)."
        )

    return False, "Signals mixed or neutral – holding position to preserve capital."


# --- Main Trading Loop ---
def main():
    conn = init_db()
    daily_count = get_daily_trade_count(conn)
    print("Agent Initialised. Analysing market...")
    print(
        f"Daily trade count so far: {daily_count}/{DAILY_MAX_TRADES} "
        f"(minimum target: {DAILY_MIN_TRADES})"
    )

    if daily_count >= DAILY_MAX_TRADES:
        print(
            f"Daily maximum of {DAILY_MAX_TRADES} trades already reached. "
            "No further trades today."
        )
        return

    for ticker in TICKERS:
        # Re-read the count so the cap is respected even within one run.
        daily_count = get_daily_trade_count(conn)
        if daily_count >= DAILY_MAX_TRADES:
            print(
                f"Daily maximum of {DAILY_MAX_TRADES} trades reached mid-run. "
                "Stopping for today."
            )
            break

        sentiment = analyze_sentiment(ticker)
        technical_signal = get_technical_signal(ticker)

        print(
            f"[{ticker}] Sentiment: {sentiment} | Technicals: {technical_signal} "
            f"| Trades today: {daily_count}"
        )

        should_trade, reason = assess_risk(daily_count, sentiment, technical_signal)
        if should_trade:
            execute_trade(ticker, reason, conn)
            daily_count = get_daily_trade_count(conn)
        else:
            print(f"  → Skipping {ticker}: {reason}")

    daily_count = get_daily_trade_count(conn)
    if daily_count < DAILY_MIN_TRADES:
        print(
            f"Warning: only {daily_count} trade(s) executed today, below the minimum "
            f"target of {DAILY_MIN_TRADES}. Consider scheduling the bot more frequently "
            "or expanding the ticker list."
        )
    else:
        print(f"Session complete. Trades today: {daily_count}/{DAILY_MAX_TRADES}.")

if __name__ == "__main__":
    main()
