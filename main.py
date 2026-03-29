import os
import sqlite3
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
TICKER = "AAPL"

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

# --- Execution Engine ---
def execute_trade(ticker, side, reason, conn):
    market_order_data = MarketOrderRequest(
        symbol=ticker,
        qty=1,
        side=OrderSide.BUY if side == "BULLISH" else OrderSide.SELL,
        time_in_force=TimeInForce.GTC
    )
    
    # Execute via Alpaca
    order = trading_client.submit_order(order_data=market_order_data)

    filled_price = None
    raw_price = getattr(order, "filled_avg_price", None)
    if raw_price is not None:
        try:
            filled_price = float(raw_price)
        except (TypeError, ValueError):
            filled_price = None

    qty = 1.0
    realized_pnl = calculate_realized_pnl(conn, ticker, side, qty, filled_price)

    if filled_price is not None:
        print(f"Executed {side} order for {ticker} @ {filled_price:.4f}. Reason: {reason}")
    else:
        print(f"Executed {side} order for {ticker}. Fill price pending. Reason: {reason}")
    
    # Log to SQLite
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO trades (ticker, side, qty, price, realized_pnl, reason) VALUES (?, ?, ?, ?, ?, ?)",
        (ticker, side, qty, filled_price, realized_pnl, reason),
    )
    conn.commit()

# --- Main Trading Loop ---
def main():
    conn = init_db()
    print("Agent Initialised. Analysing market...")
    
    sentiment = analyze_sentiment(TICKER)
    technical_signal = get_technical_signal(TICKER)
    
    print(f"[{TICKER}] Sentiment: {sentiment} | Technicals: {technical_signal}")
    
    # Confluence Check: Only trade if Brain and Technicals agree
    if sentiment == "BULLISH" and technical_signal == "BULLISH":
        execute_trade(TICKER, "BULLISH", "Confluence: RSI Oversold + Positive News", conn)
    elif sentiment == "BEARISH" and technical_signal == "BEARISH":
        execute_trade(TICKER, "BEARISH", "Confluence: RSI Overbought + Negative News", conn)
    else:
        print("Signals mixed. Holding position to preserve capital.")

if __name__ == "__main__":
    main()
