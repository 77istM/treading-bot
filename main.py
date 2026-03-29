import os
import sqlite3
from datetime import date, datetime, timedelta

import pandas as pd
import talib
import requests
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import MarketOrderRequest
from alpaca.trading.enums import OrderSide, TimeInForce
from langchain_community.llms import Ollama
from langchain_core.prompts import PromptTemplate

# Try to import Alpaca data client for live OHLCV bars
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame
    _has_data_client = True
except ImportError:
    _has_data_client = False

# --- Configuration & Setup ---
ALPACA_API_KEY = os.getenv("ALPACA_API_KEY")
ALPACA_SECRET = os.getenv("ALPACA_SECRET")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")

# --- Risk Parameters ---
DAILY_MAX_TRADES = int(os.getenv("DAILY_MAX_TRADES", "1000"))   # Hard cap per 24 hours
MAX_POSITION_PCT = float(os.getenv("MAX_POSITION_PCT", "0.03")) # Ring fence: <3% per trade
STOP_LOSS_PCT = float(os.getenv("STOP_LOSS_PCT", "0.03"))       # Default 3% stop loss
TAKE_PROFIT_PCT = float(os.getenv("TAKE_PROFIT_PCT", "0.05"))   # Default 5% take profit

# --- Ticker Universe ---
# If TICKERS env var is set, use those; otherwise the bot selects from a broad universe.
_tickers_env = os.getenv("TICKERS", "")
if _tickers_env.strip():
    TICKERS = [t.strip().upper() for t in _tickers_env.split(",") if t.strip()]
else:
    TICKERS = [
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "TSLA", "NVDA", "AMD",
        "INTC", "CRM", "NFLX", "UBER", "JPM", "BAC", "GS", "WFC",
        "JNJ", "PFE", "XOM", "CVX", "WMT", "TGT", "SPY", "QQQ",
        "IWM", "COIN", "PLTR", "SOFI", "NIO", "RIVN",
    ]

# Truncation limit for AI analysis text stored in the database.
MAX_ANALYSIS_LENGTH = 2000

# --- Initialise Clients ---
trading_client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET, paper=True)
data_client = (
    StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET)
    if _has_data_client and ALPACA_API_KEY and ALPACA_SECRET
    else None
)
llm = Ollama(model="llama3.2:3b", base_url="http://localhost:11434")


# --- Database Initialisation ---
def init_db():
    conn = sqlite3.connect("trading_bot.db")
    cursor = conn.cursor()

    cursor.execute('''CREATE TABLE IF NOT EXISTS trades
                      (id INTEGER PRIMARY KEY,
                       ticker TEXT,
                       side TEXT,
                       qty REAL,
                       price REAL,
                       stop_loss_price REAL,
                       take_profit_price REAL,
                       sentiment TEXT,
                       technical_signal TEXT,
                       geopolitics TEXT,
                       fed_sentiment TEXT,
                       fear_level TEXT,
                       trade_analysis TEXT,
                       realized_pnl REAL DEFAULT 0,
                       reason TEXT,
                       created_at TEXT DEFAULT CURRENT_TIMESTAMP)''')

    cursor.execute("PRAGMA table_info(trades)")
    existing_columns = {row[1] for row in cursor.fetchall()}

    new_columns = [
        ("price", "REAL"),
        ("stop_loss_price", "REAL"),
        ("take_profit_price", "REAL"),
        ("realized_pnl", "REAL DEFAULT 0"),
        ("created_at", "TEXT DEFAULT CURRENT_TIMESTAMP"),
        ("sentiment", "TEXT"),
        ("technical_signal", "TEXT"),
        ("geopolitics", "TEXT"),
        ("fed_sentiment", "TEXT"),
        ("fear_level", "TEXT"),
        ("trade_analysis", "TEXT"),
    ]
    for col_name, col_type in new_columns:
        if col_name not in existing_columns:
            cursor.execute(f"ALTER TABLE trades ADD COLUMN {col_name} {col_type}")

    cursor.execute('''CREATE TABLE IF NOT EXISTS settings
                      (key TEXT PRIMARY KEY,
                       value TEXT,
                       updated_at TEXT DEFAULT CURRENT_TIMESTAMP)''')
    conn.commit()
    return conn


def read_setting(conn, key, default):
    """Read a numeric setting from the settings table, falling back to default."""
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT value FROM settings WHERE key = ?", (key,))
        row = cursor.fetchone()
        if row:
            return float(row[0])
    except Exception:
        pass
    return default


def get_daily_trade_count(conn) -> int:
    """Return the number of trades executed today (calendar day, UTC)."""
    today = date.today().isoformat()
    cursor = conn.cursor()
    cursor.execute(
        "SELECT COUNT(*) FROM trades WHERE DATE(created_at) = ?",
        (today,),
    )
    row = cursor.fetchone()
    return row[0] if row else 0


# --- PnL Helpers (unchanged logic) ---
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


# --- Portfolio & Position Sizing ---
def get_portfolio_value() -> float:
    """Fetch current portfolio value from Alpaca; falls back to $100 000."""
    try:
        account = trading_client.get_account()
        return float(account.portfolio_value)
    except Exception:
        return 100_000.0


def get_current_price(ticker) -> float | None:
    """Fetch the latest market price for a ticker."""
    try:
        if data_client is not None:
            req = StockLatestTradeRequest(symbol_or_symbols=ticker)
            trade = data_client.get_stock_latest_trade(req)
            return float(trade[ticker].price)
    except Exception:
        pass
    try:
        positions = trading_client.get_all_positions()
        for pos in positions:
            if pos.symbol == ticker:
                return float(pos.current_price)
    except Exception:
        pass
    return None


def calculate_position_size(price: float, portfolio_value: float) -> int:
    """Ring fence: allocate at most MAX_POSITION_PCT of portfolio per trade."""
    if price is None or price <= 0:
        print(f"  [WARN] Invalid price ({price}) for position sizing – defaulting to 1 share.")
        return 1
    max_dollars = portfolio_value * MAX_POSITION_PCT
    return max(1, int(max_dollars / price))


# --- News Helper ---
def _fetch_headlines(query: str, page_size: int = 5) -> list[str]:
    """Fetch news headlines from NewsAPI for the given query."""
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        f"&apiKey={NEWS_API_KEY}"
        f"&pageSize={page_size}"
        f"&language=en"
        f"&sortBy=publishedAt"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        return [a["title"] for a in resp.get("articles", [])[:page_size] if a.get("title")]
    except Exception:
        return []


# --- Signal Analysis Functions ---
def analyze_sentiment(ticker: str) -> str:
    """Assess stock-specific sentiment from recent news headlines."""
    headlines = _fetch_headlines(ticker, page_size=5)
    if not headlines:
        return "NEUTRAL"

    news_text = " | ".join(headlines)
    template = """
You are an expert financial analyst. Read the following news headlines about {ticker}:
{news}
Deduce the market sentiment. Reply ONLY with one word: BULLISH, BEARISH, or NEUTRAL.
"""
    prompt = PromptTemplate(template=template, input_variables=["ticker", "news"])
    try:
        result = (prompt | llm).invoke({"ticker": ticker, "news": news_text}).strip().upper()
    except Exception:
        return "NEUTRAL"
    for word in ("BULLISH", "BEARISH", "NEUTRAL"):
        if word in result:
            return word
    return "NEUTRAL"


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
    except Exception as e:
        print(f"  [WARN] Live OHLCV fetch failed for {ticker}: {e}. Using mock data.")

    if current_rsi is None:
        mock_close = [
            150.0, 151.0, 152.0, 149.0, 148.0, 145.0, 143.0, 140.0,
            138.0, 135.0, 130.0, 128.0, 125.0, 124.0, 123.0,
        ]
        df_mock = pd.DataFrame({"close": mock_close})
        rsi_arr = talib.RSI(df_mock["close"].values, timeperiod=14)
        current_rsi = float(rsi_arr[-1])

    if current_rsi < 30:
        return "BULLISH"
    if current_rsi > 70:
        return "BEARISH"
    return "NEUTRAL"


def analyze_geopolitics() -> str:
    """Assess global geopolitical risk level from recent news."""
    headlines = _fetch_headlines("geopolitical risk war conflict sanctions trade war", page_size=5)
    if not headlines:
        return "MEDIUM_RISK"
    news_text = " | ".join(headlines)
    template = """
You are a geopolitical risk analyst. Read these recent news headlines:
{news}
Assess the current level of geopolitical risk to financial markets.
Reply ONLY with one of: LOW_RISK, MEDIUM_RISK, or HIGH_RISK.
"""
    try:
        chain = PromptTemplate(template=template, input_variables=["news"]) | llm
        result = chain.invoke({"news": news_text}).strip().upper()
    except Exception:
        return "MEDIUM_RISK"
    for level in ("LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"):
        if level in result:
            return level
    return "MEDIUM_RISK"


def analyze_fed_rate() -> str:
    """Assess Federal Reserve rate policy stance from recent news."""
    headlines = _fetch_headlines("Federal Reserve interest rate inflation FOMC", page_size=5)
    if not headlines:
        return "NEUTRAL"
    news_text = " | ".join(headlines)
    template = """
You are a monetary policy analyst. Read these recent Federal Reserve headlines:
{news}
Assess the Fed's current policy stance. Reply ONLY with one of: HAWKISH, DOVISH, or NEUTRAL.
"""
    try:
        chain = PromptTemplate(template=template, input_variables=["news"]) | llm
        result = chain.invoke({"news": news_text}).strip().upper()
    except Exception:
        return "NEUTRAL"
    for stance in ("HAWKISH", "DOVISH", "NEUTRAL"):
        if stance in result:
            return stance
    return "NEUTRAL"


def analyze_market_fear() -> str:
    """Assess market fear/VIX level from recent headlines."""
    headlines = _fetch_headlines("VIX volatility index market fear S&P 500 crash rally", page_size=5)
    if not headlines:
        return "MEDIUM"
    news_text = " | ".join(headlines)
    template = """
You are a market risk analyst. Read these recent market headlines:
{news}
Assess the current level of market fear and volatility.
Reply ONLY with one of: HIGH (high fear, VIX>30), MEDIUM (moderate fear, VIX 15-30), or LOW (low fear, VIX<15).
"""
    try:
        chain = PromptTemplate(template=template, input_variables=["news"]) | llm
        result = chain.invoke({"news": news_text}).strip().upper()
    except Exception:
        return "MEDIUM"
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in result:
            return level
    return "MEDIUM"


# --- Pre-Trade AI Analysis ---
def pre_trade_analysis(
    ticker: str,
    sentiment: str,
    technical: str,
    geopolitics: str,
    fed_rate: str,
    fear_level: str,
    stop_pct: float,
    take_pct: float,
) -> tuple:
    """Synthesise all signals through Ollama and decide trade direction.

    The bot explicitly declares how much it is willing to lose and profit
    before each trade.

    Returns
    -------
    (direction: str, should_trade: bool, reason: str, full_analysis: str)
    direction is one of "BUY", "SELL", or "HOLD".
    """
    template = """
You are a hedge fund portfolio manager with billions under management.
Analyze this potential trade for {ticker}:

MARKET SIGNALS:
- Stock Sentiment (news):   {sentiment}
- Technical Signal (RSI):   {technical}
- Geopolitical Risk:        {geopolitics}
- Federal Reserve Stance:   {fed_rate}
- Market Fear / VIX Level:  {fear_level}

RISK PARAMETERS:
- Maximum loss I am willing to accept: {stop_pct}% (stop loss)
- Target profit I want to capture:     {take_pct}% (take profit)

Think step by step:
1. What is the overall market environment saying?
2. Is this stock likely to go UP or DOWN?
3. Should I go LONG (BUY, profit if price rises) or SHORT (SELL, profit if price falls)?
4. Is the risk-reward ratio favourable given a {stop_pct}% stop and {take_pct}% target?
5. What is my confidence level?

Respond in EXACTLY this format (nothing else):
DIRECTION: [BUY or SELL or HOLD]
TRADE: [YES or NO]
CONFIDENCE: [HIGH or MEDIUM or LOW]
WILLING_TO_LOSE: {stop_pct}%
TARGETING_PROFIT: {take_pct}%
REASON: [One sentence explanation]
"""
    prompt = PromptTemplate(
        template=template,
        input_variables=[
            "ticker", "sentiment", "technical", "geopolitics",
            "fed_rate", "fear_level", "stop_pct", "take_pct",
        ],
    )
    try:
        full_analysis = (prompt | llm).invoke({
            "ticker": ticker,
            "sentiment": sentiment,
            "technical": technical,
            "geopolitics": geopolitics,
            "fed_rate": fed_rate,
            "fear_level": fear_level,
            "stop_pct": f"{stop_pct * 100:.1f}",
            "take_pct": f"{take_pct * 100:.1f}",
        }).strip()
    except Exception as e:
        return "HOLD", False, f"LLM error: {e}", ""

    direction = "HOLD"
    should_trade = False
    reason = "No clear signal."

    for line in full_analysis.splitlines():
        line = line.strip()
        if line.upper().startswith("DIRECTION:"):
            d = line.split(":", 1)[1].strip().upper()
            if d in ("BUY", "SELL", "HOLD"):
                direction = d
        elif line.upper().startswith("TRADE:"):
            should_trade = line.split(":", 1)[1].strip().upper() == "YES"
        elif line.upper().startswith("REASON:"):
            reason = line.split(":", 1)[1].strip()

    if direction == "HOLD":
        should_trade = False

    return direction, should_trade, reason, full_analysis


# --- Risk Assessment (hard rules only; AI handles strategy) ---
def assess_risk(daily_count: int, direction: str, should_trade: bool, reason: str) -> tuple:
    """Apply hard risk rules on top of the AI pre-trade analysis.

    Returns
    -------
    (trade_ok: bool, final_direction: str, final_reason: str)
    """
    if daily_count >= DAILY_MAX_TRADES:
        return False, "HOLD", f"Daily maximum of {DAILY_MAX_TRADES} trades reached – halting for today."

    if not should_trade or direction == "HOLD":
        return False, "HOLD", reason

    if direction not in ("BUY", "SELL"):
        return False, "HOLD", f"Unknown direction '{direction}' – holding."

    return True, direction, reason


# --- Execution Engine ---
def execute_trade(
    conn,
    ticker: str,
    direction: str,
    reason: str,
    full_analysis: str,
    signals: dict,
    stop_pct: float,
    take_pct: float,
) -> None:
    """Submit a market order (long or short) with ring-fence position sizing.

    Calculates and stores stop-loss and take-profit price levels.
    """
    side = OrderSide.BUY if direction == "BUY" else OrderSide.SELL

    current_price = get_current_price(ticker)
    if current_price is None:
        print(f"  [WARN] Cannot fetch current price for {ticker} – skipping trade to preserve ring fence.")
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
    print(f"  → {direction} {qty}x {ticker} @ {price_str} | Stop: {stop_str} | Target: {take_str}")
    print(f"    Reason: {reason}")

    cursor = conn.cursor()
    cursor.execute(
        """INSERT INTO trades
           (ticker, side, qty, price, stop_loss_price, take_profit_price,
            sentiment, technical_signal, geopolitics, fed_sentiment, fear_level,
            trade_analysis, realized_pnl, reason)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            ticker, direction, float(qty), filled_price, stop_price, take_price,
            signals.get("sentiment", "NEUTRAL"),
            signals.get("technical", "NEUTRAL"),
            signals.get("geopolitics", "MEDIUM_RISK"),
            signals.get("fed_rate", "NEUTRAL"),
            signals.get("fear_level", "MEDIUM"),
            full_analysis[:MAX_ANALYSIS_LENGTH] if full_analysis else reason,
            realized_pnl, reason,
        ),
    )
    conn.commit()


# --- Position Monitor (stop loss / take profit) ---
def _close_position(conn, ticker: str, qty: float, close_side: str, current_price: float, reason: str) -> None:
    """Submit a closing market order and log it."""
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
        print(f"  → Closed {ticker}: {close_side} {qty} @ {filled_price} | PnL: {realized_pnl:.2f}")
    except Exception as e:
        print(f"  [ERROR] Failed to close {ticker}: {e}")


def monitor_positions(conn, stop_pct: float, take_pct: float) -> None:
    """Check all open positions and close any that have hit stop loss or take profit."""
    try:
        positions = trading_client.get_all_positions()
    except Exception as e:
        print(f"  [WARN] Could not fetch positions: {e}")
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
                reason = (f"Stop loss triggered: price {current_price:.2f} <= "
                          f"{stop_price:.2f} ({stop_pct * 100:.1f}% down)")
                print(f"  [STOP LOSS]   LONG {ticker}: {reason}")
                _close_position(conn, ticker, qty, "SELL", current_price, reason)
            elif current_price >= take_price:
                reason = (f"Take profit triggered: price {current_price:.2f} >= "
                          f"{take_price:.2f} ({take_pct * 100:.1f}% up)")
                print(f"  [TAKE PROFIT] LONG {ticker}: {reason}")
                _close_position(conn, ticker, qty, "SELL", current_price, reason)

        elif pos_side == "short":
            stop_price = avg_entry * (1 + stop_pct)
            take_price = avg_entry * (1 - take_pct)
            if current_price >= stop_price:
                reason = (f"Stop loss triggered: price {current_price:.2f} >= "
                          f"{stop_price:.2f} ({stop_pct * 100:.1f}% up)")
                print(f"  [STOP LOSS]   SHORT {ticker}: {reason}")
                _close_position(conn, ticker, qty, "BUY", current_price, reason)
            elif current_price <= take_price:
                reason = (f"Take profit triggered: price {current_price:.2f} <= "
                          f"{take_price:.2f} ({take_pct * 100:.1f}% down)")
                print(f"  [TAKE PROFIT] SHORT {ticker}: {reason}")
                _close_position(conn, ticker, qty, "BUY", current_price, reason)


# --- Main Trading Loop ---
def main():
    conn = init_db()

    # Read configurable risk settings (may be overridden via the dashboard UI)
    stop_pct = read_setting(conn, "stop_loss_pct", STOP_LOSS_PCT)
    take_pct = read_setting(conn, "take_profit_pct", TAKE_PROFIT_PCT)

    daily_count = get_daily_trade_count(conn)
    print("Hedge Fund Bot Initialised. Analysing market...")
    print(f"Trades today: {daily_count}/{DAILY_MAX_TRADES}")
    print(f"Risk params  : Stop={stop_pct * 100:.1f}% | Take={take_pct * 100:.1f}% | Ring fence={MAX_POSITION_PCT * 100:.1f}%")

    if daily_count >= DAILY_MAX_TRADES:
        print(f"Daily maximum of {DAILY_MAX_TRADES} trades already reached. Exiting.")
        return

    # Step 1: Monitor open positions for stop loss / take profit
    print("\n[Position Monitor] Checking open positions...")
    monitor_positions(conn, stop_pct, take_pct)

    # Step 2: Compute market-wide signals once (shared across all tickers)
    print("\n[Market Analysis] Gathering market-wide signals...")
    geopolitics = analyze_geopolitics()
    fed_rate = analyze_fed_rate()
    fear_level = analyze_market_fear()
    print(f"  Geopolitics: {geopolitics} | Fed Rate: {fed_rate} | Market Fear: {fear_level}")

    # Step 3: Per-ticker analysis and execution
    for ticker in TICKERS:
        daily_count = get_daily_trade_count(conn)
        if daily_count >= DAILY_MAX_TRADES:
            print(f"\nDaily maximum of {DAILY_MAX_TRADES} trades reached. Stopping.")
            break

        print(f"\n[{ticker}] Analysing... (trades today: {daily_count})")

        sentiment = analyze_sentiment(ticker)
        technical = get_technical_signal(ticker)
        print(f"  Sentiment: {sentiment} | Technical: {technical}")

        signals = {
            "sentiment": sentiment,
            "technical": technical,
            "geopolitics": geopolitics,
            "fed_rate": fed_rate,
            "fear_level": fear_level,
        }

        # Pre-trade analysis: bot thinks through risk/reward before committing
        direction, should_trade, reason, full_analysis = pre_trade_analysis(
            ticker, sentiment, technical, geopolitics, fed_rate, fear_level,
            stop_pct, take_pct,
        )
        print(f"  AI Decision: {direction} | Trade: {should_trade} | {reason}")

        trade_ok, final_direction, final_reason = assess_risk(
            daily_count, direction, should_trade, reason
        )

        if trade_ok:
            execute_trade(conn, ticker, final_direction, final_reason, full_analysis, signals, stop_pct, take_pct)
        else:
            print(f"  → Skipping {ticker}: {final_reason}")

    daily_count = get_daily_trade_count(conn)
    print(f"\nSession complete. Trades today: {daily_count}/{DAILY_MAX_TRADES}.")


if __name__ == "__main__":
    main()
