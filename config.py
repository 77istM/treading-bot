import logging
import logging.handlers
import os

import requests
from alpaca.trading.client import TradingClient
from langchain_community.llms import Ollama

try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest, StockLatestTradeRequest
    from alpaca.data.timeframe import TimeFrame
    _has_data_client = True
except ImportError:
    StockHistoricalDataClient = None
    StockBarsRequest = None
    StockLatestTradeRequest = None
    TimeFrame = None
    _has_data_client = False

# --- Logging Setup ---
_log_format = logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
_handlers: list[logging.Handler] = [logging.StreamHandler()]
try:
    _file_handler = logging.handlers.RotatingFileHandler(
        "bot.log", maxBytes=5_000_000, backupCount=3
    )
    _file_handler.setFormatter(_log_format)
    _handlers.append(_file_handler)
except OSError:
    pass  # File logging unavailable (permissions/disk); continue with console only
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", handlers=_handlers)
logger = logging.getLogger(__name__)

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


def _validate_credentials() -> None:
    """Raise EnvironmentError if required API credentials are missing."""
    missing = [name for name, val in [
        ("ALPACA_API_KEY", ALPACA_API_KEY),
        ("ALPACA_SECRET", ALPACA_SECRET),
    ] if not val]
    if missing:
        raise EnvironmentError(
            f"Missing required environment variables: {', '.join(missing)}. "
            "Set them in your .env file before starting the bot."
        )
    if not NEWS_API_KEY:
        logger.warning("NEWS_API_KEY is not set – news/sentiment signals will be unavailable.")


def _check_ollama_health() -> None:
    """Verify Ollama is reachable before the main loop starts."""
    try:
        resp = requests.get("http://localhost:11434/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            "Ollama is not reachable at http://localhost:11434. "
            "Start it with: ollama serve\n"
            f"  Detail: {exc}"
        ) from exc
