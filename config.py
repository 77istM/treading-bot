import logging
import logging.handlers
import os

import requests
from alpaca.trading.client import TradingClient
from langchain_community.llms import Ollama

from hardening.secrets import SecretsVault

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
_vault = SecretsVault()


def _env_int(name: str, default: int) -> int:
    raw = _vault.get(name, str(default))
    if raw is None:
        return default
    try:
        return int(raw)
    except ValueError:
        logger.warning("Invalid %s=%r. Using default=%d.", name, raw, default)
        return default


def _env_float(name: str, default: float) -> float:
    raw = _vault.get(name, str(default))
    if raw is None:
        return default
    try:
        return float(raw)
    except ValueError:
        logger.warning("Invalid %s=%r. Using default=%s.", name, raw, default)
        return default


def _env_bool(name: str, default: bool = False) -> bool:
    raw = _vault.get(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


ALPACA_API_KEY = _vault.get("ALPACA_API_KEY")
ALPACA_SECRET = _vault.get("ALPACA_SECRET")
NEWS_API_KEY = _vault.get("NEWS_API_KEY")

# --- Risk Parameters ---
DAILY_MAX_TRADES = _env_int("DAILY_MAX_TRADES", 1000)   # Hard cap per 24 hours
MAX_POSITION_PCT = _env_float("MAX_POSITION_PCT", 0.03) # Ring fence: <3% per trade
STOP_LOSS_PCT = _env_float("STOP_LOSS_PCT", 0.03)       # Default 3% stop loss
TAKE_PROFIT_PCT = _env_float("TAKE_PROFIT_PCT", 0.05)   # Default 5% take profit

# --- Continuous Loop ---
LOOP_INTERVAL_SECONDS = _env_int("LOOP_INTERVAL_SECONDS", 300)   # 5-min default cycle

# --- Benchmark ETFs (US-listed proxies for global indices) ---
# SPY=S&P500, EWU=FTSE100, EWJ=Nikkei225, EWQ=CAC40, EWG=DAX
# Used by Phase 6 (Performance Attribution) to compute alpha vs. global indices.
BENCHMARK_TICKERS = ["SPY", "EWU", "EWJ", "EWQ", "EWG"]

# --- Ticker Universe ---
# ETFs (macro hedge vehicles) + core equities.
# If TICKERS env var is set, use those; otherwise the bot selects from the curated universe.
_tickers_env = _vault.get("TICKERS", "") or ""
if _tickers_env.strip():
    TICKERS = [t.strip().upper() for t in _tickers_env.split(",") if t.strip()]
else:
    # ETFs first (macro hedges), then quality equities
    TICKERS = [
        # Broad-market & factor ETFs
        "SPY", "QQQ", "IWM", "EFA", "EEM",
        # Global index proxies (benchmarks also tradeable)
        "EWU", "EWJ", "EWQ", "EWG",
        # Sector ETFs
        "XLF", "XLK", "XLE", "XLV", "GLD", "TLT",
        # Core equities
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
        "JPM", "BAC", "JNJ", "XOM", "WMT",
    ]

# --- Portfolio Risk Limits ---
MAX_DAILY_DRAWDOWN_PCT = _env_float("MAX_DAILY_DRAWDOWN_PCT", 0.05)   # Halt at -5% daily
MAX_PORTFOLIO_HEAT_PCT = _env_float("MAX_PORTFOLIO_HEAT_PCT", 0.20)   # Max 20% open exposure

# Truncation limit for AI analysis text stored in the database.
MAX_ANALYSIS_LENGTH = 2000

# --- Initialise Clients ---
trading_client = (
    TradingClient(ALPACA_API_KEY, ALPACA_SECRET, paper=True)
    if ALPACA_API_KEY and ALPACA_SECRET
    else None
)
data_client = (
    StockHistoricalDataClient(ALPACA_API_KEY, ALPACA_SECRET)
    if _has_data_client and ALPACA_API_KEY and ALPACA_SECRET
    else None
)
llm = Ollama(
    model=_vault.get("OLLAMA_MODEL", "llama3.2:3b") or "llama3.2:3b",
    base_url=_vault.get("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434",
)


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
    if _env_bool("SKIP_OLLAMA_HEALTHCHECK", default=False):
        logger.warning("Skipping Ollama health check because SKIP_OLLAMA_HEALTHCHECK=true")
        return

    ollama_url = _vault.get("OLLAMA_BASE_URL", "http://localhost:11434") or "http://localhost:11434"
    try:
        resp = requests.get(f"{ollama_url}/api/tags", timeout=5)
        resp.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Ollama is not reachable at {ollama_url}. "
            "Start it with: ollama serve\n"
            f"  Detail: {exc}"
        ) from exc
