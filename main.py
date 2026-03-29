import logging

from config import (
    TICKERS,
    DAILY_MAX_TRADES,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_POSITION_PCT,
    _validate_credentials,
    _check_ollama_health,
)
from db.schema import init_db
from db.queries import read_setting, get_daily_trade_count
from signals.sentiment import analyze_sentiment
from signals.technical import get_technical_signal
from signals.macro import analyze_geopolitics, analyze_fed_rate, analyze_market_fear
from trading.analysis import pre_trade_analysis, assess_risk
from trading.execution import execute_trade
from trading.monitor import monitor_positions

logger = logging.getLogger(__name__)


def main() -> None:
    _validate_credentials()
    _check_ollama_health()

    try:
        conn = init_db()
    except Exception as exc:
        logger.error("Failed to initialise database: %s", exc)
        return

    # Read configurable risk settings (may be overridden via the dashboard UI)
    stop_pct = read_setting(conn, "stop_loss_pct", STOP_LOSS_PCT)
    take_pct = read_setting(conn, "take_profit_pct", TAKE_PROFIT_PCT)

    daily_count = get_daily_trade_count(conn)
    logger.info("Hedge Fund Bot Initialised. Analysing market...")
    logger.info("Trades today: %d/%d", daily_count, DAILY_MAX_TRADES)
    logger.info(
        "Risk params: Stop=%.1f%% | Take=%.1f%% | Ring fence=%.1f%%",
        stop_pct * 100, take_pct * 100, MAX_POSITION_PCT * 100,
    )

    if daily_count >= DAILY_MAX_TRADES:
        logger.warning("Daily maximum of %d trades already reached. Exiting.", DAILY_MAX_TRADES)
        return

    # Step 1: Monitor open positions for stop loss / take profit
    logger.info("[Position Monitor] Checking open positions...")
    monitor_positions(conn, stop_pct, take_pct)

    # Step 2: Compute market-wide signals once (shared across all tickers)
    logger.info("[Market Analysis] Gathering market-wide signals...")
    geopolitics = analyze_geopolitics()
    fed_rate = analyze_fed_rate()
    fear_level = analyze_market_fear()
    logger.info("Geopolitics: %s | Fed Rate: %s | Market Fear: %s", geopolitics, fed_rate, fear_level)

    # Step 3: Per-ticker analysis and execution
    for ticker in TICKERS:
        daily_count = get_daily_trade_count(conn)
        if daily_count >= DAILY_MAX_TRADES:
            logger.warning("Daily maximum of %d trades reached. Stopping.", DAILY_MAX_TRADES)
            break

        logger.info("[%s] Analysing... (trades today: %d)", ticker, daily_count)

        sentiment = analyze_sentiment(ticker)
        technical = get_technical_signal(ticker)
        logger.info("  Sentiment: %s | Technical: %s", sentiment, technical)

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
        logger.info("  AI Decision: %s | Trade: %s | %s", direction, should_trade, reason)

        trade_ok, final_direction, final_reason = assess_risk(
            daily_count, direction, should_trade, reason
        )

        if trade_ok:
            execute_trade(conn, ticker, final_direction, final_reason, full_analysis, signals, stop_pct, take_pct)
        else:
            logger.info("  → Skipping %s: %s", ticker, final_reason)

    daily_count = get_daily_trade_count(conn)
    logger.info("Session complete. Trades today: %d/%d.", daily_count, DAILY_MAX_TRADES)


if __name__ == "__main__":
    main()
