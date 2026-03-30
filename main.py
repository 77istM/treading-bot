import logging
import signal
import time
import types
from datetime import date, datetime, timezone

from config import (
    TICKERS,
    DAILY_MAX_TRADES,
    STOP_LOSS_PCT,
    TAKE_PROFIT_PCT,
    MAX_POSITION_PCT,
    LOOP_INTERVAL_SECONDS,
    _validate_credentials,
    _check_ollama_health,
    trading_client,
)
from db.schema import init_db
from db.queries import read_setting, get_daily_trade_count
from signals.sentiment import analyze_sentiment
from signals.technical import get_technical_signals
from signals.earnings import get_earnings_flag
from signals.macro import analyze_geopolitics, analyze_fed_rate, analyze_market_fear
from trading.analysis import assess_risk
from trading.execution import execute_trade
from trading.monitor import monitor_positions
from trading.strategies import StrategyContext, StrategySelector, detect_market_regime
from reflection.engine import (
    reflect_on_stop_loss,
    reflect_on_trade,
    run_end_of_day_reflection,
    eod_already_run_today,
)
from risk.controller import PortfolioRiskController
from hardening.alerts import get_notifier

logger = logging.getLogger(__name__)
notifier = get_notifier()

# ---------------------------------------------------------------------------
# Graceful-shutdown support
# ---------------------------------------------------------------------------
_RUNNING = True


def _handle_signal(signum: int, frame: types.FrameType | None) -> None:
    global _RUNNING
    logger.info("Received signal %s — shutting down after the current cycle.", signum)
    _RUNNING = False


signal.signal(signal.SIGTERM, _handle_signal)
signal.signal(signal.SIGINT, _handle_signal)


# ---------------------------------------------------------------------------
# Market-hours helpers
# ---------------------------------------------------------------------------

def _market_is_open() -> bool:
    """Return True if Alpaca reports the US market is currently open."""
    try:
        clock = trading_client.get_clock()
        return bool(clock.is_open)
    except Exception as exc:
        logger.warning("Could not check market clock: %s. Assuming open.", exc)
        return True


def _is_eod_window() -> bool:
    """Return True if the current UTC time is past 21:00 (≈ 4 PM US/Eastern + buffer).

    21:00 UTC ≈ 16:00–17:00 US Eastern (accounts for DST).
    """
    return datetime.now(timezone.utc).hour >= 21


# ---------------------------------------------------------------------------
# Single trading cycle
# ---------------------------------------------------------------------------

def _run_trading_cycle(conn, risk_ctrl: PortfolioRiskController) -> None:
    """Execute one full monitor → analyse → trade cycle."""
    stop_pct = read_setting(conn, "stop_loss_pct", STOP_LOSS_PCT)
    take_pct = read_setting(conn, "take_profit_pct", TAKE_PROFIT_PCT)

    # Step 1 — Monitor open positions; collect any stop-loss / take-profit events
    logger.info("[Position Monitor] Checking open positions...")
    closed_events = monitor_positions(conn, stop_pct, take_pct)

    # Trigger real-time reflection for every stop-loss event
    for event in closed_events:
        if event.get("is_stop_loss"):
            reflect_on_stop_loss(
                conn,
                ticker=event["ticker"],
                trade_id=event.get("trade_id"),
                signals=event.get("signals", {}),
                pnl=event.get("pnl", 0.0),
                stop_reason=event["reason"],
            )
            notifier.send(
                level="warning",
                title="Stop Loss Triggered",
                message=f"{event['ticker']} closed by stop-loss.",
                details={
                    "trade_id": event.get("trade_id"),
                    "reason": event.get("reason"),
                    "pnl": event.get("pnl", 0.0),
                },
            )

    # Step 2 — Portfolio risk gate
    can_trade, halt_reason = risk_ctrl.can_trade()
    if not can_trade:
        logger.warning("[RISK GATE] Skipping new trades this cycle: %s", halt_reason)
        notifier.send(
            level="warning",
            title="Risk Gate Halt",
            message="New trades blocked by portfolio risk controller.",
            details={"reason": halt_reason},
        )
        return

    # Step 3 — Daily trade-count gate
    daily_count = get_daily_trade_count(conn)
    if daily_count >= DAILY_MAX_TRADES:
        logger.warning("Daily maximum of %d trades already reached. Skipping new trades.", DAILY_MAX_TRADES)
        return

    # Step 4 — Market-wide signals (computed once, shared across all tickers)
    logger.info("[Market Analysis] Gathering market-wide signals...")
    geopolitics = analyze_geopolitics()
    fed_rate = analyze_fed_rate()
    fear_level = analyze_market_fear()
    regime = detect_market_regime()
    selector = StrategySelector()
    logger.info(
        "Geopolitics: %s | Fed Rate: %s | Market Fear: %s | Regime: %s",
        geopolitics,
        fed_rate,
        fear_level,
        regime,
    )

    # Step 5 — Per-ticker analysis and execution
    for ticker in TICKERS:
        if not _RUNNING:
            break
        daily_count = get_daily_trade_count(conn)
        if daily_count >= DAILY_MAX_TRADES:
            logger.warning("Daily maximum of %d trades reached. Stopping.", DAILY_MAX_TRADES)
            break

        logger.info("[%s] Analysing... (trades today: %d)", ticker, daily_count)

        sentiment = analyze_sentiment(ticker)
        tech = get_technical_signals(ticker)
        earnings = get_earnings_flag(ticker)
        logger.info(
            "  Sentiment: %s | Tech: %s | Earnings: %s",
            sentiment, tech["summary"], earnings,
        )

        signals = {
            "sentiment": sentiment,
            "technical": tech["summary"],
            "rsi": tech["rsi"],
            "macd": tech["macd"],
            "bbands": tech["bbands"],
            "volume": tech["volume"],
            "momentum_score": tech["momentum_score"],
            "earnings": earnings,
            "geopolitics": geopolitics,
            "fed_rate": fed_rate,
            "fear_level": fear_level,
        }

        strategy_context = StrategyContext(
            ticker=ticker,
            sentiment=sentiment,
            technical=tech["summary"],
            rsi=tech["rsi"],
            macd=tech["macd"],
            bbands=tech["bbands"],
            volume=tech["volume"],
            momentum_score=float(tech["momentum_score"]),
            earnings=earnings,
            geopolitics=geopolitics,
            fed_rate=fed_rate,
            fear_level=fear_level,
        )
        strategy_decision = selector.choose(strategy_context, regime)
        strategy_name = strategy_decision.strategy_name
        strategy_regime = strategy_decision.regime
        direction = strategy_decision.direction
        should_trade = strategy_decision.should_trade
        reason = strategy_decision.reason
        full_analysis = (
            f"STRATEGY: {strategy_name}\n"
            f"REGIME: {strategy_regime}\n"
            f"CONFIDENCE: {strategy_decision.confidence}\n"
            f"DECISION: {direction} | TRADE={should_trade}\n"
            f"REASON: {reason}"
        )
        signals["strategy_name"] = strategy_name
        signals["strategy_regime"] = strategy_regime

        logger.info(
            "  Strategy Decision: %s | Regime: %s | Dir: %s | Trade: %s | %s",
            strategy_name,
            strategy_regime,
            direction,
            should_trade,
            reason,
        )

        trade_ok, final_direction, final_reason = assess_risk(
            daily_count, direction, should_trade, reason
        )

        if trade_ok:
            trade_id = execute_trade(
                conn,
                ticker,
                final_direction,
                final_reason,
                full_analysis,
                signals,
                stop_pct,
                take_pct,
                strategy_name=strategy_name,
                strategy_regime=strategy_regime,
            )
            # Post-trade reflection: commit a 24h prediction using the actual fill price
            entry_price = 0.0
            if trade_id is not None:
                try:
                    row = conn.execute(
                        "SELECT price FROM trades WHERE id = ?", (trade_id,)
                    ).fetchone()
                    if row and row[0] is not None:
                        entry_price = float(row[0])
                except Exception:
                    pass
            reflect_on_trade(
                conn,
                ticker=ticker,
                trade_id=trade_id,
                direction=final_direction,
                signals=signals,
                entry_price=entry_price,
                reason=final_reason,
            )
        else:
            logger.info("  → Skipping %s: %s", ticker, final_reason)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main() -> None:
    try:
        _validate_credentials()
        _check_ollama_health()
    except Exception as exc:
        logger.error("Startup validation failed: %s", exc)
        notifier.send(
            level="critical",
            title="Startup Validation Failed",
            message=str(exc),
        )
        return

    try:
        conn = init_db()
    except Exception as exc:
        logger.error("Failed to initialise database: %s", exc)
        notifier.send(
            level="critical",
            title="Database Init Failed",
            message=str(exc),
        )
        return

    stop_pct = read_setting(conn, "stop_loss_pct", STOP_LOSS_PCT)
    take_pct = read_setting(conn, "take_profit_pct", TAKE_PROFIT_PCT)
    daily_count = get_daily_trade_count(conn)

    logger.info("Hedge Fund Bot started — continuous loop mode.")
    logger.info("Trades today: %d/%d", daily_count, DAILY_MAX_TRADES)
    logger.info(
        "Risk params: Stop=%.1f%% | Take=%.1f%% | Ring fence=%.1f%% | Cycle=%ds",
        stop_pct * 100, take_pct * 100, MAX_POSITION_PCT * 100, LOOP_INTERVAL_SECONDS,
    )
    notifier.send(
        level="info",
        title="Bot Started",
        message="Trading bot entered continuous loop mode.",
        details={
            "tickers": len(TICKERS),
            "daily_trade_limit": DAILY_MAX_TRADES,
            "cycle_seconds": LOOP_INTERVAL_SECONDS,
        },
    )

    risk_ctrl = PortfolioRiskController(conn)
    risk_ctrl.record_day_start()

    last_eod_date: date | None = None

    while _RUNNING:
        now_utc = datetime.now(timezone.utc)

        # Reset day-start value at the start of each new calendar day
        if last_eod_date != now_utc.date():
            risk_ctrl.record_day_start()

        # Check if we should run the EOD reflection (after 21:00 UTC, once per day)
        if _is_eod_window() and last_eod_date != now_utc.date():
            if not eod_already_run_today(conn):
                logger.info("[EOD] Running end-of-day reflection...")
                run_end_of_day_reflection(conn)
            last_eod_date = now_utc.date()

        # Only trade during market hours
        if not _market_is_open():
            logger.debug("Market closed — sleeping %ds.", LOOP_INTERVAL_SECONDS)
            time.sleep(LOOP_INTERVAL_SECONDS)
            continue

        try:
            _run_trading_cycle(conn, risk_ctrl)
        except Exception as exc:
            logger.error("Unhandled error in trading cycle: %s", exc, exc_info=True)
            notifier.send(
                level="error",
                title="Trading Cycle Error",
                message=str(exc),
            )

        daily_count = get_daily_trade_count(conn)
        logger.info("Cycle complete. Trades today: %d/%d. Sleeping %ds.",
                    daily_count, DAILY_MAX_TRADES, LOOP_INTERVAL_SECONDS)
        time.sleep(LOOP_INTERVAL_SECONDS)

    logger.info("Bot stopped cleanly.")
    notifier.send(
        level="info",
        title="Bot Stopped",
        message="Trading bot shut down cleanly.",
    )


if __name__ == "__main__":
    main()

