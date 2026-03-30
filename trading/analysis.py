import logging
import sqlite3

from langchain_core.prompts import PromptTemplate

from config import llm, DAILY_MAX_TRADES

logger = logging.getLogger(__name__)


def pre_trade_analysis(
    ticker: str,
    sentiment: str,
    technical: str,
    geopolitics: str,
    fed_rate: str,
    fear_level: str,
    stop_pct: float,
    take_pct: float,
    conn: sqlite3.Connection | None = None,
    rsi_signal: str = "NEUTRAL",
    macd_signal: str = "NEUTRAL",
    bbands_signal: str = "NEUTRAL",
    volume_signal: str = "NORMAL",
    earnings_flag: str = "UNKNOWN",
    momentum_score: float = 0.0,
) -> tuple:
    """Synthesise all signals through Ollama and decide trade direction.

    The bot explicitly declares how much it is willing to lose and profit
    before each trade.  When a database connection is supplied, recent
    stop-loss and EOD lessons are injected into the prompt so the bot can
    learn from past mistakes.

    Returns
    -------
    (direction: str, should_trade: bool, reason: str, full_analysis: str)
    direction is one of "BUY", "SELL", or "HOLD".
    """
    # Build a lessons section from recent reflections (if available)
    lessons_section = ""
    if conn is not None:
        try:
            from reflection.engine import get_recent_lessons
            lessons = get_recent_lessons(conn, ticker=ticker, n=3)
            if lessons:
                lessons_text = "\n".join(f"  - {l}" for l in lessons)
                lessons_section = f"\nPAST LESSONS (from previous stop-losses / EOD reviews):\n{lessons_text}\n"
        except Exception as exc:
            logger.debug("Could not inject lessons for %s: %s", ticker, exc)

    template = """
You are a hedge fund portfolio manager with billions under management.
Analyze this potential trade for {ticker}:

MACRO & SENTIMENT SIGNALS:
- News Sentiment:           {sentiment}
- Geopolitical Risk:        {geopolitics}
- Federal Reserve Stance:   {fed_rate}
- Market Fear / VIX Level:  {fear_level}

TECHNICAL ANALYSIS SIGNALS:
- RSI (14-period):          {rsi_signal}      [< 30 = oversold BULLISH  |  > 70 = overbought BEARISH]
- MACD (12,26,9):           {macd_signal}     [BULLISH = above signal line  |  BEARISH = below signal]
- Bollinger Bands (20,2σ):  {bbands_signal}   [BULLISH = price below lower  |  BEARISH = price above upper]
- Volume:                   {volume_signal}   [SPIKE_UP/SPIKE_DOWN = 2× avg volume  |  NORMAL = baseline]
- Earnings Proximity:       {earnings_flag}   [NEAR = event risk in ~7 days  |  SAFE = no imminent event]
- Momentum Score:           {momentum_score:+.1f}  [−3 = strongly bearish  |  0 = neutral  |  +3 = strongly bullish]
- Technical Summary:        {technical}

RISK PARAMETERS:
- Maximum loss I am willing to accept: {stop_pct}% (stop loss)
- Target profit I want to capture:     {take_pct}% (take profit)
{lessons_section}
Think step by step:
1. What is the overall macro environment saying?
2. Do the technical indicators (RSI, MACD, BBands, Volume) confirm the macro view?
3. Does the momentum score ({momentum_score:+.1f}) support going LONG or SHORT?
4. If EARNINGS_NEAR, should I reduce conviction or avoid the trade entirely?
5. Should I go LONG (BUY, profit if price rises) or SHORT (SELL, profit if price falls)?
6. Is the risk-reward ratio favourable given a {stop_pct}% stop and {take_pct}% target?
7. Do any PAST LESSONS change my assessment?

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
            "rsi_signal", "macd_signal", "bbands_signal",
            "volume_signal", "earnings_flag", "momentum_score",
            "lessons_section",
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
            "rsi_signal": rsi_signal,
            "macd_signal": macd_signal,
            "bbands_signal": bbands_signal,
            "volume_signal": volume_signal,
            "earnings_flag": earnings_flag,
            "momentum_score": momentum_score,
            "lessons_section": lessons_section,
        }).strip()
    except Exception as exc:
        logger.error("LLM pre-trade analysis failed for %s: %s.", ticker, exc)
        return "HOLD", False, f"LLM error: {exc}", ""

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
