import logging

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
