"""Post-trade reflection engine.

Three reflection modes are supported:
  1. stop_loss   — triggered immediately whenever a stop-loss fires
  2. post_trade  — triggered after each executed trade (stores a prediction)
  3. end_of_day  — triggered once after market close; reviews the full day

Lessons are stored in the `reflections` table and injected into future
pre-trade prompts so the bot can learn from its mistakes.
"""
import logging
import sqlite3
from datetime import date, datetime, timezone

from langchain_core.prompts import PromptTemplate

from config import llm, MAX_ANALYSIS_LENGTH

logger = logging.getLogger(__name__)

# Short prediction / lesson snippet max length (stored separately from full analysis text)
MAX_LESSON_LENGTH = 200


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _invoke_llm(prompt_template: str, variables: dict) -> str:
    """Invoke Ollama with the given template and return stripped text."""
    try:
        chain = PromptTemplate(template=prompt_template, input_variables=list(variables.keys())) | llm
        return chain.invoke(variables).strip()
    except Exception as exc:
        logger.warning("Reflection LLM call failed: %s", exc)
        return ""


def _store_reflection(
    conn: sqlite3.Connection,
    reflection_type: str,
    lesson: str,
    raw_analysis: str,
    trade_id: int | None = None,
    ticker: str | None = None,
    outcome: str | None = None,
    pnl: float | None = None,
) -> None:
    """Persist a reflection record to the database."""
    try:
        conn.execute(
            """INSERT INTO reflections
               (reflection_type, trade_id, ticker, outcome, pnl, lesson, raw_analysis)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                reflection_type,
                trade_id,
                ticker,
                outcome,
                pnl,
                lesson[:MAX_LESSON_LENGTH] if lesson else "",
                raw_analysis[:MAX_ANALYSIS_LENGTH] if raw_analysis else "",
            ),
        )
        conn.commit()
    except Exception as exc:
        logger.warning("Failed to store reflection: %s", exc)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def reflect_on_stop_loss(
    conn: sqlite3.Connection,
    ticker: str,
    trade_id: int | None,
    signals: dict,
    pnl: float,
    stop_reason: str,
) -> str:
    """Reflect immediately after a stop-loss fires.

    The LLM analyses what went wrong and generates a lesson for future trades.
    Returns the lesson string (empty string on failure).
    """
    template = """
You are a post-trade risk analyst for a quantitative hedge fund.
A position in {ticker} was just stopped out with a loss of ${loss_magnitude:.2f}.

ORIGINAL TRADE SIGNALS:
- Sentiment:        {sentiment}
- Technical (RSI):  {technical}
- Geopolitical:     {geopolitics}
- Fed Rate Stance:  {fed_rate}
- Market Fear:      {fear_level}

STOP-LOSS REASON: {stop_reason}

Conduct a ruthless post-mortem:
1. Which signal(s) were misleading or wrong?
2. Were there warning signs we ignored?
3. What rule should the bot follow next time to avoid this mistake?

Respond in EXACTLY this format:
FAILED_SIGNALS: [list the misleading signals]
ROOT_CAUSE: [one sentence]
LESSON: [one concrete, actionable rule for future trades]
"""
    analysis = _invoke_llm(
        template,
        {
            "ticker": ticker,
            "loss_magnitude": abs(pnl),
            "sentiment": signals.get("sentiment", "UNKNOWN"),
            "technical": signals.get("technical", "UNKNOWN"),
            "geopolitics": signals.get("geopolitics", "UNKNOWN"),
            "fed_rate": signals.get("fed_rate", "UNKNOWN"),
            "fear_level": signals.get("fear_level", "UNKNOWN"),
            "stop_reason": stop_reason,
        },
    )

    lesson = ""
    for line in analysis.splitlines():
        if line.upper().startswith("LESSON:"):
            lesson = line.split(":", 1)[1].strip()
            break
    if not lesson and analysis:
        lesson = analysis[:MAX_LESSON_LENGTH]

    logger.warning("[REFLECTION][STOP-LOSS] %s: %s", ticker, lesson)
    _store_reflection(
        conn,
        reflection_type="stop_loss",
        lesson=lesson,
        raw_analysis=analysis,
        trade_id=trade_id,
        ticker=ticker,
        outcome="stop_loss",
        pnl=pnl,
    )
    return lesson


def reflect_on_trade(
    conn: sqlite3.Connection,
    ticker: str,
    trade_id: int | None,
    direction: str,
    signals: dict,
    entry_price: float,
    reason: str,
) -> str:
    """Store a short-term prediction immediately after a trade is placed.

    This is a lightweight «commit to a prediction» step. The EOD reflection
    compares these predictions to actual outcomes.
    Returns the prediction string (empty string on failure).
    """
    template = """
You are a quantitative trader who just opened a {direction} position in {ticker} at ${entry_price:.2f}.

SIGNALS USED:
- Sentiment:        {sentiment}
- Technical (RSI):  {technical}
- Geopolitical:     {geopolitics}
- Fed Rate Stance:  {fed_rate}
- Market Fear:      {fear_level}

REASON FOR TRADE: {reason}

Commit to a 24-hour prediction:
1. Expected price direction in the next 24h?
2. Key risk that could invalidate this trade?
3. Confidence level?

Respond in EXACTLY this format:
PREDICTION: [UP or DOWN or FLAT]
KEY_RISK: [one sentence]
CONFIDENCE: [HIGH or MEDIUM or LOW]
"""
    analysis = _invoke_llm(
        template,
        {
            "ticker": ticker,
            "direction": direction,
            "entry_price": entry_price,
            "sentiment": signals.get("sentiment", "UNKNOWN"),
            "technical": signals.get("technical", "UNKNOWN"),
            "geopolitics": signals.get("geopolitics", "UNKNOWN"),
            "fed_rate": signals.get("fed_rate", "UNKNOWN"),
            "fear_level": signals.get("fear_level", "UNKNOWN"),
            "reason": reason,
        },
    )

    prediction = analysis[:MAX_LESSON_LENGTH] if analysis else "No prediction generated."
    logger.info("[REFLECTION][POST-TRADE] %s prediction: %s", ticker, prediction[:80])
    _store_reflection(
        conn,
        reflection_type="post_trade",
        lesson=prediction,
        raw_analysis=analysis,
        trade_id=trade_id,
        ticker=ticker,
        outcome="open",
        pnl=0.0,
    )
    return prediction


def run_end_of_day_reflection(conn: sqlite3.Connection) -> str:
    """Summarise the full trading day and extract lessons for tomorrow.

    Should be called once per day, after market close.
    Returns a summary string.
    """
    today = date.today().isoformat()
    try:
        cursor = conn.cursor()
        cursor.execute(
            """SELECT ticker, side, qty, price, realized_pnl, reason,
                      sentiment, technical_signal, geopolitics, fed_sentiment, fear_level
               FROM trades WHERE DATE(created_at) = ? ORDER BY id ASC""",
            (today,),
        )
        rows = cursor.fetchall()
    except Exception as exc:
        logger.warning("EOD reflection: could not fetch today's trades: %s", exc)
        return ""

    if not rows:
        logger.info("[EOD REFLECTION] No trades today (%s). Skipping.", today)
        return ""

    trades_summary = "\n".join(
        f"  {side} {qty} {ticker} @ {price} | PnL={pnl:.2f} | Reason: {reason} | "
        f"Signals: S={sent}, T={tech}, G={geo}, F={fed}, V={fear}"
        for ticker, side, qty, price, pnl, reason, sent, tech, geo, fed, fear in rows
    )

    total_pnl = sum(float(r[4] or 0) for r in rows)
    wins = sum(1 for r in rows if float(r[4] or 0) > 0)
    losses = sum(1 for r in rows if float(r[4] or 0) < 0)

    template = """
You are the chief risk officer of a quantitative hedge fund reviewing today's trading activity.

DATE: {today}
TOTAL TRADES: {n_trades}
WINS: {wins} | LOSSES: {losses}
NET P&L: ${total_pnl:.2f}

TRADE LOG:
{trades_summary}

Conduct a rigorous end-of-day review:
1. What worked well today and why?
2. What went wrong and what was the root cause?
3. Were the signals (sentiment/technical/macro) reliable today?
4. What are the top 3 rules the bot should apply TOMORROW?

Respond in EXACTLY this format:
WHAT_WORKED: [one sentence]
WHAT_FAILED: [one sentence]
SIGNAL_ACCURACY: [assessment of which signals were reliable]
RULE_1: [actionable rule for tomorrow]
RULE_2: [actionable rule for tomorrow]
RULE_3: [actionable rule for tomorrow]
SUMMARY: [2-sentence overall assessment]
"""
    analysis = _invoke_llm(
        template,
        {
            "today": today,
            "n_trades": len(rows),
            "wins": wins,
            "losses": losses,
            "total_pnl": total_pnl,
            "trades_summary": trades_summary,
        },
    )

    # Extract the multi-line lesson
    lesson_lines = []
    for line in analysis.splitlines():
        u = line.upper()
        if any(u.startswith(k) for k in ("RULE_1:", "RULE_2:", "RULE_3:", "SUMMARY:")):
            lesson_lines.append(line.strip())
    lesson = " | ".join(lesson_lines) if lesson_lines else analysis[:300]

    logger.info("[EOD REFLECTION] %s: %s", today, lesson[:120])
    _store_reflection(
        conn,
        reflection_type="end_of_day",
        lesson=lesson,
        raw_analysis=analysis,
        ticker=None,
        outcome=f"pnl={total_pnl:.2f}",
        pnl=total_pnl,
    )
    return lesson


def get_recent_lessons(
    conn: sqlite3.Connection,
    ticker: str | None = None,
    n: int = 3,
) -> list[str]:
    """Return recent lessons to inject into future pre-trade prompts.

    Retrieves the most recent stop-loss and EOD lessons, optionally filtered
    by ticker.
    """
    try:
        cursor = conn.cursor()
        if ticker:
            cursor.execute(
                """SELECT lesson FROM reflections
                   WHERE (ticker = ? OR ticker IS NULL)
                     AND lesson != ''
                     AND reflection_type IN ('stop_loss', 'end_of_day')
                   ORDER BY id DESC LIMIT ?""",
                (ticker, n),
            )
        else:
            cursor.execute(
                """SELECT lesson FROM reflections
                   WHERE lesson != ''
                     AND reflection_type IN ('stop_loss', 'end_of_day')
                   ORDER BY id DESC LIMIT ?""",
                (n,),
            )
        return [row[0] for row in cursor.fetchall()]
    except Exception as exc:
        logger.warning("Could not fetch recent lessons: %s", exc)
        return []


def eod_already_run_today(conn: sqlite3.Connection) -> bool:
    """Return True if an end-of-day reflection has already been stored today."""
    today = date.today().isoformat()
    try:
        cursor = conn.cursor()
        cursor.execute(
            "SELECT COUNT(*) FROM reflections WHERE reflection_type = 'end_of_day' AND DATE(created_at) = ?",
            (today,),
        )
        return (cursor.fetchone() or (0,))[0] > 0
    except Exception:
        return False
