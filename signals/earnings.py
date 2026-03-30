"""Earnings proximity detection via NewsAPI headline pattern matching.

The goal is to flag tickers where earnings are likely within the next ~7 days
so the pre-trade LLM can apply extra caution (high uncertainty event risk).

This module requires no additional Python dependencies beyond what is already
in requirements.txt (requests + NEWS_API_KEY env var).
"""
import logging
import re
from datetime import datetime, timedelta, timezone

import requests

from config import NEWS_API_KEY

logger = logging.getLogger(__name__)

# Patterns that suggest UPCOMING earnings (not already-reported results)
_UPCOMING_PATTERNS = [
    r"\bearnings\b.{0,60}\b(preview|expect|schedul|due|upcoming|ahead|next week|this week|tomorrow)\b",
    r"\b(preview|expect|schedul|due|upcoming|ahead|next week|this week|tomorrow)\b.{0,60}\bearnings\b",
    r"\b(due|expected|scheduled)\b.{0,30}\b(report|results|earnings)\b",
    r"\breports?\s+earnings?\b.{0,30}\b(on|next|this|tomorrow|monday|tuesday|wednesday|thursday|friday)\b",
    r"\bearnings?\s+(call|release|date)\b.{0,20}\b(scheduled|upcoming|next|this)\b",
    r"\b(q[1-4]|first|second|third|fourth)\s+quarter\b.{0,30}\b(report|results|earnings)\b.{0,30}\b(due|expected|upcoming|next)\b",
]
_UPCOMING_RE = [re.compile(p, re.IGNORECASE) for p in _UPCOMING_PATTERNS]

# Patterns that indicate earnings ALREADY HAPPENED (so not an upcoming risk)
_PAST_PATTERNS = [
    r"\b(beat|missed|miss|topped|exceeded|fell short)\b.{0,30}\b(expectations?|estimates?|consensus)\b",
    r"\b(reported|announced|posted|delivered)\b.{0,30}\b(earnings|results|profit|loss|EPS)\b",
]
_PAST_RE = [re.compile(p, re.IGNORECASE) for p in _PAST_PATTERNS]


def get_earnings_flag(ticker: str) -> str:
    """Detect whether earnings are imminent for *ticker*.

    Searches the most recent NewsAPI headlines for the ticker and looks for
    language suggesting earnings are due within ~7 days.

    Returns
    -------
    "NEAR"    — earnings likely within ~7 days (elevated event risk)
    "SAFE"    — no imminent earnings signal found
    "UNKNOWN" — NEWS_API_KEY missing or fetch failed
    """
    if not NEWS_API_KEY:
        return "UNKNOWN"

    # Only look at articles from the last 4 days (recent news only)
    from_date = (datetime.now(timezone.utc) - timedelta(days=4)).strftime("%Y-%m-%d")
    query = f"{ticker} earnings quarterly results"
    url = (
        f"https://newsapi.org/v2/everything"
        f"?q={requests.utils.quote(query)}"
        f"&apiKey={NEWS_API_KEY}"
        f"&pageSize=10"
        f"&language=en"
        f"&sortBy=publishedAt"
        f"&from={from_date}"
    )
    try:
        resp = requests.get(url, timeout=10).json()
        articles = resp.get("articles", [])[:10]
        headlines = [a.get("title", "") for a in articles if a.get("title")]
    except Exception as exc:
        logger.warning("Earnings flag fetch failed for %s: %s", ticker, exc)
        return "UNKNOWN"

    if not headlines:
        return "SAFE"

    combined = " | ".join(headlines)

    # First check: if the dominant signal is "already reported" → SAFE
    past_hits = sum(1 for pat in _PAST_RE if pat.search(combined))
    upcoming_hits = sum(1 for pat in _UPCOMING_RE if pat.search(combined))

    if upcoming_hits > past_hits:
        logger.info("[EARNINGS] %s: upcoming earnings detected (score %d vs %d).",
                    ticker, upcoming_hits, past_hits)
        return "NEAR"

    return "SAFE"
