import logging

import requests
from langchain_core.prompts import PromptTemplate

from config import NEWS_API_KEY, llm

logger = logging.getLogger(__name__)


def _fetch_headlines(query: str, page_size: int = 5) -> list[str]:
    """Fetch news headlines from NewsAPI for the given query."""
    if not NEWS_API_KEY:
        logger.debug("NEWS_API_KEY is not set – skipping headline fetch for: %s.", query)
        return []
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
    except Exception as exc:
        logger.warning("Failed to fetch headlines for query '%s': %s.", query, exc)
        return []


def analyze_sentiment(ticker: str) -> str:
    """Assess stock-specific sentiment from recent news headlines."""
    headlines = _fetch_headlines(ticker, page_size=5)
    if not headlines:
        logger.info("No headlines found for %s – returning NEUTRAL sentiment.", ticker)
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
    except Exception as exc:
        logger.warning("LLM sentiment analysis failed for %s: %s. Returning NEUTRAL.", ticker, exc)
        return "NEUTRAL"
    for word in ("BULLISH", "BEARISH", "NEUTRAL"):
        if word in result:
            return word
    return "NEUTRAL"
