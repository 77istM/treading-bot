import logging

from langchain_core.prompts import PromptTemplate

from config import llm
from signals.sentiment import _fetch_headlines

logger = logging.getLogger(__name__)


def analyze_geopolitics() -> str:
    """Assess global geopolitical risk level from recent news."""
    headlines = _fetch_headlines("geopolitical risk war conflict sanctions trade war", page_size=5)
    if not headlines:
        logger.info("No geopolitics headlines found – returning MEDIUM_RISK.")
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
    except Exception as exc:
        logger.warning("LLM geopolitics analysis failed: %s. Returning MEDIUM_RISK.", exc)
        return "MEDIUM_RISK"
    for level in ("LOW_RISK", "MEDIUM_RISK", "HIGH_RISK"):
        if level in result:
            return level
    return "MEDIUM_RISK"


def analyze_fed_rate() -> str:
    """Assess Federal Reserve rate policy stance from recent news."""
    headlines = _fetch_headlines("Federal Reserve interest rate inflation FOMC", page_size=5)
    if not headlines:
        logger.info("No Fed rate headlines found – returning NEUTRAL.")
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
    except Exception as exc:
        logger.warning("LLM Fed rate analysis failed: %s. Returning NEUTRAL.", exc)
        return "NEUTRAL"
    for stance in ("HAWKISH", "DOVISH", "NEUTRAL"):
        if stance in result:
            return stance
    return "NEUTRAL"


def analyze_market_fear() -> str:
    """Assess market fear/VIX level from recent headlines."""
    headlines = _fetch_headlines("VIX volatility index market fear S&P 500 crash rally", page_size=5)
    if not headlines:
        logger.info("No market fear headlines found – returning MEDIUM.")
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
    except Exception as exc:
        logger.warning("LLM market fear analysis failed: %s. Returning MEDIUM.", exc)
        return "MEDIUM"
    for level in ("HIGH", "MEDIUM", "LOW"):
        if level in result:
            return level
    return "MEDIUM"
