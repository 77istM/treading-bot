import logging
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import numpy as np

from config import StockBarsRequest, TimeFrame, data_client

logger = logging.getLogger(__name__)


@dataclass
class StrategyContext:
    ticker: str
    sentiment: str
    technical: str
    rsi: str
    macd: str
    bbands: str
    volume: str
    momentum_score: float
    earnings: str
    geopolitics: str
    fed_rate: str
    fear_level: str


@dataclass
class StrategyDecision:
    strategy_name: str
    regime: str
    direction: str
    should_trade: bool
    reason: str
    confidence: str = "LOW"


class BaseStrategy:
    name = "base"

    def evaluate(self, ctx: StrategyContext, regime: str) -> StrategyDecision:
        raise NotImplementedError


def _fetch_closes(symbols: list[str], lookback_days: int = 120) -> dict[str, np.ndarray]:
    if data_client is None or StockBarsRequest is None or TimeFrame is None:
        return {}

    try:
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=lookback_days)
        req = StockBarsRequest(
            symbol_or_symbols=symbols,
            timeframe=TimeFrame.Day,
            start=start,
            end=end,
        )
        bars = data_client.get_stock_bars(req)
        df = bars.df
        if df.empty:
            return {}

        out: dict[str, np.ndarray] = {}
        if hasattr(df.index, "levels"):
            for sym in symbols:
                try:
                    sdf = df.xs(sym, level=0)
                    out[sym] = sdf["close"].values.astype(float)
                except Exception:
                    continue
        else:
            out[symbols[0]] = df["close"].values.astype(float)
        return out
    except Exception as exc:
        logger.debug("Could not fetch closes for strategy engine: %s", exc)
        return {}


def detect_market_regime() -> str:
    """Classify market as TRENDING or RANGING from SPY daily closes."""
    closes_map = _fetch_closes(["SPY"], lookback_days=120)
    spy = closes_map.get("SPY")
    if spy is None or len(spy) < 40:
        return "RANGING"

    window = spy[-30:]
    x = np.arange(len(window), dtype=float)
    slope, _ = np.polyfit(x, window, 1)
    vol = float(np.std(window))
    if vol <= 0:
        return "RANGING"

    trend_strength = abs(slope * len(window) / vol)
    return "TRENDING" if trend_strength >= 1.1 else "RANGING"


class MomentumStrategy(BaseStrategy):
    name = "momentum"

    def evaluate(self, ctx: StrategyContext, regime: str) -> StrategyDecision:
        score = float(ctx.momentum_score)
        if ctx.macd == "BULLISH":
            score += 0.5
        elif ctx.macd == "BEARISH":
            score -= 0.5

        if ctx.volume == "SPIKE_UP":
            score += 0.25
        elif ctx.volume == "SPIKE_DOWN":
            score -= 0.25

        if ctx.earnings == "NEAR":
            score *= 0.6

        if score >= 1.25:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="BUY",
                should_trade=True,
                reason=f"Momentum long setup (score={score:+.2f}, MACD={ctx.macd}, VOL={ctx.volume}).",
                confidence="HIGH" if score >= 2.0 else "MEDIUM",
            )
        if score <= -1.25:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="SELL",
                should_trade=True,
                reason=f"Momentum short setup (score={score:+.2f}, MACD={ctx.macd}, VOL={ctx.volume}).",
                confidence="HIGH" if score <= -2.0 else "MEDIUM",
            )

        return StrategyDecision(
            strategy_name=self.name,
            regime=regime,
            direction="HOLD",
            should_trade=False,
            reason=f"Momentum score weak ({score:+.2f}).",
            confidence="LOW",
        )


class MeanReversionStrategy(BaseStrategy):
    name = "mean_reversion"

    def evaluate(self, ctx: StrategyContext, regime: str) -> StrategyDecision:
        bullish = 0
        bearish = 0

        if ctx.rsi == "BULLISH":
            bullish += 1
        elif ctx.rsi == "BEARISH":
            bearish += 1

        if ctx.bbands == "BULLISH":
            bullish += 1
        elif ctx.bbands == "BEARISH":
            bearish += 1

        if ctx.sentiment == "BULLISH":
            bullish += 0.5
        elif ctx.sentiment == "BEARISH":
            bearish += 0.5

        if ctx.earnings == "NEAR":
            bullish *= 0.7
            bearish *= 0.7

        if bullish >= 1.5 and bullish > bearish:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="BUY",
                should_trade=True,
                reason=f"Mean-reversion long: RSI/BBands oversold cluster ({bullish:.1f}).",
                confidence="MEDIUM",
            )
        if bearish >= 1.5 and bearish > bullish:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="SELL",
                should_trade=True,
                reason=f"Mean-reversion short: RSI/BBands overbought cluster ({bearish:.1f}).",
                confidence="MEDIUM",
            )

        return StrategyDecision(
            strategy_name=self.name,
            regime=regime,
            direction="HOLD",
            should_trade=False,
            reason="No strong mean-reversion edge.",
            confidence="LOW",
        )


class PairsTradingStrategy(BaseStrategy):
    name = "pairs_trading"

    _pairs_map = {
        "SPY": "QQQ",
        "QQQ": "SPY",
        "XLF": "JPM",
        "JPM": "XLF",
        "XLE": "XOM",
        "XOM": "XLE",
        "GLD": "TLT",
        "TLT": "GLD",
        "EFA": "EEM",
        "EEM": "EFA",
    }

    def evaluate(self, ctx: StrategyContext, regime: str) -> StrategyDecision:
        pair = self._pairs_map.get(ctx.ticker)
        if not pair:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="HOLD",
                should_trade=False,
                reason="No configured pair for ticker.",
                confidence="LOW",
            )

        closes_map = _fetch_closes([ctx.ticker, pair], lookback_days=180)
        c1 = closes_map.get(ctx.ticker)
        c2 = closes_map.get(pair)
        if c1 is None or c2 is None or len(c1) < 60 or len(c2) < 60:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="HOLD",
                should_trade=False,
                reason=f"Insufficient pair history vs {pair}.",
                confidence="LOW",
            )

        n = min(len(c1), len(c2), 120)
        x = np.log(c2[-n:])
        y = np.log(c1[-n:])
        var_x = float(np.var(x))
        if var_x <= 0:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="HOLD",
                should_trade=False,
                reason=f"Flat pair variance vs {pair}.",
                confidence="LOW",
            )

        beta = float(np.cov(y, x)[0, 1] / var_x)
        spread = y - beta * x
        spread_mean = float(np.mean(spread[:-1]))
        spread_std = float(np.std(spread[:-1]))
        if spread_std <= 0:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="HOLD",
                should_trade=False,
                reason=f"Pair spread has zero dispersion vs {pair}.",
                confidence="LOW",
            )

        z = float((spread[-1] - spread_mean) / spread_std)
        # Positive z-score means ticker is rich vs pair -> short ticker.
        if z >= 2.0:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="SELL",
                should_trade=True,
                reason=f"Pairs short vs {pair}: spread z-score={z:.2f}.",
                confidence="HIGH" if z >= 2.5 else "MEDIUM",
            )
        if z <= -2.0:
            return StrategyDecision(
                strategy_name=self.name,
                regime=regime,
                direction="BUY",
                should_trade=True,
                reason=f"Pairs long vs {pair}: spread z-score={z:.2f}.",
                confidence="HIGH" if z <= -2.5 else "MEDIUM",
            )

        return StrategyDecision(
            strategy_name=self.name,
            regime=regime,
            direction="HOLD",
            should_trade=False,
            reason=f"Pair spread near fair value vs {pair} (z={z:.2f}).",
            confidence="LOW",
        )


class StrategySelector:
    """Select strategy decisions based on market regime."""

    def __init__(self) -> None:
        self.momentum = MomentumStrategy()
        self.mean_reversion = MeanReversionStrategy()
        self.pairs = PairsTradingStrategy()

    def choose(
        self,
        ctx: StrategyContext,
        regime: str,
        allowed_directions: set[str] | None = None,
    ) -> StrategyDecision:
        allowed = {d.upper() for d in (allowed_directions or {"BUY", "SELL"})}

        if regime == "TRENDING":
            order = [self.momentum, self.pairs, self.mean_reversion]
        else:
            order = [self.pairs, self.mean_reversion, self.momentum]

        fallback = None
        blocked_trade: StrategyDecision | None = None
        for strat in order:
            decision = strat.evaluate(ctx, regime)
            if fallback is None:
                fallback = decision
            if decision.should_trade and decision.direction in allowed:
                return decision
            if decision.should_trade and blocked_trade is None:
                blocked_trade = decision

        if blocked_trade is not None:
            allowed_txt = ",".join(sorted(allowed))
            return StrategyDecision(
                strategy_name=blocked_trade.strategy_name,
                regime=blocked_trade.regime,
                direction="HOLD",
                should_trade=False,
                reason=(
                    f"Trade blocked by direction filter: {blocked_trade.direction} not in "
                    f"[{allowed_txt}]. Original: {blocked_trade.reason}"
                ),
                confidence="LOW",
            )

        if fallback is not None:
            return fallback

        return StrategyDecision(
            strategy_name="none",
            regime=regime,
            direction="HOLD",
            should_trade=False,
            reason="No strategy available.",
            confidence="LOW",
        )
