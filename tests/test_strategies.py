import unittest
from unittest.mock import patch

import numpy as np

from trading.strategies import (
    MeanReversionStrategy,
    MomentumStrategy,
    PairsTradingStrategy,
    StrategyContext,
    StrategySelector,
    detect_market_regime,
)


class TestPhase5Strategies(unittest.TestCase):
    def _ctx(self, **overrides):
        base = {
            "ticker": "SPY",
            "sentiment": "NEUTRAL",
            "technical": "NEUTRAL",
            "rsi": "NEUTRAL",
            "macd": "NEUTRAL",
            "bbands": "NEUTRAL",
            "volume": "NORMAL",
            "momentum_score": 0.0,
            "earnings": "SAFE",
            "geopolitics": "LOW_RISK",
            "fed_rate": "NEUTRAL",
            "fear_level": "LOW",
        }
        base.update(overrides)
        return StrategyContext(**base)

    def test_momentum_strategy_buy_signal(self):
        strategy = MomentumStrategy()
        decision = strategy.evaluate(
            self._ctx(macd="BULLISH", volume="SPIKE_UP", momentum_score=1.0),
            regime="TRENDING",
        )
        self.assertTrue(decision.should_trade)
        self.assertEqual(decision.direction, "BUY")
        self.assertEqual(decision.strategy_name, "momentum")

    def test_mean_reversion_strategy_sell_signal(self):
        strategy = MeanReversionStrategy()
        decision = strategy.evaluate(
            self._ctx(rsi="BEARISH", bbands="BEARISH", sentiment="BEARISH"),
            regime="RANGING",
        )
        self.assertTrue(decision.should_trade)
        self.assertEqual(decision.direction, "SELL")
        self.assertEqual(decision.strategy_name, "mean_reversion")

    @patch("trading.strategies._fetch_closes")
    def test_pairs_strategy_extreme_spread_sell(self, mock_fetch_closes):
        n = 120
        pair = np.linspace(100.0, 120.0, n)
        ticker = pair.copy()
        ticker[-1] = pair[-1] * 1.35

        mock_fetch_closes.return_value = {"SPY": ticker, "QQQ": pair}

        strategy = PairsTradingStrategy()
        decision = strategy.evaluate(self._ctx(ticker="SPY"), regime="RANGING")

        self.assertTrue(decision.should_trade)
        self.assertEqual(decision.direction, "SELL")
        self.assertEqual(decision.strategy_name, "pairs_trading")

    def test_strategy_selector_prefers_momentum_in_trending(self):
        selector = StrategySelector()
        decision = selector.choose(
            self._ctx(macd="BULLISH", volume="SPIKE_UP", momentum_score=1.0),
            regime="TRENDING",
        )
        self.assertEqual(decision.strategy_name, "momentum")
        self.assertEqual(decision.direction, "BUY")

    @patch("trading.strategies._fetch_closes")
    def test_strategy_selector_prefers_pairs_in_ranging_when_triggered(self, mock_fetch_closes):
        n = 120
        pair = np.linspace(100.0, 120.0, n)
        ticker = pair.copy()
        ticker[-1] = pair[-1] * 1.35

        mock_fetch_closes.return_value = {"SPY": ticker, "QQQ": pair}

        selector = StrategySelector()
        decision = selector.choose(self._ctx(ticker="SPY"), regime="RANGING")
        self.assertEqual(decision.strategy_name, "pairs_trading")
        self.assertEqual(decision.direction, "SELL")

    @patch("trading.strategies._fetch_closes")
    def test_strategy_selector_buy_only_blocks_short_and_returns_hold(self, mock_fetch_closes):
        # Ensure no pair signal so momentum short becomes the first trade candidate.
        n = 120
        flat = np.linspace(100.0, 101.0, n)
        mock_fetch_closes.return_value = {"SPY": flat, "QQQ": flat}

        selector = StrategySelector()
        decision = selector.choose(
            self._ctx(
                ticker="SPY",
                macd="BEARISH",
                volume="NORMAL",
                momentum_score=-1.0,  # -> momentum sell trigger at -1.5 after MACD adjustment
            ),
            regime="RANGING",
            allowed_directions={"BUY"},
        )

        self.assertFalse(decision.should_trade)
        self.assertEqual(decision.direction, "HOLD")
        self.assertIn("Trade blocked by direction filter", decision.reason)

    @patch("trading.strategies._fetch_closes")
    def test_detect_market_regime_trending(self, mock_fetch_closes):
        prices = np.linspace(100.0, 160.0, 120)
        mock_fetch_closes.return_value = {"SPY": prices}

        regime = detect_market_regime()
        self.assertEqual(regime, "TRENDING")


if __name__ == "__main__":
    unittest.main()
