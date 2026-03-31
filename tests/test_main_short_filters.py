import sqlite3
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import main


class TestMainShortDirectionFilters(unittest.TestCase):
    def setUp(self) -> None:
        self.conn = sqlite3.connect(":memory:")
        self.risk_ctrl = Mock()
        self.risk_ctrl.can_trade.return_value = (True, "")

    def tearDown(self) -> None:
        self.conn.close()

    def _decision(self) -> SimpleNamespace:
        return SimpleNamespace(
            strategy_name="momentum",
            regime="RANGING",
            direction="SELL",
            should_trade=True,
            reason="Momentum short setup",
            confidence="MEDIUM",
        )

    def _patch_common(self):
        selector_instance = Mock()
        selector_instance.choose.return_value = self._decision()

        return patch.multiple(
            main,
            monitor_positions=Mock(return_value=[]),
            read_setting=Mock(side_effect=[0.03, 0.05]),
            get_daily_trade_count=Mock(return_value=0),
            analyze_geopolitics=Mock(return_value="LOW_RISK"),
            analyze_fed_rate=Mock(return_value="NEUTRAL"),
            analyze_market_fear=Mock(return_value="LOW"),
            detect_market_regime=Mock(return_value="RANGING"),
            StrategySelector=Mock(return_value=selector_instance),
            analyze_sentiment=Mock(return_value="NEUTRAL"),
            get_technical_signals=Mock(
                return_value={
                    "summary": "NEUTRAL",
                    "rsi": "NEUTRAL",
                    "macd": "BEARISH",
                    "bbands": "NEUTRAL",
                    "volume": "NORMAL",
                    "momentum_score": -1.0,
                }
            ),
            get_earnings_flag=Mock(return_value="SAFE"),
            assess_risk=Mock(return_value=(True, "SELL", "risk ok")),
            reflect_on_trade=Mock(),
            _market_is_open=Mock(return_value=True),
        )

    def test_stock_short_disabled_blocks_sell(self) -> None:
        with self._patch_common(), patch.object(main, "TICKERS", ["AAPL"]), patch.object(
            main, "read_bool_setting", Mock(side_effect=[False, True])
        ), patch.object(main, "execute_trade", Mock()) as execute_trade_mock:
            main._run_trading_cycle(self.conn, self.risk_ctrl)

        execute_trade_mock.assert_not_called()

    def test_crypto_short_disabled_blocks_sell(self) -> None:
        with self._patch_common(), patch.object(main, "TICKERS", ["BTC/USD"]), patch.object(
            main, "read_bool_setting", Mock(side_effect=[True, False])
        ), patch.object(main, "execute_trade", Mock()) as execute_trade_mock:
            main._run_trading_cycle(self.conn, self.risk_ctrl)

        execute_trade_mock.assert_not_called()

    def test_stock_short_enabled_allows_sell(self) -> None:
        with self._patch_common(), patch.object(main, "TICKERS", ["AAPL"]), patch.object(
            main, "read_bool_setting", Mock(side_effect=[True, False])
        ), patch.object(main, "execute_trade", Mock(return_value=None)) as execute_trade_mock:
            main._run_trading_cycle(self.conn, self.risk_ctrl)

        execute_trade_mock.assert_called_once()
        kwargs = execute_trade_mock.call_args.kwargs
        self.assertEqual(kwargs["allow_short"], True)


if __name__ == "__main__":
    unittest.main()
