import unittest

import pandas as pd

from pnl.attribution import compute_signal_accuracy, compute_signal_outcome_breakdown


class TestAttributionPhase6(unittest.TestCase):
    def test_signal_accuracy_includes_sentiment(self) -> None:
        closed_trades = pd.DataFrame(
            [
                {
                    "sentiment": "BULLISH",
                    "technical_signal": "BULLISH",
                    "price_move_pct": 0.020,
                    "realized_pnl": 10.0,
                },
                {
                    "sentiment": "BULLISH",
                    "technical_signal": "BEARISH",
                    "price_move_pct": -0.010,
                    "realized_pnl": -5.0,
                },
                {
                    "sentiment": "BEARISH",
                    "technical_signal": "BEARISH",
                    "price_move_pct": -0.030,
                    "realized_pnl": 7.0,
                },
                {
                    "sentiment": "NEUTRAL",
                    "technical_signal": "NEUTRAL",
                    "price_move_pct": 0.040,
                    "realized_pnl": 2.0,
                },
                {
                    "sentiment": "BEARISH",
                    "technical_signal": "BULLISH",
                    "price_move_pct": 0.000,
                    "realized_pnl": 0.0,
                },
            ]
        )

        accuracy_df = compute_signal_accuracy(closed_trades)
        self.assertFalse(accuracy_df.empty)

        sentiment_row = accuracy_df[accuracy_df["signal"] == "sentiment"].iloc[0]
        self.assertEqual(int(sentiment_row["samples"]), 3)
        self.assertEqual(int(sentiment_row["correct"]), 2)
        self.assertAlmostEqual(float(sentiment_row["accuracy"]), 2 / 3, places=6)

        technical_row = accuracy_df[accuracy_df["signal"] == "technical_signal"].iloc[0]
        self.assertEqual(int(technical_row["samples"]), 3)
        self.assertEqual(int(technical_row["correct"]), 3)
        self.assertAlmostEqual(float(technical_row["accuracy"]), 1.0, places=6)

    def test_signal_outcome_breakdown_counts_and_win_rate(self) -> None:
        closed_trades = pd.DataFrame(
            [
                {"sentiment": "BULLISH", "realized_pnl": 12.0},
                {"sentiment": "BULLISH", "realized_pnl": -3.0},
                {"sentiment": "BULLISH", "realized_pnl": 0.0},
                {"sentiment": "BEARISH", "realized_pnl": -8.0},
            ]
        )

        breakdown_df = compute_signal_outcome_breakdown(closed_trades)
        self.assertFalse(breakdown_df.empty)

        bullish = breakdown_df[
            (breakdown_df["signal"] == "sentiment")
            & (breakdown_df["signal_state"] == "BULLISH")
        ].iloc[0]

        self.assertEqual(int(bullish["trades"]), 3)
        self.assertEqual(int(bullish["wins"]), 1)
        self.assertEqual(int(bullish["losses"]), 1)
        self.assertEqual(int(bullish["flat"]), 1)
        self.assertAlmostEqual(float(bullish["total_pnl"]), 9.0, places=6)
        self.assertAlmostEqual(float(bullish["avg_pnl"]), 3.0, places=6)
        self.assertAlmostEqual(float(bullish["win_rate"]), 1 / 3, places=6)


if __name__ == "__main__":
    unittest.main()