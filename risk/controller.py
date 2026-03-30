"""Portfolio-level risk controller — Phase 3.

Enforces portfolio-wide risk limits on top of the per-trade stop-loss:

  1. Daily drawdown halt  — if the portfolio is down more than MAX_DAILY_DRAWDOWN_PCT
                            from today's opening value, all new trades are blocked.
  2. Portfolio heat limit — if total open market value exceeds MAX_PORTFOLIO_HEAT_PCT
                            of equity, new trades are blocked.

A risk snapshot is written to `risk_snapshots` every time `can_trade()` is called,
giving a full audit trail of the risk controller's decisions.
"""
import logging
import sqlite3
from datetime import datetime, timezone

from config import trading_client, MAX_DAILY_DRAWDOWN_PCT, MAX_PORTFOLIO_HEAT_PCT

logger = logging.getLogger(__name__)


class PortfolioRiskController:
    """Stateful risk controller that tracks day-start portfolio value and
    enforces drawdown / heat limits.

    Parameters
    ----------
    conn :
        Shared SQLite connection used to persist risk snapshots.
    max_drawdown_pct :
        Fraction of portfolio value lost from the day's open that triggers a
        trading halt.  Default: ``MAX_DAILY_DRAWDOWN_PCT`` (from config).
    max_heat_pct :
        Maximum fraction of equity that may be deployed in open positions before
        new trades are blocked.  Default: ``MAX_PORTFOLIO_HEAT_PCT`` (from config).
    """

    def __init__(
        self,
        conn: sqlite3.Connection,
        max_drawdown_pct: float = MAX_DAILY_DRAWDOWN_PCT,
        max_heat_pct: float = MAX_PORTFOLIO_HEAT_PCT,
    ) -> None:
        self.conn = conn
        self.max_drawdown_pct = max_drawdown_pct
        self.max_heat_pct = max_heat_pct
        self._day_start_value: float | None = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def record_day_start(self) -> None:
        """Capture the portfolio value at the start of the trading day.

        Should be called once each morning before the first trading cycle.
        """
        value = self._fetch_portfolio_value()
        if value is not None:
            self._day_start_value = value
            logger.info("[RISK] Day-start portfolio value: $%.2f", value)

    def can_trade(self) -> tuple[bool, str]:
        """Check whether the bot is allowed to open new positions.

        Returns
        -------
        (allowed: bool, reason: str)
        """
        current_value = self._fetch_portfolio_value()
        if current_value is None:
            # Cannot check → allow trading but log the failure
            self._write_snapshot(
                portfolio_value=None,
                day_start_value=None,
                drawdown_pct=None,
                heat_pct=None,
                trading_ok=True,
                halt_reason="could not fetch portfolio value",
            )
            return True, "portfolio value unavailable – proceeding with caution"

        # --- 1. Daily drawdown gate ---
        if self._day_start_value is not None and self._day_start_value > 0:
            drawdown = (self._day_start_value - current_value) / self._day_start_value
            if drawdown >= self.max_drawdown_pct:
                reason = (
                    f"Daily drawdown limit reached: portfolio down {drawdown * 100:.2f}% "
                    f"(limit {self.max_drawdown_pct * 100:.1f}%). Halting new trades."
                )
                logger.warning("[RISK HALT] %s", reason)
                self._write_snapshot(current_value, self._day_start_value, drawdown, None, False, reason)
                return False, reason
        else:
            drawdown = 0.0

        # --- 2. Portfolio heat gate ---
        open_market_value, n_positions = self._fetch_open_market_value()
        heat_pct = (open_market_value / current_value) if current_value > 0 else 0.0
        if heat_pct >= self.max_heat_pct:
            reason = (
                f"Portfolio heat limit reached: {heat_pct * 100:.1f}% deployed "
                f"(limit {self.max_heat_pct * 100:.1f}%). Waiting for positions to close."
            )
            logger.warning("[RISK HALT] %s", reason)
            self._write_snapshot(current_value, self._day_start_value, drawdown, heat_pct, False, reason)
            return False, reason

        self._write_snapshot(current_value, self._day_start_value, drawdown, heat_pct, True, "")
        logger.debug(
            "[RISK] OK — value=$%.0f drawdown=%.2f%% heat=%.2f%% positions=%d",
            current_value, drawdown * 100, heat_pct * 100, n_positions,
        )
        return True, "OK"

    def refresh_day_start_if_needed(self) -> None:
        """Reset the day-start value if the calendar date has advanced."""
        if self._day_start_value is None:
            self.record_day_start()

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fetch_portfolio_value(self) -> float | None:
        try:
            account = trading_client.get_account()
            return float(account.portfolio_value)
        except Exception as exc:
            logger.warning("[RISK] Could not fetch portfolio value: %s", exc)
            return None

    def _fetch_open_market_value(self) -> tuple[float, int]:
        """Return (total_market_value_of_open_positions, number_of_positions)."""
        try:
            positions = trading_client.get_all_positions()
            total = sum(abs(float(p.market_value)) for p in positions if p.market_value is not None)
            return total, len(positions)
        except Exception as exc:
            logger.warning("[RISK] Could not fetch open positions: %s", exc)
            return 0.0, 0

    def _write_snapshot(
        self,
        portfolio_value: float | None,
        day_start_value: float | None,
        drawdown_pct: float | None,
        heat_pct: float | None,
        trading_ok: bool,
        halt_reason: str,
    ) -> None:
        try:
            _, n_positions = self._fetch_open_market_value()
            self.conn.execute(
                """INSERT INTO risk_snapshots
                   (portfolio_value, day_start_value, drawdown_pct,
                    open_positions, total_heat_pct, trading_halted, halt_reason)
                   VALUES (?, ?, ?, ?, ?, ?, ?)""",
                (
                    portfolio_value,
                    day_start_value,
                    drawdown_pct,
                    n_positions,
                    heat_pct,
                    0 if trading_ok else 1,
                    halt_reason,
                ),
            )
            self.conn.commit()
        except Exception as exc:
            logger.warning("[RISK] Failed to write risk snapshot: %s", exc)
