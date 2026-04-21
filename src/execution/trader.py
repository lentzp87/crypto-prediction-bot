from __future__ import annotations

"""
Trade Execution + Position Management.

Handles paper/live trade execution, Kelly-lite sizing,
tiered TP/SL, trailing stops, circuit breakers.
Scaled for $2K bankroll.
"""

import asyncio
import time
import logging
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import httpx

logger = logging.getLogger("trader")


@dataclass
class Position:
    """An open position (paper or live)."""
    trade_id: int               # DB trade ID
    ticker: str
    side: str                   # "yes" or "no"
    entry_price_cents: int
    count: int                  # number of contracts
    cost_usd: float
    mode: str                   # "paper" or "live"
    created_at: float = field(default_factory=time.time)

    # Current market state
    current_price_cents: int = 0
    unrealized_pnl: float = 0.0

    # Trailing stop state
    highest_price: int = 0
    trailing_active: bool = False
    trailing_stop_price: int = 0

    @property
    def age_minutes(self) -> float:
        return (time.time() - self.created_at) / 60.0

    def update_price(self, price_cents: int):
        """Update with latest market price."""
        self.current_price_cents = price_cents
        pnl_per_contract = (price_cents - self.entry_price_cents) / 100.0
        self.unrealized_pnl = pnl_per_contract * self.count

        # Track highest price for trailing stop
        if price_cents > self.highest_price:
            self.highest_price = price_cents

    @property
    def tp_target(self) -> int:
        """Take profit target based on entry price bucket."""
        ep = self.entry_price_cents
        if ep <= 39:
            return ep + 20
        elif ep <= 69:
            return ep + 25
        else:  # 70-85
            return 97

    @property
    def sl_target(self) -> int:
        """Stop loss target based on entry price bucket."""
        ep = self.entry_price_cents
        if ep <= 39:
            return max(1, ep - 10)
        elif ep <= 69:
            return max(1, ep - 12)
        else:
            return max(1, ep - 8)


class Trader:
    """
    Manages trade execution and position monitoring.

    Paper mode: tracks everything but doesn't hit Kalshi API.
    Live mode: places real orders via Kalshi.
    """

    def __init__(self, settings, database):
        self.settings = settings
        self.db = database
        self.positions: dict[int, Position] = {}  # trade_id → Position
        self.consecutive_losses: int = 0
        self.circuit_breaker_until: float = 0
        self.event_log: deque = deque(maxlen=settings.max_event_log_size)
        self._running = False
        self._client: Optional[httpx.AsyncClient] = None

    @property
    def mode(self) -> str:
        return "paper" if self.settings.kalshi_env == "demo" else "live"

    @property
    def is_paused(self) -> bool:
        """True if circuit breaker is active."""
        return time.time() < self.circuit_breaker_until

    # ── Trade Entry ────────────────────────────────────────────

    async def execute_trade(self, signal, consensus) -> Optional[int]:
        """
        Execute a trade based on signal + AI consensus.
        Returns trade_id or None if blocked.
        """
        # Pre-flight checks
        block_reason = self._check_limits(signal)
        if block_reason:
            self.db.record_skip(
                ticker=signal.ticker,
                side=signal.side,
                edge_cents=signal.edge_cents,
                skip_reason=block_reason,
            )
            self._log_event("SKIP", signal.ticker, block_reason)
            return None

        # Calculate position size (Kelly-lite)
        size_usd, count = self._calculate_size(signal, consensus)

        if self.mode == "paper":
            return await self._paper_trade(signal, size_usd, count)
        else:
            return await self._live_trade(signal, size_usd, count)

    def _check_limits(self, signal) -> Optional[str]:
        """Check all risk limits. Returns block reason or None."""

        # Circuit breaker
        if self.is_paused:
            remaining = (self.circuit_breaker_until - time.time()) / 60
            return f"Circuit breaker active ({remaining:.0f} min remaining)"

        # Daily loss limit
        today = self.db.get_today_stats(mode=self.mode)
        if today["today_pnl"] <= -self.settings.max_daily_loss_usd:
            return f"Daily loss limit reached (${today['today_pnl']:.2f})"

        # Daily trade count
        if today["today_trades"] >= self.settings.max_trades_per_day:
            return f"Max daily trades reached ({today['today_trades']})"

        # Max open positions
        if len(self.positions) >= self.settings.max_positions:
            return f"Max open positions ({len(self.positions)})"

        # Same strike limit
        same_strike = sum(
            1 for p in self.positions.values()
            if abs(self._parse_strike(p.ticker) - signal.strike_price) < 0.01
        )
        if same_strike >= self.settings.max_same_strike:
            return f"Max positions on same strike ({same_strike})"

        # Same window limit (positions expiring within 15 min of each other)
        # Simplified: count positions with similar tickers
        same_series = sum(
            1 for p in self.positions.values()
            if signal.ticker[:20] == p.ticker[:20]  # same date prefix
        )
        if same_series >= self.settings.max_same_window:
            return f"Max positions in same window ({same_series})"

        return None

    def _calculate_size(self, signal, consensus) -> tuple[float, int]:
        """Kelly-lite position sizing scaled for $2K bankroll."""
        base_size = self.settings.base_trade_size_usd  # $40

        confidence_mult = 1.0

        # Scale up for strong signals
        if signal.edge_cents >= 10:
            confidence_mult += 0.3
        if consensus.confidence >= 0.8:
            confidence_mult += 0.2
        if consensus.follow_count == consensus.active_count and consensus.active_count >= 3:
            confidence_mult += 0.2  # all 3 models agree

        trade_size = min(
            base_size * confidence_mult,
            self.settings.max_trade_size_usd,
        )

        # Calculate contract count
        entry_price = int(signal.kalshi_implied * 100)
        if entry_price <= 0:
            entry_price = 50
        cost_per_contract = entry_price / 100.0
        count = max(1, int(trade_size / cost_per_contract))

        actual_cost = count * cost_per_contract

        return actual_cost, count

    # ── Paper Trading ──────────────────────────────────────────

    async def _paper_trade(self, signal, cost_usd: float, count: int) -> int:
        """Record a paper trade."""
        entry_cents = int(signal.kalshi_implied * 100)

        trade_id = self.db.record_trade(
            ticker=signal.ticker,
            side=signal.side,
            entry_price_cents=entry_cents,
            count=count,
            cost_usd=cost_usd,
            mode="paper",
        )

        position = Position(
            trade_id=trade_id,
            ticker=signal.ticker,
            side=signal.side,
            entry_price_cents=entry_cents,
            count=count,
            cost_usd=cost_usd,
            mode="paper",
            current_price_cents=entry_cents,
            highest_price=entry_cents,
        )
        self.positions[trade_id] = position

        self._log_event(
            "TRADE",
            signal.ticker,
            f"Paper {signal.side.upper()} @ {entry_cents}¢ × {count} "
            f"(${cost_usd:.2f}) edge={signal.edge_cents:.1f}¢"
        )

        logger.info(
            f"Paper trade: {signal.side} {signal.ticker} "
            f"@ {entry_cents}¢ × {count} = ${cost_usd:.2f}"
        )

        return trade_id

    # ── Live Trading ───────────────────────────────────────────

    async def _live_trade(self, signal, cost_usd: float, count: int) -> Optional[int]:
        """Place a real order on Kalshi."""
        entry_cents = int(signal.kalshi_implied * 100)

        try:
            # TODO: Implement Kalshi order placement
            # POST /trade-api/v2/portfolio/orders
            # For now, fall back to paper-like tracking
            logger.warning("Live trading not yet implemented — treating as paper")
            return await self._paper_trade(signal, cost_usd, count)

        except Exception as e:
            logger.error(f"Live trade failed: {e}")
            self._log_event("ERROR", signal.ticker, f"Live trade failed: {e}")
            return None

    # ── Position Monitoring ────────────────────────────────────

    async def monitor_positions(self, get_market_price):
        """
        Continuously monitor open positions for TP/SL/trailing/stale.
        get_market_price: async callable(ticker) → price_cents or None
        """
        self._running = True

        while self._running:
            try:
                positions_to_close = []

                for trade_id, pos in list(self.positions.items()):
                    # Get current market price
                    price = await get_market_price(pos.ticker)
                    if price is not None:
                        pos.update_price(price)

                    # Check exit conditions
                    exit_reason = self._check_exit(pos)
                    if exit_reason:
                        positions_to_close.append((trade_id, exit_reason))

                # Close positions outside the iteration
                for trade_id, reason in positions_to_close:
                    await self._close_position(trade_id, reason)

            except Exception as e:
                logger.error(f"Position monitor error: {e}")

            await asyncio.sleep(self.settings.exit_monitor_interval_sec)

    def _check_exit(self, pos: Position) -> Optional[str]:
        """Check if position should be closed. Returns exit reason or None."""

        price = pos.current_price_cents
        entry = pos.entry_price_cents
        profit_cents = price - entry

        # Take profit
        if price >= pos.tp_target:
            return f"take_profit (target={pos.tp_target}¢, got={price}¢)"

        # Stop loss
        if price <= pos.sl_target:
            return f"stop_loss (target={pos.sl_target}¢, got={price}¢)"

        # Trailing stop logic
        if profit_cents >= 6 and not pos.trailing_active:
            pos.trailing_active = True
            pos.trailing_stop_price = entry + 2  # lock in +2¢
            logger.debug(f"Trailing stop activated for {pos.ticker} at {pos.trailing_stop_price}¢")

        if pos.trailing_active:
            # Update trailing stop based on highest price
            new_trail = pos.highest_price - 4
            if new_trail > pos.trailing_stop_price:
                pos.trailing_stop_price = new_trail

            # High-value positions: tighter trail
            if price >= 80:
                tight_trail = pos.highest_price - 5
                if tight_trail > pos.trailing_stop_price:
                    pos.trailing_stop_price = tight_trail

            if price <= pos.trailing_stop_price:
                return f"trailing_stop (trail={pos.trailing_stop_price}¢, price={price}¢)"

        # Stale timeout (15-min contracts)
        if pos.age_minutes >= self.settings.stale_timeout_15m_minutes:
            return f"stale_timeout ({pos.age_minutes:.0f} min old)"

        # Settlement (contract resolved)
        if price >= 99 or price <= 1:
            return f"settlement (price={price}¢)"

        return None

    async def _close_position(self, trade_id: int, reason: str):
        """Close a position and update DB."""
        pos = self.positions.pop(trade_id, None)
        if not pos:
            return

        exit_price = pos.current_price_cents
        pnl_per_contract = (exit_price - pos.entry_price_cents) / 100.0
        pnl = pnl_per_contract * pos.count

        self.db.close_trade(
            trade_id=trade_id,
            exit_price_cents=exit_price,
            pnl_usd=pnl,
            exit_reason=reason,
        )

        # Track consecutive losses for circuit breaker
        if pnl < 0:
            self.consecutive_losses += 1
            if self.consecutive_losses >= self.settings.circuit_breaker_losses:
                pause_sec = self.settings.circuit_breaker_pause_min * 60
                self.circuit_breaker_until = time.time() + pause_sec
                self._log_event(
                    "CIRCUIT_BREAKER",
                    pos.ticker,
                    f"{self.consecutive_losses} consecutive losses — pausing {self.settings.circuit_breaker_pause_min} min"
                )
                logger.warning(f"Circuit breaker triggered: {self.consecutive_losses} losses")
        else:
            self.consecutive_losses = 0

        self._log_event(
            "EXIT",
            pos.ticker,
            f"{reason} | {pos.side} @ {pos.entry_price_cents}¢→{exit_price}¢ "
            f"P&L: ${pnl:+.2f}"
        )

        logger.info(
            f"Closed {pos.ticker}: {reason} | "
            f"{pos.entry_price_cents}¢→{exit_price}¢ P&L=${pnl:+.2f}"
        )

    async def stop(self):
        self._running = False

    # ── Helpers ─────────────────────────────────────────────────

    @staticmethod
    def _parse_strike(ticker: str) -> float:
        """Extract strike price from ticker."""
        try:
            return float(ticker.split("-T")[-1])
        except (ValueError, IndexError):
            return 0.0

    def _log_event(self, event_type: str, ticker: str, detail: str):
        self.event_log.appendleft({
            "type": event_type,
            "ticker": ticker,
            "detail": detail,
            "timestamp": time.time(),
        })

    # ── Serialization ──────────────────────────────────────────

    def get_positions_dict(self) -> list[dict]:
        return [
            {
                "trade_id": p.trade_id,
                "ticker": p.ticker,
                "side": p.side,
                "entry_cents": p.entry_price_cents,
                "current_cents": p.current_price_cents,
                "count": p.count,
                "cost_usd": p.cost_usd,
                "unrealized_pnl": round(p.unrealized_pnl, 2),
                "age_min": round(p.age_minutes, 1),
                "tp_target": p.tp_target,
                "sl_target": p.sl_target,
                "trailing_active": p.trailing_active,
                "mode": p.mode,
            }
            for p in self.positions.values()
        ]

    def get_events(self, limit: int = 200) -> list[dict]:
        return list(self.event_log)[:limit]

    def to_dict(self) -> dict:
        stats = self.db.get_trade_stats(mode=self.mode)
        today = self.db.get_today_stats(mode=self.mode)
        return {
            "mode": self.mode,
            "open_positions": len(self.positions),
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_active": self.is_paused,
            **stats,
            **today,
        }
