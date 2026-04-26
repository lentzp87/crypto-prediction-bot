from __future__ import annotations

"""
Trade Execution + Position Management.

Handles paper/live trade execution, Kelly-lite sizing,
tiered TP/SL, trailing stops, circuit breakers.
Scaled for $2K bankroll.
"""

import asyncio
import base64
import os
import time
import logging
import uuid
from dataclasses import dataclass, field
from typing import Optional
from collections import deque

import httpx
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import padding

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
    synced: bool = False         # True if loaded from Kalshi on startup (not bot-placed)
    created_at: float = field(default_factory=time.time)

    # Current market state
    current_price_cents: int = 0
    unrealized_pnl: float = 0.0
    close_attempts: int = 0
    minutes_to_close: float = 0.0  # actual contract expiry, not position age

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
        """Dynamic take profit based on entry price and edge quality."""
        ep = self.entry_price_cents
        if ep <= 39:
            return ep + 20
        elif ep <= 69:
            return ep + 25
        else:
            return 97

    @property
    def sl_target(self) -> int:
        """Dynamic stop loss — tighter for aged positions."""
        ep = self.entry_price_cents
        age = self.age_minutes

        # Base stop loss by entry bucket
        if ep <= 39:
            base_sl = max(1, ep - 10)
        elif ep <= 69:
            base_sl = max(1, ep - 12)
        else:
            base_sl = max(1, ep - 8)

        # Tighten stop as position ages (for 15M contracts)
        if age > 10:
            # After 10 min, tighten by 3¢
            base_sl = max(base_sl, ep - 5)

        return max(1, base_sl)


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

        # Live Kalshi balance (updated periodically)
        self.kalshi_cash: float = 0.0
        self.kalshi_portfolio: float = 0.0
        self._balance_updated_at: float = 0

        # Daily equity tracking — reset each UTC day
        self._daily_start_balance: float = 0.0
        self._daily_start_date: str = ""

        # Cooldown tracking: ticker_series → timestamp of last close
        self._cooldowns: dict[str, float] = {}

        # Adaptive learning — updated by bot's learning loop
        self.adaptive_sizing_mult: float = 1.0
        self.asset_preference: dict = {}

    @property
    def mode(self) -> str:
        return getattr(self.settings, 'trading_mode', 'paper')

    @property
    def is_paused(self) -> bool:
        """True if circuit breaker is active."""
        return time.time() < self.circuit_breaker_until

    async def verify_live_auth(self) -> bool:
        """Check that Kalshi API auth works before placing real orders."""
        if self.mode != "live":
            return True
        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=15)
            path = "/trade-api/v2/portfolio/balance"
            headers = self._sign_request("GET", path)
            resp = await self._client.get(
                f"{self.settings.kalshi_base_url}{path}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            balance_cents = data.get("balance", 0)
            logger.info(f"LIVE MODE: Kalshi auth verified, balance=${balance_cents/100:.2f}")
            self._log_event("SYSTEM", "KALSHI", f"Live auth OK — balance=${balance_cents/100:.2f}")
            return True
        except Exception as e:
            logger.error(f"LIVE MODE AUTH FAILED: {e} — falling back to paper")
            self._log_event("ERROR", "KALSHI", f"Live auth failed: {e} — using paper mode")
            return False

    async def sync_kalshi_positions(self):
        """
        Load existing positions from Kalshi API into internal tracking.
        Called on startup so the bot can manage positions across restarts.
        """
        if self.mode != "live":
            return
        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=15)

            path = "/trade-api/v2/portfolio/positions"
            headers = self._sign_request("GET", path)
            resp = await self._client.get(
                f"{self.settings.kalshi_base_url}{path}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()

            positions = data.get("market_positions", [])
            synced = 0

            # Build set of already-tracked tickers to avoid duplicating on restart
            already_tracked = {
                (p.ticker, p.side) for p in self.positions.values()
            }

            for p in positions:
                ticker = p.get("ticker", "")
                pos_fp = float(p.get("position_fp", 0))
                if pos_fp == 0:
                    continue  # flat position, skip

                exposure = float(p.get("market_exposure_dollars", "0"))
                count = int(abs(pos_fp))
                side = "yes" if pos_fp > 0 else "no"

                # Skip if we already track this ticker+side (prevents duplication)
                # Check BOTH the pre-loop set AND positions added during this loop
                if (ticker, side) in already_tracked:
                    logger.info(f"Skipping already-tracked position: {ticker} {side}")
                    continue

                # Mark as tracked NOW so duplicates later in the same loop are caught
                already_tracked.add((ticker, side))

                # Calculate average entry price
                if count > 0:
                    entry_cents = int((exposure / count) * 100)
                else:
                    entry_cents = 50

                # Record in DB so it persists
                trade_id = self.db.record_trade(
                    ticker=ticker,
                    side=side,
                    entry_price_cents=entry_cents,
                    count=count,
                    cost_usd=exposure,
                    mode="live",
                )

                position = Position(
                    trade_id=trade_id,
                    ticker=ticker,
                    side=side,
                    entry_price_cents=entry_cents,
                    count=count,
                    cost_usd=exposure,
                    mode="live",
                    synced=True,  # Mark as synced — losses don't count for circuit breaker
                    current_price_cents=entry_cents,
                    highest_price=entry_cents,
                )
                self.positions[trade_id] = position
                synced += 1

            if synced > 0:
                logger.info(f"Synced {synced} existing Kalshi positions")
                self._log_event("SYSTEM", "KALSHI", f"Synced {synced} positions from Kalshi")
            else:
                logger.info("No existing Kalshi positions to sync")

            # Restore circuit breaker state from DB (persists across restarts)
            try:
                cb_until = float(self.db.get_state("circuit_breaker_until", "0"))
                cb_losses = int(self.db.get_state("consecutive_losses", "0"))
                if cb_until > time.time():
                    self.circuit_breaker_until = cb_until
                    remaining = (cb_until - time.time()) / 60
                    logger.info(f"Circuit breaker restored: {remaining:.0f} min remaining")
                    self._log_event("SYSTEM", "KALSHI",
                        f"Circuit breaker restored from DB ({remaining:.0f} min remaining)")
                else:
                    self.circuit_breaker_until = 0
                self.consecutive_losses = cb_losses
                logger.info(f"Restored circuit breaker state: {cb_losses} consecutive losses")
            except Exception as e:
                logger.warning(f"Could not restore circuit breaker state: {e}")
                self.consecutive_losses = 0
                self.circuit_breaker_until = 0

        except Exception as e:
            logger.error(f"Failed to sync Kalshi positions: {e}")
            self._log_event("ERROR", "KALSHI", f"Position sync failed: {e}")

    async def fetch_kalshi_balance(self):
        """Fetch live cash + portfolio value from Kalshi API."""
        if self.mode != "live":
            return
        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=15)
            path = "/trade-api/v2/portfolio/balance"
            headers = self._sign_request("GET", path)
            resp = await self._client.get(
                f"{self.settings.kalshi_base_url}{path}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json()
            self.kalshi_cash = data.get("balance", 0) / 100.0
            self.kalshi_portfolio = data.get("portfolio_value", 0) / 100.0
            self._balance_updated_at = time.time()

            # Track daily starting equity — snapshot the first balance fetch each UTC day
            from datetime import datetime
            today_str = datetime.utcnow().strftime("%Y-%m-%d")
            if self._daily_start_date != today_str:
                total = self.kalshi_cash + self.kalshi_portfolio
                self._daily_start_balance = total
                self._daily_start_date = today_str
                logger.info(f"Daily starting equity set: ${total:.2f} for {today_str}")
        except Exception as e:
            logger.warning(f"Failed to fetch Kalshi balance: {e}")

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
            return await self._paper_trade(signal, size_usd, count, consensus)
        else:
            return await self._live_trade(signal, size_usd, count, consensus)

    def _check_limits(self, signal) -> Optional[str]:
        """Check all risk limits. Returns block reason or None."""

        # Circuit breaker
        if self.is_paused:
            remaining = (self.circuit_breaker_until - time.time()) / 60
            return f"Circuit breaker active ({remaining:.0f} min remaining)"

        # ── Price sanity: reject bad risk/reward ──────────────
        entry_cents = int(signal.kalshi_implied * 100)

        # WIDENED: trade 20-80¢ range (was 35-65¢) — more aggressive
        if entry_cents < 20:
            return f"Price too low ({entry_cents}¢ < 20¢) — longshot"
        if entry_cents > 80:
            return f"Price too high ({entry_cents}¢ > 80¢) — bad risk/reward"

        # ── Duplicate ticker check: NEVER buy a contract we already hold ──
        # This prevents position stacking across restarts
        for p in self.positions.values():
            if p.ticker == signal.ticker and p.side == signal.side:
                return f"Already holding {signal.ticker} {signal.side}"

        # Daily loss limit — use real Kalshi balance vs TODAY'S starting equity,
        # not the original deposit. Otherwise multi-day drawdowns compound unfairly.
        today = self.db.get_today_stats(mode=self.mode)
        if self.mode == "live" and self.kalshi_cash > 0 and self._daily_start_balance > 0:
            real_total = self.kalshi_cash + self.kalshi_portfolio
            daily_pnl = real_total - self._daily_start_balance
            if daily_pnl <= -self.settings.max_daily_loss_usd:
                return f"Daily loss limit reached (today: ${daily_pnl:.2f} vs start ${self._daily_start_balance:.2f})"
        else:
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

        # ── Cooldown: don't re-enter a contract series we just traded ──
        series_key = signal.ticker[:25]  # e.g. "KXBTCD-26APR2317-T78"
        last_close = self._cooldowns.get(series_key, 0)
        cooldown_remaining = self.settings.cooldown_seconds - (time.time() - last_close)
        if cooldown_remaining > 0:
            return f"Cooldown on {series_key} ({cooldown_remaining:.0f}s remaining)"

        return None

    # ── Max contracts: Kalshi orderbooks are thin, limit to avoid
    # eating through the book and getting terrible fills ──
    MAX_CONTRACTS = 10   # full send — chunked 1 at a time for thin books

    def _calculate_size(self, signal, consensus) -> tuple[float, int]:
        """Kelly-lite position sizing with max-loss cap + contract cap."""
        base_size = self.settings.base_trade_size_usd

        # Trade quality score — edge is just one ingredient
        quality_score = 1.0

        # Edge quality
        if signal.edge_cents >= 14:
            quality_score += 0.3  # strong edge
        elif signal.edge_cents >= 10:
            quality_score += 0.1  # decent edge

        # AI confidence
        if consensus.confidence >= 0.75:
            quality_score += 0.2

        # Unanimous agreement bonus
        if consensus.follow_count == consensus.active_count and consensus.active_count >= 3:
            quality_score += 0.2

        # Spread penalty (if available from signal)
        spread = signal.indicators.get("spread", 0) if hasattr(signal, 'indicators') else 0
        if spread > 4:
            quality_score -= 0.2  # wide spread = worse quality

        # Apply adaptive sizing multiplier (learned from recent performance)
        quality_score *= self.adaptive_sizing_mult

        # Apply asset preference multiplier (learned from asset-specific win rates)
        ticker = signal.ticker if hasattr(signal, 'ticker') else ""
        asset = "ETH" if "ETH" in ticker else "BTC"
        asset_pref = self.asset_preference.get(asset, {})
        asset_mult = asset_pref.get("size_mult", 1.0) if asset_pref else 1.0
        quality_score *= asset_mult

        trade_size = min(
            base_size * quality_score,
            self.settings.max_trade_size_usd,
        )

        # Calculate contract count
        entry_price = int(signal.kalshi_implied * 100)
        if entry_price <= 0:
            entry_price = 50
        cost_per_contract = entry_price / 100.0
        count = max(1, int(trade_size / cost_per_contract))

        # ── Max loss cap: never risk more than $30 on a single position ──
        max_loss_usd = 30.0
        max_contracts_by_loss = max(1, int(max_loss_usd / cost_per_contract))
        count = min(count, max_contracts_by_loss)

        # ── Hard contract cap: Kalshi books are thin ──
        count = min(count, getattr(self.settings, 'max_contracts_per_trade', self.MAX_CONTRACTS))

        actual_cost = count * cost_per_contract

        return actual_cost, count

    # ── Paper Trading (realistic simulation) ─────────────────

    # Simulate real-world friction so paper results ≈ live results
    PAPER_ENTRY_SLIPPAGE = 1   # ¢ worse on entry (signal already uses ask price)
    PAPER_EXIT_SLIPPAGE = 2    # ¢ worse on exit (hitting the bid)

    async def _paper_trade(self, signal, cost_usd: float, count: int, consensus=None) -> int:
        """Record a paper trade — signal already uses executable ask price."""
        raw_entry = int(signal.kalshi_implied * 100)

        # Signal already uses ask price (executable), add 1¢ for realism
        entry_cents = min(raw_entry + 1, 95)

        # Recalculate cost with slipped entry
        cost_usd = count * (entry_cents / 100.0)

        trade_id = self.db.record_trade(
            ticker=signal.ticker,
            side=signal.side,
            entry_price_cents=entry_cents,
            count=count,
            cost_usd=cost_usd,
            mode="paper",
            follow_count=consensus.follow_count if consensus else None,
            active_count=consensus.active_count if consensus else None,
            avg_confidence=round(consensus.confidence, 3) if consensus else None,
            edge_cents=round(signal.edge_cents, 1) if signal else None,
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
            minutes_to_close=signal.minutes_to_close,
        )
        self.positions[trade_id] = position

        self._log_event(
            "TRADE",
            signal.ticker,
            f"Paper {signal.side.upper()} @ {entry_cents}¢ (raw={raw_entry}¢ +{self.PAPER_ENTRY_SLIPPAGE}¢ slip) "
            f"× {count} (${cost_usd:.2f}) edge={signal.edge_cents:.1f}¢"
        )

        logger.info(
            f"Paper trade: {signal.side} {signal.ticker} "
            f"@ {entry_cents}¢ (raw {raw_entry}¢) × {count} = ${cost_usd:.2f}"
        )

        return trade_id

    # ── Live Trading ───────────────────────────────────────────

    async def _live_trade(self, signal, cost_usd: float, count: int, consensus=None) -> Optional[int]:
        """Place a real limit order on Kalshi."""
        # Signal already carries the executable ask price from the scanner.
        # Add 1¢ tolerance to ensure fill, but no more — edge was calculated at ask.
        entry_cents = min(int(signal.kalshi_implied * 100) + 1, 95)

        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=15)

            path = "/trade-api/v2/portfolio/orders"
            headers = self._sign_request("POST", path)
            headers["Content-Type"] = "application/json"

            # Buy in chunks of 2 to avoid insufficient_resting_volume errors
            # Kalshi books are very thin — even 3 can fail
            total_filled = 0
            remaining = count
            chunk_size = 1
            order_id = ""

            while remaining > 0:
                buy_count = min(remaining, chunk_size)
                order_body = {
                    "ticker": signal.ticker,
                    "action": "buy",
                    "side": signal.side,
                    "count": buy_count,
                    "type": "limit",
                    "yes_price": entry_cents if signal.side == "yes" else None,
                    "no_price": entry_cents if signal.side == "no" else None,
                    "time_in_force": "fill_or_kill",
                    "client_order_id": str(uuid.uuid4()),
                }
                order_body = {k: v for k, v in order_body.items() if v is not None}

                resp = await self._client.post(
                    f"{self.settings.kalshi_base_url}{path}",
                    headers=headers,
                    json=order_body,
                )
                resp.raise_for_status()
                order_data = resp.json().get("order", {})
                order_id = order_data.get("order_id", order_id)
                status = order_data.get("status", "unknown")
                raw_filled = order_data.get("fill_count", 0) or order_data.get("fill_count_fp", 0)
                try:
                    filled = int(float(raw_filled)) if raw_filled else 0
                except (ValueError, TypeError):
                    filled = 0

                if filled > 0:
                    total_filled += filled
                    remaining -= filled
                else:
                    break  # no more liquidity

            if total_filled > 0:
                actual_count = total_filled
                actual_cost = actual_count * (entry_cents / 100.0)

                trade_id = self.db.record_trade(
                    ticker=signal.ticker,
                    side=signal.side,
                    entry_price_cents=entry_cents,
                    count=actual_count,
                    cost_usd=actual_cost,
                    mode="live",
                    follow_count=consensus.follow_count if consensus else None,
                    active_count=consensus.active_count if consensus else None,
                    avg_confidence=round(consensus.confidence, 3) if consensus else None,
                    edge_cents=round(signal.edge_cents, 1) if signal else None,
                )

                position = Position(
                    trade_id=trade_id,
                    ticker=signal.ticker,
                    side=signal.side,
                    entry_price_cents=entry_cents,
                    count=actual_count,
                    cost_usd=actual_cost,
                    mode="live",
                    current_price_cents=entry_cents,
                    highest_price=entry_cents,
                    minutes_to_close=signal.minutes_to_close,
                )
                self.positions[trade_id] = position

                self._log_event(
                    "LIVE_TRADE",
                    signal.ticker,
                    f"LIVE {signal.side.upper()} @ {entry_cents}¢ × {actual_count} "
                    f"(${actual_cost:.2f}) edge={signal.edge_cents:.1f}¢"
                )
                logger.info(
                    f"LIVE order filled: {signal.side} {signal.ticker} "
                    f"@ {entry_cents}¢ × {actual_count} = ${actual_cost:.2f} "
                    f"order_id={order_id}"
                )
                return trade_id
            else:
                # Order not filled (fill_or_kill rejected)
                self._log_event(
                    "ORDER_REJECTED",
                    signal.ticker,
                    f"Fill-or-kill not filled: {signal.side} @ {entry_cents}¢ × {count} "
                    f"status={status}"
                )
                logger.warning(f"Order not filled: {status} for {signal.ticker}")
                return None

        except httpx.HTTPStatusError as e:
            body = e.response.text[:300] if e.response else ""
            logger.error(f"Kalshi order API error: {e.response.status_code} {body}")
            self._log_event("ERROR", signal.ticker, f"Order API {e.response.status_code}: {body[:100]}")
            return None
        except Exception as e:
            logger.error(f"Live trade failed: {e}")
            self._log_event("ERROR", signal.ticker, f"Live trade failed: {e}")
            return None

    async def _live_close(self, pos: Position) -> Optional[float]:
        """Close a live position by selling on Kalshi."""
        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=15)

            path = "/trade-api/v2/portfolio/orders"
            headers = self._sign_request("POST", path)
            headers["Content-Type"] = "application/json"

            # For sells, price LOWER to cross the spread (accept the bid)
            # Use a bigger discount (5¢) to ensure fill
            exit_cents = max(1, pos.current_price_cents - 5)

            # Try to sell remaining contracts (supports partial fills across retries)
            remaining = pos.count
            total_filled = 0

            # Sell in smaller chunks — Kalshi books are very thin
            chunk_size = 1  # max 2 at a time

            while remaining > 0:
                sell_count = min(remaining, chunk_size)
                order_body = {
                    "ticker": pos.ticker,
                    "action": "sell",
                    "side": pos.side,
                    "count": sell_count,
                    "type": "limit",
                    "yes_price": exit_cents if pos.side == "yes" else None,
                    "no_price": exit_cents if pos.side == "no" else None,
                    "time_in_force": "fill_or_kill",
                    "client_order_id": str(uuid.uuid4()),
                }
                order_body = {k: v for k, v in order_body.items() if v is not None}

                resp = await self._client.post(
                    f"{self.settings.kalshi_base_url}{path}",
                    headers=headers,
                    json=order_body,
                )
                resp.raise_for_status()
                order_data = resp.json().get("order", {})
                status = order_data.get("status", "unknown")
                raw_filled = order_data.get("fill_count", 0) or order_data.get("fill_count_fp", 0)
                try:
                    filled = int(float(raw_filled)) if raw_filled else 0
                except (ValueError, TypeError):
                    filled = 0

                if filled > 0:
                    total_filled += filled
                    remaining -= filled
                else:
                    # This chunk didn't fill — stop trying
                    break

            if total_filled > 0:
                pnl = (exit_cents - pos.entry_price_cents) / 100.0 * total_filled
                logger.info(f"LIVE close filled: {pos.ticker} @ {exit_cents}¢ × {total_filled} P&L=${pnl:+.2f}")
                self._log_event("LIVE_EXIT", pos.ticker,
                    f"Sold {pos.side} @ {exit_cents}¢ × {total_filled}/{pos.count} P&L=${pnl:+.2f}")

                if remaining > 0:
                    # Partial close — update position with remaining contracts
                    pos.count = remaining
                    pos.cost_usd = remaining * (pos.entry_price_cents / 100.0)
                    self.positions[pos.trade_id] = pos  # put back with reduced count
                    self._log_event("PARTIAL", pos.ticker,
                        f"{remaining} contracts remaining — will retry")

                return pnl
            else:
                logger.warning(f"Close order not filled for {pos.ticker}: "
                              f"side={pos.side} price={exit_cents}¢ count={pos.count}")
                self._log_event("CLOSE_FAIL", pos.ticker,
                    f"Not filled: {pos.side} @ {exit_cents}¢ × {pos.count}")
                return None

        except httpx.HTTPStatusError as e:
            body = e.response.text[:200] if e.response else ""
            logger.error(f"Live close API error for {pos.ticker}: {e.response.status_code} {body}")
            self._log_event("CLOSE_ERROR", pos.ticker,
                f"API {e.response.status_code}: {body[:100]}")
            return None
        except Exception as e:
            logger.error(f"Live close failed for {pos.ticker}: {e}")
            self._log_event("CLOSE_ERROR", pos.ticker, f"Failed: {e}")
            return None

    # ── Kalshi Request Signing ─────────────────────────────────

    def _load_private_key(self):
        """Load RSA private key (cached). Tries B64 env → PEM env → file."""
        if not hasattr(self, '_private_key') or self._private_key is None:
            try:
                # Option 1: Base64-encoded PEM (immune to Render newline mangling)
                b64_env = os.environ.get("KALSHI_PRIVATE_KEY_B64", "")
                if b64_env:
                    pem_bytes = base64.b64decode(b64_env)
                    self._private_key = serialization.load_pem_private_key(pem_bytes, password=None)
                    logger.info("Trader RSA key loaded from B64 env var")
                    return self._private_key

                # Option 2: Raw PEM string (fix newline mangling)
                pem_env = os.environ.get("KALSHI_PRIVATE_KEY_PEM", "")
                if pem_env:
                    if "\\n" in pem_env:
                        pem_env = pem_env.replace("\\n", "\n")
                    self._private_key = serialization.load_pem_private_key(
                        pem_env.encode(), password=None
                    )
                    logger.info("Trader RSA key loaded from PEM env var")
                    return self._private_key

                # Option 3: PEM file
                key_path = self.settings.kalshi_private_key_path
                with open(key_path, "rb") as f:
                    self._private_key = serialization.load_pem_private_key(
                        f.read(), password=None
                    )
                logger.info("Trader RSA key loaded from file")
            except Exception as e:
                logger.error(f"Failed to load RSA key for trading: {e}")
                self._private_key = None
        return self._private_key

    def _sign_request(self, method: str, path: str) -> dict:
        """Sign a Kalshi API request with RSA-PSS."""
        if not self.settings.kalshi_api_key:
            return {}
        private_key = self._load_private_key()
        if not private_key:
            return {}

        timestamp_ms = str(int(time.time() * 1000))
        path_no_query = path.split("?")[0]
        message = f"{timestamp_ms}{method.upper()}{path_no_query}".encode()

        signature = private_key.sign(
            message,
            padding.PSS(
                mgf=padding.MGF1(hashes.SHA256()),
                salt_length=padding.PSS.DIGEST_LENGTH,
            ),
            hashes.SHA256(),
        )
        return {
            "KALSHI-ACCESS-KEY": self.settings.kalshi_api_key,
            "KALSHI-ACCESS-TIMESTAMP": timestamp_ms,
            "KALSHI-ACCESS-SIGNATURE": base64.b64encode(signature).decode(),
        }

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
                    # Get current market price (returns YES probability)
                    price = await get_market_price(pos.ticker)
                    if price is not None:
                        # For NO positions, our value = 100 - YES probability
                        effective_price = (100 - price) if pos.side == "no" else price
                        pos.update_price(effective_price)

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

        # ── Time-aware exits ──
        # Use actual contract time remaining, not position age
        actual_remaining = pos.minutes_to_close - pos.age_minutes
        is_15m = "15M" in pos.ticker

        # For LIVE 15M contracts: LET THEM SETTLE. The edge is in the binary
        # outcome (0¢ or 100¢). Exiting early at the model price = $0 P&L every time.
        # Only exit early if we have a big profit to lock in (20¢+).
        if is_15m and pos.mode == "live":
            if actual_remaining <= 3 and profit_cents >= 20:
                return f"time_exit (big profit, {actual_remaining:.0f}min left, locking in +{profit_cents}¢)"
            # Otherwise let Kalshi auto-settle — don't exit early
            # The position will resolve at 0 or 100 at expiry
            pass
        elif is_15m:
            # Paper mode: still use time exits
            if actual_remaining <= 3 and profit_cents > 0:
                return f"time_exit (profitable, {actual_remaining:.0f}min left, locking in +{profit_cents}¢)"
            if actual_remaining <= 1:
                return f"time_exit ({actual_remaining:.0f}min left, P&L={profit_cents:+d}¢)"

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
        """Close a position and update DB. For live positions, sell on Kalshi first."""
        pos = self.positions.pop(trade_id, None)
        if not pos:
            return

        exit_price = pos.current_price_cents
        original_count = pos.count  # snapshot before _live_close can mutate it

        # For live positions, place a sell order on Kalshi
        # Skip selling if price is near settlement (99¢/1¢) — let Kalshi auto-settle
        near_settlement = exit_price >= 97 or exit_price <= 3
        # For 15M contracts near expiry, skip the sell — let Kalshi auto-settle
        is_15m = "15M" in pos.ticker
        if is_15m and pos.age_minutes >= 14:
            self._log_event("EXIT", pos.ticker,
                f"15M contract near expiry ({pos.age_minutes:.0f}m) — letting Kalshi auto-settle")
        elif pos.mode == "live" and not near_settlement and "settlement" not in reason:
            pos.close_attempts += 1
            result = await self._live_close(pos)
            if result is None:
                if pos.close_attempts >= 3:
                    # Give up after 3 attempts — let it settle or try again later
                    self._log_event("FORCE_CLOSE", pos.ticker,
                        f"Giving up after {pos.close_attempts} close attempts — recording as loss")
                else:
                    # Retry next cycle
                    self.positions[trade_id] = pos
                    self._log_event("WARN", pos.ticker,
                        f"Close failed attempt {pos.close_attempts}/3 ({reason}), will retry")
                    return

            # Handle partial fills: _live_close may have reduced pos.count
            if pos.count < original_count and pos.count > 0:
                # Partial fill — record P&L for filled contracts only,
                # keep the position open for remaining contracts
                filled_count = original_count - pos.count
                pnl_partial = (exit_price - pos.entry_price_cents) / 100.0 * filled_count

                # Record the partial close as a separate trade close
                # but DON'T close the original — it's still alive with fewer contracts
                self._log_event("PARTIAL_CLOSE", pos.ticker,
                    f"Closed {filled_count}/{original_count} @ {exit_price}¢ P&L=${pnl_partial:+.2f}, "
                    f"{pos.count} remaining")

                # Update consecutive losses / circuit breaker for the partial
                if pnl_partial < 0 and not pos.synced:
                    self.consecutive_losses += 1
                    if self.consecutive_losses >= self.settings.circuit_breaker_losses and not self.is_paused:
                        pause_sec = self.settings.circuit_breaker_pause_min * 60
                        self.circuit_breaker_until = time.time() + pause_sec
                        self._log_event("CIRCUIT_BREAKER", pos.ticker,
                            f"{self.consecutive_losses} consecutive losses — pausing")
                        self.consecutive_losses = 0
                elif pnl_partial >= 0:
                    self.consecutive_losses = 0

                # Position is already back in self.positions (put there by _live_close)
                return

        elif pos.mode == "live" and near_settlement:
            self._log_event("EXIT", pos.ticker, f"Near settlement ({exit_price}¢) — letting Kalshi auto-settle")

        # Paper mode: simulate exit slippage (hitting the bid)
        # Don't apply slippage at settlement (1¢/99¢) — those are auto-resolved
        if pos.mode == "paper" and not near_settlement and "settlement" not in reason:
            exit_price = max(1, exit_price - self.PAPER_EXIT_SLIPPAGE)

        pnl_per_contract = (exit_price - pos.entry_price_cents) / 100.0
        pnl = pnl_per_contract * original_count

        # Record cooldown so we don't immediately re-enter this contract
        series_key = pos.ticker[:25]
        self._cooldowns[series_key] = time.time()

        # Prune old cooldowns (keep dict from growing forever)
        cutoff = time.time() - 600  # 10 min
        self._cooldowns = {k: v for k, v in self._cooldowns.items() if v > cutoff}

        self.db.close_trade(
            trade_id=trade_id,
            exit_price_cents=exit_price,
            pnl_usd=pnl,
            exit_reason=reason,
        )

        # Track consecutive losses for circuit breaker
        # Skip synced positions — losses from pre-existing positions on restart
        # shouldn't trigger the breaker and block fresh trading
        if pnl < 0 and not pos.synced:
            self.consecutive_losses += 1
            # Only trigger circuit breaker if NOT already paused —
            # otherwise each loss during the pause resets the timer and it never expires
            if self.consecutive_losses >= self.settings.circuit_breaker_losses and not self.is_paused:
                pause_sec = self.settings.circuit_breaker_pause_min * 60
                self.circuit_breaker_until = time.time() + pause_sec
                self._log_event(
                    "CIRCUIT_BREAKER",
                    pos.ticker,
                    f"{self.consecutive_losses} consecutive losses — pausing {self.settings.circuit_breaker_pause_min} min"
                )
                logger.warning(f"Circuit breaker triggered: {self.consecutive_losses} losses")
                # Reset counter so it takes another N losses to re-trigger after pause ends
                self.consecutive_losses = 0
            # Persist circuit breaker state to DB (survives restarts)
            self._persist_circuit_breaker()
        elif pnl >= 0:
            self.consecutive_losses = 0
            self._persist_circuit_breaker()

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

    def _persist_circuit_breaker(self):
        """Save circuit breaker state to DB so it survives restarts."""
        try:
            self.db.set_state("circuit_breaker_until", str(self.circuit_breaker_until))
            self.db.set_state("consecutive_losses", str(self.consecutive_losses))
        except Exception as e:
            logger.warning(f"Failed to persist circuit breaker state: {e}")

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
        cb_remaining = max(0, self.circuit_breaker_until - time.time()) if self.is_paused else 0
        return {
            "mode": self.mode,
            "open_positions": len(self.positions),
            "consecutive_losses": self.consecutive_losses,
            "circuit_breaker_active": self.is_paused,
            "circuit_breaker_remaining_sec": round(cb_remaining),
            "circuit_breaker_threshold": self.settings.circuit_breaker_losses,
            "kalshi_cash": self.kalshi_cash,
            "kalshi_portfolio": self.kalshi_portfolio,
            **stats,
            **today,
        }
