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

        # Live Kalshi balance (updated periodically)
        self.kalshi_cash: float = 0.0
        self.kalshi_portfolio: float = 0.0
        self._balance_updated_at: float = 0

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

            for p in positions:
                ticker = p.get("ticker", "")
                pos_fp = float(p.get("position_fp", 0))
                if pos_fp == 0:
                    continue  # flat position, skip

                exposure = float(p.get("market_exposure_dollars", "0"))
                count = int(abs(pos_fp))
                side = "yes" if pos_fp > 0 else "no"

                # Calculate average entry price
                if count > 0:
                    entry_cents = int((exposure / count) * 100)
                else:
                    entry_cents = 50

                # Use a synthetic trade_id (negative to distinguish from DB trades)
                trade_id = -(synced + 1)

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
            return await self._paper_trade(signal, size_usd, count)
        else:
            return await self._live_trade(signal, size_usd, count)

    def _check_limits(self, signal) -> Optional[str]:
        """Check all risk limits. Returns block reason or None."""

        # Circuit breaker
        if self.is_paused:
            remaining = (self.circuit_breaker_until - time.time()) / 60
            return f"Circuit breaker active ({remaining:.0f} min remaining)"

        # ── Price sanity: reject bad risk/reward ──────────────
        entry_cents = int(signal.kalshi_implied * 100)

        # NO contracts above 45¢ are terrible risk/reward — you risk 45+¢ to win <55¢
        # The bot was buying NO at 60-88¢ and getting wiped out
        if signal.side == "no" and entry_cents > 45:
            return f"NO price too high ({entry_cents}¢ > 45¢ cap) — bad risk/reward"

        # YES contracts above 85¢ also bad — paying 85+¢ to win <15¢
        if signal.side == "yes" and entry_cents > 85:
            return f"YES price too high ({entry_cents}¢ > 85¢ cap) — bad risk/reward"

        # Very cheap contracts (<5¢) are almost certainly losing bets
        if entry_cents < 5:
            return f"Price too low ({entry_cents}¢) — likely to expire worthless"

        # ── Duplicate ticker check: NEVER buy a contract we already hold ──
        # This prevents position stacking across restarts
        for p in self.positions.values():
            if p.ticker == signal.ticker and p.side == signal.side:
                return f"Already holding {signal.ticker} {signal.side}"

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
        """Kelly-lite position sizing with max-loss cap."""
        base_size = self.settings.base_trade_size_usd  # $9

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

        # ── Max loss cap: never risk more than $15 on a single position ──
        # If all contracts go to zero, we lose cost_per_contract * count
        max_loss_usd = 15.0
        max_contracts_by_loss = max(1, int(max_loss_usd / cost_per_contract))
        count = min(count, max_contracts_by_loss)

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
        """Place a real limit order on Kalshi."""
        # Use bid price + 3¢ spread buffer to cross the bid-ask and actually get filled
        bid_cents = int(signal.kalshi_implied * 100)
        spread_buffer = 3  # cents to add to cross the spread
        entry_cents = min(bid_cents + spread_buffer, 95)  # cap at 95¢

        try:
            if not self._client:
                self._client = httpx.AsyncClient(timeout=15)

            path = "/trade-api/v2/portfolio/orders"
            headers = self._sign_request("POST", path)
            headers["Content-Type"] = "application/json"

            client_order_id = str(uuid.uuid4())

            order_body = {
                "ticker": signal.ticker,
                "action": "buy",
                "side": signal.side,
                "count": count,
                "type": "limit",
                "yes_price": entry_cents if signal.side == "yes" else None,
                "no_price": entry_cents if signal.side == "no" else None,
                "time_in_force": "fill_or_kill",  # immediate fill or cancel
                "client_order_id": client_order_id,
            }
            # Remove None price field
            order_body = {k: v for k, v in order_body.items() if v is not None}

            resp = await self._client.post(
                f"{self.settings.kalshi_base_url}{path}",
                headers=headers,
                json=order_body,
            )
            resp.raise_for_status()
            order_data = resp.json().get("order", {})

            order_id = order_data.get("order_id", "")
            status = order_data.get("status", "unknown")
            # fill_count_fp can be a string — coerce to int safely
            raw_filled = order_data.get("fill_count", 0) or order_data.get("fill_count_fp", 0)
            try:
                filled = int(float(raw_filled)) if raw_filled else 0
            except (ValueError, TypeError):
                filled = 0

            if status in ("filled", "resting") or filled > 0:
                actual_count = filled if filled > 0 else count
                actual_cost = actual_count * (entry_cents / 100.0)

                trade_id = self.db.record_trade(
                    ticker=signal.ticker,
                    side=signal.side,
                    entry_price_cents=entry_cents,
                    count=actual_count,
                    cost_usd=actual_cost,
                    mode="live",
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
                )
                self.positions[trade_id] = position

                self._log_event(
                    "LIVE_TRADE",
                    signal.ticker,
                    f"LIVE {signal.side.upper()} @ {entry_cents}¢ × {actual_count} "
                    f"(${actual_cost:.2f}) bid={bid_cents}¢ edge={signal.edge_cents:.1f}¢"
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
            exit_cents = max(1, pos.current_price_cents - 2)
            order_body = {
                "ticker": pos.ticker,
                "action": "sell",
                "side": pos.side,
                "count": pos.count,
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

            if status in ("filled",) or filled > 0:
                pnl = (exit_cents - pos.entry_price_cents) / 100.0 * pos.count
                logger.info(f"LIVE close filled: {pos.ticker} @ {exit_cents}¢ P&L=${pnl:+.2f}")
                return pnl
            else:
                logger.warning(f"Close order not filled for {pos.ticker}: {status}")
                return None

        except Exception as e:
            logger.error(f"Live close failed for {pos.ticker}: {e}")
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
        """Close a position and update DB. For live positions, sell on Kalshi first."""
        pos = self.positions.pop(trade_id, None)
        if not pos:
            return

        exit_price = pos.current_price_cents

        # For live positions, place a sell order on Kalshi
        # Skip selling if price is near settlement (99¢/1¢) — let Kalshi auto-settle
        near_settlement = exit_price >= 97 or exit_price <= 3
        if pos.mode == "live" and not near_settlement and "settlement" not in reason:
            result = await self._live_close(pos)
            if result is None:
                # Sell failed — put position back and retry next cycle
                self.positions[trade_id] = pos
                self._log_event("WARN", pos.ticker, f"Close failed ({reason}), will retry")
                return
        elif pos.mode == "live" and near_settlement:
            self._log_event("EXIT", pos.ticker, f"Near settlement ({exit_price}¢) — letting Kalshi auto-settle")

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
            "kalshi_cash": self.kalshi_cash,
            "kalshi_portfolio": self.kalshi_portfolio,
            **stats,
            **today,
        }
