from __future__ import annotations

"""
Main Bot Orchestrator — launches and coordinates all async tasks.
"""

import asyncio
import gc
import time
import logging
from typing import Optional

from config.settings import Settings
from src.crypto.btc_engine import BTCEngine
from src.scanner.kalshi_scanner import KalshiScanner
from src.ai.validator import AIValidator
from src.execution.trader import Trader
from src.utils.database import Database
from src.utils.notifications import Notifier
from src.dashboard.server import DashboardServer

logger = logging.getLogger("bot")


class CryptoPredictionBot:
    """
    Main bot that coordinates all subsystems:
    - BTC + ETH price engines (Binance WebSocket)
    - Kalshi contract scanner (BTC + ETH)
    - AI consensus validator
    - Trade execution + position management
    - Dashboard server
    - Notifications
    - ADAPTIVE LEARNING ENGINE — learns from its own results
    """

    def __init__(self, settings: Settings):
        self.settings = settings
        self.db = Database(settings.db_path)
        self._adaptive_params: dict = {}  # updated by learning loop

        # BTC engine
        self.btc_engine = BTCEngine(settings)

        # ETH engine (reuses same class with different symbol/ws)
        self.eth_engine = None
        if getattr(settings, 'eth_enabled', False):
            self.eth_engine = BTCEngine(
                settings,
                symbol=settings.eth_binance_symbol,
                ws_url=settings.eth_binance_ws_url,
            )

        self.ai_validator = AIValidator(settings)
        self.trader = Trader(settings, self.db)
        self.notifier = Notifier(settings)

        # Engines dict for scanner to look up by asset
        self.engines = {"BTC": self.btc_engine}
        if self.eth_engine:
            self.engines["ETH"] = self.eth_engine

        self.scanner = KalshiScanner(settings, self.engines, self.db)

        # Wire momentum-burst callbacks → trigger immediate Kalshi scan
        self.btc_engine.on_burst(self._on_burst)
        if self.eth_engine:
            self.eth_engine.on_burst(self._on_burst)

        self.dashboard = DashboardServer(
            settings, self.btc_engine, self.scanner,
            self.trader, self.ai_validator, self.db,
            eth_engine=self.eth_engine,
        )
        self._running = False

    async def start(self):
        """Launch all concurrent tasks."""
        self._running = True
        logger.info(f"Starting Crypto Prediction Bot (mode={self.trader.mode})")
        logger.info(f"Wallet: ${self.settings.wallet_size_usd} | "
                     f"Max loss: ${self.settings.max_daily_loss_usd}/day | "
                     f"Min edge: {self.settings.min_edge_cents}¢")

        assets = list(self.engines.keys())
        logger.info(f"Assets: {', '.join(assets)}")

        # Verify live trading auth and sync existing positions
        if self.trader.mode == "live":
            auth_ok = await self.trader.verify_live_auth()
            if auth_ok:
                await self.trader.sync_kalshi_positions()
                await self.trader.fetch_kalshi_balance()
            else:
                logger.warning("Live auth failed — bot will run but orders will fail")

        # Bootstrap learning data before first scan
        await self._refresh_learning()

        tasks = [
            asyncio.create_task(self.btc_engine.start(), name="btc_engine"),
            asyncio.create_task(self.btc_engine.poll_funding_rate(), name="btc_funding"),
            asyncio.create_task(self.scanner.start(self._on_signal), name="kalshi_scanner"),
            asyncio.create_task(self.trader.monitor_positions(self._get_market_price), name="exit_monitor"),
            asyncio.create_task(self._stats_loop(), name="stats_loop"),
            asyncio.create_task(self._notification_loop(), name="notification_loop"),
            asyncio.create_task(self._learning_loop(), name="learning_loop"),
        ]

        # Add ETH engine tasks if enabled
        if self.eth_engine:
            tasks.append(asyncio.create_task(self.eth_engine.start(), name="eth_engine"))
            tasks.append(asyncio.create_task(self.eth_engine.poll_funding_rate(), name="eth_funding"))

        # Start dashboard
        await self.dashboard.start()
        logger.info("All systems running")

        try:
            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_EXCEPTION)
            for task in done:
                if task.exception():
                    logger.error(f"Task {task.get_name()} crashed: {task.exception()}")
        except asyncio.CancelledError:
            logger.info("Bot shutting down...")
        finally:
            await self.stop()

    async def stop(self):
        """Graceful shutdown."""
        self._running = False
        await self.btc_engine.stop()
        if self.eth_engine:
            await self.eth_engine.stop()
        await self.scanner.stop()
        await self.trader.stop()
        logger.info("Bot stopped")

    # ── Burst Handler ─────────────────────────────────────────

    def _on_burst(self, symbol: str, direction: str, pct_change: float):
        """Called when a price engine detects a momentum burst."""
        logger.info(f"BURST: {symbol} {direction} {pct_change:+.2f}%")
        self.trader._log_event(
            "BURST", symbol,
            f"{direction} {pct_change:+.2f}% — triggering immediate scan"
        )
        self.scanner.request_burst_scan()

    # ── Signal Handler ─────────────────────────────────────────

    async def _on_signal(self, signal):
        """
        Called by scanner when an edge is detected.
        Tiered gate:
          - 15¢+ edge → AUTO-TRADE (skip AI, the math is strong enough)
          - 8-15¢ edge → AI validation, 1/3 FOLLOW is enough
          - <8¢ edge → shouldn't get here (scanner filters), but require 2/3
        """
        logger.info(
            f"Signal: {signal.side.upper()} {signal.ticker} "
            f"edge={signal.edge_cents:.1f}¢ prob={signal.our_probability:.1%}"
        )

        # ── HIGH EDGE BYPASS: 15¢+ edge skips AI entirely ──
        if signal.edge_cents >= 15:
            self.trader._log_event(
                "AUTO_TRADE", signal.ticker,
                f"High edge {signal.edge_cents:.1f}¢ — bypassing AI gate"
            )
            logger.info(f"AUTO_TRADE: {signal.ticker} edge={signal.edge_cents:.1f}¢ — skipping AI")

            # Create a dummy consensus for the trade
            from src.ai.validator import ConsensusResult, ModelResponse
            consensus = ConsensusResult(
                action="FOLLOW",
                side=signal.side,
                confidence=0.8,
                models=[],
                follow_count=3,
                skip_count=0,
                active_count=3,
            )
            should_trade = True
        else:
            # ── NORMAL AI GATE ──
            consensus = await self.ai_validator.validate(signal)

            # Record AI decision
            models = {m.model: {"action": m.action, "confidence": m.confidence, "reasoning": m.reasoning}
                      for m in consensus.models}

            self.db.record_ai_decision(
                signal_id=0,
                ticker=signal.ticker,
                gpt=models.get("gpt-4.1-nano", models.get("gpt-4o-mini", {})),
                claude=models.get("claude-haiku-4.5", {}),
                gemini=models.get("gemini-2.5-flash-lite", models.get("gemini-2.0-flash", {})),
                consensus_action=consensus.action,
                consensus_side=consensus.side,
            )

            # Log AI consensus details to event log (visible on dashboard)
            model_votes = []
            for m in consensus.models:
                vote = f"{m.model}={m.action}"
                if m.confidence:
                    vote += f"({m.confidence:.0%})"
                model_votes.append(vote)
            votes_str = ", ".join(model_votes)

            self.trader._log_event(
                "AI", signal.ticker,
                f"{consensus.action} ({consensus.follow_count}/{consensus.active_count}) "
                f"edge={signal.edge_cents:.1f}¢ | {votes_str}"
            )

            should_trade = consensus.action == "FOLLOW"

            # AGGRESSIVE: even 1/3 FOLLOW is enough for 10¢+ edges
            if (not should_trade
                    and consensus.follow_count >= 1
                    and signal.edge_cents >= 10):
                should_trade = True
                self.trader._log_event(
                    "OVERRIDE", signal.ticker,
                    f"1/{consensus.active_count} FOLLOW + strong edge {signal.edge_cents:.1f}¢ — taking it"
                )

            # Fallback: if ALL models failed (0/0), auto-follow for decent edges
            if (not should_trade
                    and consensus.active_count == 0
                    and signal.edge_cents >= 8):
                should_trade = True
                self.trader._log_event(
                    "AI", signal.ticker,
                    f"AUTO-FOLLOW (all AI down, edge={signal.edge_cents:.1f}¢)"
                )

        if should_trade:
            trade_id = await self.trader.execute_trade(signal, consensus)
            if trade_id:
                await self.notifier.notify_trade(
                    ticker=signal.ticker,
                    side=signal.side,
                    entry=int(signal.kalshi_implied * 100),
                    count=1,
                    cost=0,
                    edge=signal.edge_cents,
                )
        else:
            self.db.record_skip(
                ticker=signal.ticker,
                side=signal.side,
                edge_cents=signal.edge_cents,
                skip_reason=f"AI consensus: {consensus.action} "
                            f"({consensus.follow_count}/{consensus.active_count})",
            )

    # ── Market Price Lookup ────────────────────────────────────

    async def _get_market_price(self, ticker: str) -> Optional[int]:
        """
        Get current market price for a contract.
        For LIVE mode: fetch real orderbook price from Kalshi API.
        Fallback: use model estimate with actual remaining time.
        """
        # Try live Kalshi price first (for live mode)
        if self.trader.mode == "live":
            live_price = await self._fetch_kalshi_price(ticker)
            if live_price is not None:
                return live_price

        # Fallback: model-based estimate
        try:
            strike = float(ticker.split("-T")[-1])
        except (ValueError, IndexError):
            return None

        # Determine which engine to use based on ticker
        if "KXETH" in ticker:
            engine = self.eth_engine or self.btc_engine
        else:
            engine = self.btc_engine

        price = engine.price
        if price == 0:
            return None

        # Use actual remaining time from the position, not a fixed 5 min
        # Parse close time from ticker to get real remaining minutes
        from src.scanner.kalshi_scanner import parse_ticker
        parsed = parse_ticker(ticker)
        if parsed:
            from datetime import datetime, timezone
            _, close_time = parsed
            remaining_min = max(0.5, (close_time - datetime.now(timezone.utc)).total_seconds() / 60.0)
        else:
            remaining_min = 5.0

        est = engine.estimate_probability(strike, minutes_to_close=remaining_min)
        simulated_price = int(est.probability * 100)
        return max(1, min(99, simulated_price))

    async def _fetch_kalshi_price(self, ticker: str) -> Optional[int]:
        """Fetch real market price from Kalshi orderbook."""
        try:
            client = self.trader._client
            if not client:
                return None
            path = f"/trade-api/v2/markets/{ticker}"
            headers = self.trader._sign_request("GET", path)
            resp = await client.get(
                f"{self.settings.kalshi_base_url}{path}",
                headers=headers,
            )
            resp.raise_for_status()
            data = resp.json().get("market", {})
            # Use yes_bid as the current tradeable price
            yes_bid = data.get("yes_bid_dollars", "")
            yes_ask = data.get("yes_ask_dollars", "")
            if yes_bid and yes_ask:
                bid = int(float(yes_bid) * 100)
                ask = int(float(yes_ask) * 100)
                mid = (bid + ask) // 2
                return max(1, min(99, mid))
            return None
        except Exception:
            return None  # silent fallback to model price

    # ── Background Tasks ───────────────────────────────────────

    async def _stats_loop(self):
        """Log stats, prune memory, run GC."""
        while self._running:
            await asyncio.sleep(self.settings.stats_interval_sec)

            # Refresh Kalshi balance every stats cycle
            await self.trader.fetch_kalshi_balance()

            stats = self.trader.to_dict()
            btc = self.btc_engine.to_dict()

            eth_str = ""
            if self.eth_engine:
                eth = self.eth_engine.to_dict()
                eth_str = f" | ETH=${eth['price']:,.2f}"

            logger.info(
                f"Stats | BTC=${btc['price']:,.2f}{eth_str} | "
                f"P&L=${stats['total_pnl']:+.2f} | "
                f"Win={stats['win_rate']:.0f}% | "
                f"Open={stats['open_positions']} | "
                f"Today={stats['today_trades']} trades"
            )

            gc.collect()

    async def _notification_loop(self):
        """Send periodic status notifications."""
        while self._running:
            await asyncio.sleep(self.settings.ntfy_status_interval_min * 60)

            stats = self.trader.to_dict()
            stats["price"] = self.btc_engine.price
            await self.notifier.notify_status(stats)

    # ── ADAPTIVE LEARNING ──────────────────────────────────────

    async def _learning_loop(self):
        """Refresh learning data every 10 minutes."""
        while self._running:
            await asyncio.sleep(600)  # 10 min
            try:
                await self._refresh_learning()
            except Exception as e:
                logger.error(f"Learning loop error: {e}")

    async def _refresh_learning(self):
        """Pull adaptive params from DB and push to all subsystems."""
        try:
            params = self.db.get_adaptive_params(mode=self.trader.mode)
            self._adaptive_params = params

            # Feed model weights to AI validator
            self.ai_validator.update_weights(
                model_weights=params.get("model_weights", {}),
                model_accuracy=params.get("model_accuracy", {}),
            )

            # Feed sizing multiplier to trader
            self.trader.adaptive_sizing_mult = params.get("sizing_multiplier", 1.0)
            self.trader.asset_preference = params.get("asset_preference", {})

            # Feed learned min edge to scanner
            if params.get("learning_active", False):
                learned_edge = params.get("learned_min_edge", 10.0)
                # Don't go below 6¢ even if data says so — need some minimum quality
                self.scanner.adaptive_min_edge = max(6.0, learned_edge)

            # Log bad hours
            bad_hours = params.get("bad_hours_utc", [])
            self.scanner.bad_hours = bad_hours

            status = "ACTIVE" if params.get("learning_active") else "WARMING UP"
            logger.info(
                f"Learning [{status}]: min_edge={params.get('learned_min_edge', '?')}¢ "
                f"sizing={params.get('sizing_multiplier', 1.0):.1f}x "
                f"recent_WR={params.get('recent_win_rate', '?')}% "
                f"bad_hours={bad_hours or 'none'} "
                f"weights={params.get('model_weights', {})}"
            )
        except Exception as e:
            logger.warning(f"Failed to refresh learning data: {e}")
