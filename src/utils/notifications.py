"""
Notification system via ntfy.sh with rate limiting.

Rate limits:
- Per-category cooldown: 5 min
- Token bucket: 1 token/60s, burst of 5
- Status updates: every 30 min
"""

import asyncio
import time
import logging
from typing import Optional

import httpx

logger = logging.getLogger("notifications")


class Notifier:
    """Push notifications via ntfy.sh with rate limiting."""

    def __init__(self, settings):
        self.topic = settings.ntfy_topic
        self.base_url = settings.ntfy_base_url
        self.enabled = bool(self.topic)

        # Rate limiting
        self._category_cooldowns: dict[str, float] = {}
        self._cooldown_sec = 300  # 5 min per category
        self._tokens = 5.0
        self._max_tokens = 5.0
        self._token_rate = 1.0 / 60.0  # 1 per 60 seconds
        self._last_token_refill = time.time()

    async def send(self, title: str, message: str, category: str = "general",
                   priority: int = 3, tags: Optional[list[str]] = None):
        """Send a notification if rate limits allow."""
        if not self.enabled:
            return

        # Check category cooldown
        now = time.time()
        last_sent = self._category_cooldowns.get(category, 0)
        if now - last_sent < self._cooldown_sec:
            return

        # Check token bucket
        self._refill_tokens()
        if self._tokens < 1.0:
            return

        self._tokens -= 1.0
        self._category_cooldowns[category] = now

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                headers = {
                    "Title": title,
                    "Priority": str(priority),
                }
                if tags:
                    headers["Tags"] = ",".join(tags)

                await client.post(
                    f"{self.base_url}/{self.topic}",
                    content=message,
                    headers=headers,
                )
                logger.debug(f"Notification sent: {title}")
        except Exception as e:
            logger.warning(f"Notification failed: {e}")

    async def notify_trade(self, ticker: str, side: str, entry: int,
                           count: int, cost: float, edge: float):
        """Notify on trade execution."""
        await self.send(
            title=f"Trade: {side.upper()} {ticker[-15:]}",
            message=f"Entry: {entry}¢ × {count} = ${cost:.2f}\nEdge: {edge:.1f}¢",
            category="trade",
            priority=4,
            tags=["chart_with_upwards_trend", "moneybag"],
        )

    async def notify_exit(self, ticker: str, reason: str, pnl: float):
        """Notify on position exit."""
        emoji = "white_check_mark" if pnl > 0 else "x"
        await self.send(
            title=f"Exit: {ticker[-15:]} P&L ${pnl:+.2f}",
            message=f"Reason: {reason}",
            category="exit",
            priority=3,
            tags=[emoji],
        )

    async def notify_status(self, stats: dict):
        """Periodic status update."""
        await self.send(
            title="Bot Status Update",
            message=(
                f"BTC: ${stats.get('price', 0):,.2f}\n"
                f"P&L Today: ${stats.get('today_pnl', 0):+.2f}\n"
                f"Open: {stats.get('open_positions', 0)} | "
                f"Trades: {stats.get('today_trades', 0)}\n"
                f"Win Rate: {stats.get('win_rate', 0):.1f}%"
            ),
            category="status",
            priority=2,
            tags=["robot"],
        )

    async def notify_circuit_breaker(self, losses: int, pause_min: int):
        """Notify on circuit breaker activation."""
        await self.send(
            title="Circuit Breaker Activated",
            message=f"{losses} consecutive losses — pausing {pause_min} min",
            category="alert",
            priority=5,
            tags=["warning", "rotating_light"],
        )

    def _refill_tokens(self):
        now = time.time()
        elapsed = now - self._last_token_refill
        self._tokens = min(self._max_tokens, self._tokens + elapsed * self._token_rate)
        self._last_token_refill = now
