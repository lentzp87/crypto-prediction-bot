"""
Dashboard API Server — aiohttp serving REST endpoints + static HTML.
"""

import os
import time
import logging
import json

from aiohttp import web

logger = logging.getLogger("dashboard")


class DashboardServer:
    """aiohttp-based dashboard with REST API endpoints."""

    def __init__(self, settings, btc_engine, kalshi_scanner, trader, ai_validator, database, eth_engine=None):
        self.settings = settings
        self.btc_engine = btc_engine
        self.eth_engine = eth_engine
        self.scanner = kalshi_scanner
        self.trader = trader
        self.ai_validator = ai_validator
        self.db = database
        self.start_time = time.time()
        self.app = web.Application()
        self._setup_routes()

    def _setup_routes(self):
        self.app.router.add_get("/", self._serve_dashboard)
        self.app.router.add_get("/api/stats", self._api_stats)
        self.app.router.add_get("/api/positions", self._api_positions)
        self.app.router.add_get("/api/btc", self._api_btc)
        self.app.router.add_get("/api/events", self._api_events)
        self.app.router.add_get("/api/pnl_history", self._api_pnl_history)
        self.app.router.add_get("/api/closed", self._api_closed)
        self.app.router.add_get("/api/status", self._api_status)
        self.app.router.add_get("/api/consensus", self._api_consensus)

    async def start(self):
        runner = web.AppRunner(self.app)
        await runner.setup()
        site = web.TCPSite(
            runner,
            self.settings.dashboard_host,
            self.settings.dashboard_port,
        )
        await site.start()
        logger.info(f"Dashboard running on http://{self.settings.dashboard_host}:{self.settings.dashboard_port}")

    # ── HTML Dashboard ─────────────────────────────────────────

    async def _serve_dashboard(self, request):
        html_path = os.path.join(os.path.dirname(__file__), "dashboard.html")
        try:
            with open(html_path) as f:
                return web.Response(text=f.read(), content_type="text/html")
        except FileNotFoundError:
            return web.Response(text="<h1>Dashboard HTML not found</h1>", content_type="text/html")

    # ── API Endpoints ──────────────────────────────────────────

    async def _api_stats(self, request):
        """Win rate, P&L, total trades, open positions."""
        data = self.trader.to_dict()
        return web.json_response(data)

    async def _api_positions(self, request):
        """Open positions with live prices."""
        return web.json_response(self.trader.get_positions_dict())

    async def _api_btc(self, request):
        """BTC + ETH engine stats: price, indicators, candle count."""
        data = self.btc_engine.to_dict()
        data["scanner"] = self.scanner.to_dict()
        if self.eth_engine:
            data["eth"] = self.eth_engine.to_dict()
        return web.json_response(data)

    async def _api_events(self, request):
        """Rolling event log."""
        return web.json_response(self.trader.get_events())

    async def _api_pnl_history(self, request):
        """Time-series P&L for charting."""
        return web.json_response(self.db.get_pnl_history())

    async def _api_closed(self, request):
        """Closed trades history."""
        return web.json_response(self.db.get_closed_trades(mode=self.trader.mode))

    async def _api_consensus(self, request):
        """Consensus performance: 2/3 vs 3/3 win rates."""
        return web.json_response(self.db.get_consensus_stats())

    async def _api_status(self, request):
        """Bot health: uptime, memory, task status."""
        import os

        uptime = time.time() - self.start_time
        hours = int(uptime // 3600)
        minutes = int((uptime % 3600) // 60)

        # Memory usage without psutil dependency
        try:
            import resource
            mem_mb = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / (1024 * 1024), 1)
        except Exception:
            mem_mb = 0.0

        return web.json_response({
            "uptime": f"{hours}h {minutes}m",
            "uptime_seconds": uptime,
            "memory_mb": mem_mb,
            "mode": self.trader.mode,
            "btc_ready": self.btc_engine.ready,
            "btc_price": self.btc_engine.price,
            "open_positions": len(self.trader.positions),
            "ai_validator": self.ai_validator.to_dict(),
            "wallet_size": self.settings.wallet_size_usd,
        })
