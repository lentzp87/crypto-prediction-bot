"""
Configuration — loads from .env with Pydantic validation.
V4.0 — capital preservation + friction-adjusted edge.
"""

from pydantic_settings import BaseSettings
from pydantic import Field
from typing import Optional


class Settings(BaseSettings):
    """All bot configuration, loaded from environment variables / .env file."""

    # ── Kalshi API ──────────────────────────────────────────────
    kalshi_api_key: str = ""
    kalshi_private_key_path: str = "./keys/kalshi.pem"
    kalshi_env: str = Field(default="prod", description="'demo' or 'prod' — which API for market data")
    trading_mode: str = Field(default="live", description="'paper' for simulated trades, 'live' for real orders")

    @property
    def kalshi_base_url(self) -> str:
        if self.kalshi_env == "prod":
            return "https://api.elections.kalshi.com"
        return "https://demo-api.kalshi.co"

    # ── AI Model Keys (all optional) ───────────────────────────
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # ── Risk Management ($524 bankroll) — FULL SEND ─────────────
    wallet_size_usd: float = 524.0
    max_daily_loss_usd: float = 75.0        # ~14% of bankroll max/day
    max_positions: int = 12                 # lots of at-bats
    max_trades_per_day: int = 9999           # uncapped
    min_edge_cents: float = 4.0             # 4¢ after friction (dailies = 5¢ raw)
    max_single_trade_usd: float = 40.0      # ~7.6% of wallet
    circuit_breaker_losses: int = 5         # tolerant
    circuit_breaker_pause_min: int = 2      # 2 min pause, get back in

    # ── Position Limits ────────────────────────────────────────
    max_same_strike: int = 3
    max_same_window: int = 5                # stack up on good windows
    cooldown_seconds: int = 60              # 1 min cooldown

    # ── Sizing (Kelly-lite, scaled for $524) ───────────────────
    base_trade_size_usd: float = 10.0       # bigger base
    max_trade_size_usd: float = 20.0        # up to ~3.8% of wallet

    # ── Asset-Specific Configs ─────────────────────────────────
    btc_min_edge_cents: float = 4.0         # after friction (dailies only 1¢ friction)
    btc_max_spread_cents: float = 8.0       # wider tolerance
    btc_jump_multiplier_cap: float = 2.0

    eth_min_edge_cents: float = 4.0         # same as BTC
    eth_max_spread_cents: float = 7.0       # wider tolerance
    eth_jump_multiplier_cap: float = 2.5
    daily_max_spread_cents: float = 25.0    # daily contracts: very wide (limit orders)
    max_contracts_per_trade: int = 15        # big positions

    # ── Take Profit / Stop Loss (cents) ────────────────────────
    # Tiered by entry price bucket
    # Low (15-39¢):  TP +20¢, SL -10¢
    # Mid (40-69¢):  TP +25¢, SL -12¢
    # High (70-85¢): ride to 97¢, SL -8¢

    # ── Trailing Stops ─────────────────────────────────────────
    trailing_activate_cents: int = 6        # activate at +6¢ profit
    trailing_offset_cents: int = 4          # trail at entry + 2¢ (6 - 4)

    # ── Stale Timeout ──────────────────────────────────────────
    stale_timeout_15m_minutes: int = 20     # exit 15-min contracts after 20 min

    # ── Binance.US (US-compliant endpoints) ──────────────────
    binance_ws_url: str = "wss://stream.binance.us:9443/ws/btcusd@kline_1m"
    binance_klines_url: str = "https://api.binance.us/api/v3/klines"
    binance_funding_url: str = ""  # Binance.US doesn't offer funding rate
    binance_symbol: str = "BTCUSD"  # Binance.US uses BTCUSD not BTCUSDT
    candle_buffer_size: int = 120           # 2 hours of 1-min candles

    # ── ETH Support ────────────────────────────────────────────
    eth_enabled: bool = True
    eth_binance_ws_url: str = "wss://stream.binance.us:9443/ws/ethusd@kline_1m"
    eth_binance_symbol: str = "ETHUSD"

    # ── Notifications ──────────────────────────────────────────
    ntfy_topic: Optional[str] = None
    ntfy_base_url: str = "https://ntfy.sh"
    ntfy_status_interval_min: int = 30

    # ── Dashboard ──────────────────────────────────────────────
    dashboard_port: int = 8080
    dashboard_host: str = "0.0.0.0"

    # ── Database ───────────────────────────────────────────────
    db_path: str = "data/crypto_bot.db"

    # ── Scanner Intervals ──────────────────────────────────────
    kalshi_scan_interval_sec: int = 30      # faster scanning (was 45)
    funding_poll_interval_sec: int = 300    # 5 min
    exit_monitor_interval_sec: int = 20     # faster exit checks (was 30)
    stats_interval_sec: int = 300           # 5 min

    # ── Memory Management (for Render) ─────────────────────────
    max_paper_trades_in_memory: int = 50
    max_event_log_size: int = 200
    gc_interval_sec: int = 300

    model_config = {
        "env_file": ".env",
        "env_file_encoding": "utf-8",
        "extra": "ignore",
    }


# Singleton
settings = Settings()
