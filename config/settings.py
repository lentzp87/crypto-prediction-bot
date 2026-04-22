"""
Configuration — loads from .env with Pydantic validation.
Adjusted for $465 bankroll.
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
    trading_mode: str = Field(default="paper", description="'paper' for simulated trades, 'live' for real orders")

    @property
    def kalshi_base_url(self) -> str:
        if self.kalshi_env == "prod":
            return "https://api.elections.kalshi.com"
        return "https://demo-api.kalshi.co"

    # ── AI Model Keys (all optional) ───────────────────────────
    openai_api_key: Optional[str] = None
    anthropic_api_key: Optional[str] = None
    google_api_key: Optional[str] = None

    # ── Risk Management ($465 bankroll) ─────────────────────────
    wallet_size_usd: float = 465.0
    max_daily_loss_usd: float = 46.0        # ~10% of wallet
    max_positions: int = 8
    max_trades_per_day: int = 30
    min_edge_cents: float = 3.0
    max_single_trade_usd: float = 23.0      # ~5% of wallet
    circuit_breaker_losses: int = 3         # consecutive losses → pause
    circuit_breaker_pause_min: int = 30     # minutes to pause

    # ── Position Limits ────────────────────────────────────────
    max_same_strike: int = 2                # max positions on same strike
    max_same_window: int = 3                # max positions expiring in same 15-min window
    cooldown_seconds: int = 120             # between trades on same contract series

    # ── Sizing (Kelly-lite, scaled for $465) ───────────────────
    base_trade_size_usd: float = 9.0        # default per trade (~2% of $465)
    max_trade_size_usd: float = 23.0        # cap per trade (~5% of $465)

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
    kalshi_scan_interval_sec: int = 45
    funding_poll_interval_sec: int = 300    # 5 min
    exit_monitor_interval_sec: int = 30
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
