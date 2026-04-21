"""
Crypto Prediction Bot — Entry Point.

Usage:
    python main.py

Requires .env file with configuration (see .env.example).
"""

import asyncio
import logging
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config.settings import settings
from src.bot import CryptoPredictionBot


def setup_logging():
    """Configure logging for all modules."""
    log_format = "%(asctime)s | %(name)-15s | %(levelname)-7s | %(message)s"
    date_format = "%H:%M:%S"

    logging.basicConfig(
        level=logging.INFO,
        format=log_format,
        datefmt=date_format,
        handlers=[
            logging.StreamHandler(sys.stdout),
        ],
    )

    # Quiet down noisy libraries
    logging.getLogger("httpx").setLevel(logging.WARNING)
    logging.getLogger("websockets").setLevel(logging.WARNING)
    logging.getLogger("aiohttp").setLevel(logging.WARNING)
    logging.getLogger("sqlalchemy").setLevel(logging.WARNING)


def main():
    setup_logging()
    logger = logging.getLogger("main")

    logger.info("=" * 60)
    logger.info("  Crypto Prediction Bot v1.0.0")
    logger.info(f"  Mode: {'PAPER' if settings.kalshi_env == 'demo' else 'LIVE'}")
    logger.info(f"  Wallet: ${settings.wallet_size_usd:,.0f}")
    logger.info(f"  Max daily loss: ${settings.max_daily_loss_usd:,.0f}")
    logger.info(f"  Min edge: {settings.min_edge_cents}¢")
    logger.info(f"  Dashboard: http://localhost:{settings.dashboard_port}")
    logger.info("=" * 60)

    # Check AI model availability
    models = []
    if settings.openai_api_key:
        models.append("GPT-4o-mini")
    if settings.anthropic_api_key:
        models.append("Claude Haiku")
    if settings.google_api_key:
        models.append("Gemini Flash")

    if not models:
        logger.warning("No AI model keys configured! Bot will skip AI validation.")
    else:
        logger.info(f"  AI Models: {', '.join(models)}")

    if not settings.kalshi_api_key:
        logger.warning("No Kalshi API key — scanner will use unauthenticated access (limited)")

    # Create and run bot
    bot = CryptoPredictionBot(settings)

    try:
        asyncio.run(bot.start())
    except KeyboardInterrupt:
        logger.info("Shutting down (Ctrl+C)...")
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
