from __future__ import annotations

"""
BTC Price Engine — Binance WebSocket + Technical Indicators + Probability Model.

Streams real-time 1-min BTC/USDT candles, computes indicators, and estimates
probabilities for Kalshi KXBTC15M contracts.
"""

import asyncio
import json
import time
import logging
import math
from dataclasses import dataclass, field
from typing import Optional
from collections import deque
from statistics import NormalDist

import httpx
import websockets
import numpy as np

logger = logging.getLogger("btc_engine")

# Standard normal CDF
NORM = NormalDist(0, 1)


@dataclass
class Candle:
    """One-minute OHLCV candle."""
    timestamp: float
    open: float
    high: float
    low: float
    close: float
    volume: float
    is_closed: bool = True


@dataclass
class Indicators:
    """Computed technical indicators snapshot."""
    price: float = 0.0
    rsi: float = 50.0
    vwap: float = 0.0
    bollinger_upper: float = 0.0
    bollinger_lower: float = 0.0
    bollinger_mid: float = 0.0
    momentum: float = 0.0          # 5-candle % change
    ema9: float = 0.0
    ema21: float = 0.0
    volatility_15m: float = 0.0    # OLD simple vol (kept for dashboard display)
    funding_rate: float = 0.0
    candle_count: int = 0
    last_update: float = 0.0

    # ── HAR volatility model ──────────────────────────────────
    har_vol_forecast: float = 0.0   # HAR-predicted vol for next period (%)
    rv_short: float = 0.0           # realized vol, last 5 min
    rv_medium: float = 0.0          # realized vol, last 30 min
    rv_long: float = 0.0            # realized vol, last 120 min

    # ── Jump detection ─────────────────────────────────────────
    jump_active: bool = False       # True if recent jump detected
    jump_intensity: float = 0.0     # how many std devs the jump was
    jumps_last_hour: int = 0        # count of jumps in last 60 candles
    jump_vol_multiplier: float = 1.0  # vol inflation factor from jumps

    @property
    def bb_position(self) -> str:
        """Where price sits relative to Bollinger Bands."""
        if self.bollinger_upper == 0:
            return "unknown"
        if self.price >= self.bollinger_upper:
            return "above_upper"
        elif self.price <= self.bollinger_lower:
            return "below_lower"
        elif self.price > self.bollinger_mid:
            return "upper_half"
        else:
            return "lower_half"


@dataclass
class ProbabilityEstimate:
    """Probability that BTC will be above strike at close time."""
    probability: float
    base_prob: float
    momentum_adj: float
    rsi_adj: float
    bb_adj: float
    funding_adj: float
    z_score: float
    distance: float
    scaled_vol: float


class BTCEngine:
    """
    Core BTC analysis engine.

    - Connects to Binance WebSocket for real-time 1-min candles
    - Maintains a rolling buffer of candles
    - Computes technical indicators on each new candle
    - Estimates probabilities for strike-price contracts
    """

    def __init__(self, settings, symbol: str = None, ws_url: str = None):
        self.settings = settings
        self.symbol = symbol or getattr(settings, 'binance_symbol', 'BTCUSD')
        self.ws_url = ws_url or settings.binance_ws_url
        self.candles: deque[Candle] = deque(maxlen=settings.candle_buffer_size)
        self.current_candle: Optional[Candle] = None
        self.indicators = Indicators()
        self.funding_rate: float = 0.0
        self._ws = None
        self._running = False
        self._reconnect_delay = 5

        # Momentum burst detection
        self._burst_callbacks: list = []
        self._recent_prices: deque = deque(maxlen=30)  # last 30 ticks
        self._last_burst_time: float = 0
        self.burst_threshold_pct: float = 0.15  # 0.15% move in short window = burst

    def on_burst(self, callback):
        """Register a callback for momentum bursts."""
        self._burst_callbacks.append(callback)

    # ── Public API ─────────────────────────────────────────────

    @property
    def price(self) -> float:
        """Current BTC price."""
        if self.current_candle:
            return self.current_candle.close
        if self.candles:
            return self.candles[-1].close
        return 0.0

    @property
    def ready(self) -> bool:
        """True when we have enough candles for all indicators."""
        return len(self.candles) >= 21  # need at least 21 for EMA21

    def estimate_probability(self, strike_price: float, minutes_to_close: float) -> ProbabilityEstimate:
        """
        Estimate probability that BTC will be ABOVE strike_price
        at close time (minutes_to_close minutes from now).

        Uses HAR volatility forecast + jump-adjusted normal CDF.

        The HAR model forecasts volatility using three time horizons:
          - Short (5 min): captures current regime
          - Medium (30 min): captures session trend
          - Long (120 min): captures baseline level

        Jump detection inflates vol when large moves are detected,
        because normal CDF systematically underprices tail risk
        during jump events (BTC averages ~3.5 jumps/day).

        NO indicator adjustments (momentum/RSI/BB) — those double-count
        information already in the current price.
        """
        current = self.price
        har_vol = self.indicators.har_vol_forecast
        old_vol = self.indicators.volatility_15m

        # Use HAR forecast if available, fall back to simple vol
        base_vol = har_vol if har_vol > 0 else old_vol

        if current == 0 or base_vol == 0:
            return ProbabilityEstimate(
                probability=-1.0, base_prob=0,
                momentum_adj=0, rsi_adj=0, bb_adj=0, funding_adj=0,
                z_score=0, distance=0, scaled_vol=0,
            )

        # Distance from strike
        distance = current - strike_price

        # Apply jump vol multiplier
        # During jump events, inflate vol to account for fat tails
        jump_mult = self.indicators.jump_vol_multiplier
        adjusted_vol = base_vol * jump_mult

        # Scale volatility to time remaining
        # HAR forecast is calibrated to ~5-min RV, scale to contract horizon
        time_scale = math.sqrt(max(minutes_to_close, 0.5) / 5.0)
        scaled_vol = adjusted_vol * time_scale
        price_std = current * (scaled_vol / 100.0)

        if price_std < 0.01:
            price_std = 0.01

        # Z-score and base probability
        z_score = distance / price_std
        base_prob = NORM.cdf(z_score)

        # No indicator adjustments — pure vol model
        momentum_adj = 0.0
        rsi_adj = 0.0
        bb_adj = 0.0
        funding_adj = 0.0

        probability = max(0.01, min(0.99, base_prob))

        # Log jump info when it matters
        if self.indicators.jump_active:
            logger.info(
                f"JUMP ACTIVE: intensity={self.indicators.jump_intensity:.1f}σ, "
                f"multiplier={jump_mult:.2f}x, "
                f"jumps_1h={self.indicators.jumps_last_hour}, "
                f"vol {base_vol:.3f}%→{adjusted_vol:.3f}%"
            )

        return ProbabilityEstimate(
            probability=probability,
            base_prob=base_prob,
            momentum_adj=momentum_adj,
            rsi_adj=rsi_adj,
            bb_adj=bb_adj,
            funding_adj=funding_adj,
            z_score=z_score,
            distance=distance,
            scaled_vol=scaled_vol,
        )

    # ── Binance WebSocket ──────────────────────────────────────

    async def start(self):
        """Seed historical candles then connect WebSocket."""
        self._running = True
        await self._seed_candles()
        self._compute_indicators()
        logger.info(f"[{self.symbol}] Seeded {len(self.candles)} candles, ${self.price:,.2f}")
        await self._ws_loop()

    async def stop(self):
        """Gracefully disconnect."""
        self._running = False
        if self._ws:
            await self._ws.close()

    async def _seed_candles(self):
        """Fetch recent 1-min candles from Binance REST API."""
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.get(
                    self.settings.binance_klines_url,
                    params={"symbol": self.symbol, "interval": "1m", "limit": 60},
                )
                resp.raise_for_status()
                data = resp.json()

            for k in data:
                candle = Candle(
                    timestamp=k[0] / 1000.0,
                    open=float(k[1]),
                    high=float(k[2]),
                    low=float(k[3]),
                    close=float(k[4]),
                    volume=float(k[5]),
                    is_closed=True,
                )
                self.candles.append(candle)

        except Exception as e:
            logger.error(f"Failed to seed candles: {e}")

    async def _ws_loop(self):
        """Connect to Binance WebSocket with auto-reconnect."""
        while self._running:
            try:
                async with websockets.connect(
                    self.ws_url,
                    ping_interval=30,
                    ping_timeout=10,
                ) as ws:
                    self._ws = ws
                    self._reconnect_delay = 5
                    logger.info(f"[{self.symbol}] WebSocket connected")

                    async for msg in ws:
                        if not self._running:
                            break
                        self._handle_kline(json.loads(msg))

            except websockets.ConnectionClosed:
                logger.warning(f"WS disconnected, reconnecting in {self._reconnect_delay}s...")
            except Exception as e:
                logger.error(f"WS error: {e}, reconnecting in {self._reconnect_delay}s...")
                self._reconnect_delay = min(self._reconnect_delay + 5, 30)

            if self._running:
                await asyncio.sleep(self._reconnect_delay)

    def _handle_kline(self, data: dict):
        """Process incoming kline message from Binance."""
        try:
            k = data.get("k", {})
            candle = Candle(
                timestamp=k["t"] / 1000.0,
                open=float(k["o"]),
                high=float(k["h"]),
                low=float(k["l"]),
                close=float(k["c"]),
                volume=float(k["v"]),
                is_closed=k.get("x", False),
            )

            if candle.is_closed:
                # Finalized candle — append to buffer
                self.candles.append(candle)
                self.current_candle = None
                self._compute_indicators()
            else:
                # In-progress — update current candle in place
                self.current_candle = candle
                self.indicators.price = candle.close
                self.indicators.last_update = time.time()

            # Burst detection on every tick
            self._check_burst(candle.close)

        except (KeyError, ValueError) as e:
            logger.error(f"Bad kline data: {e}")

    # ── Funding Rate ───────────────────────────────────────────

    async def poll_funding_rate(self):
        """Poll Binance perpetual funding rate periodically."""
        while self._running:
            if not self.settings.binance_funding_url:
                # Binance.US doesn't offer funding rate — skip silently
                await asyncio.sleep(self.settings.funding_poll_interval_sec)
                continue
            try:
                async with httpx.AsyncClient(timeout=10) as client:
                    resp = await client.get(
                        self.settings.binance_funding_url,
                        params={"symbol": getattr(self.settings, 'binance_symbol', 'BTCUSD')},
                    )
                    resp.raise_for_status()
                    data = resp.json()
                    self.funding_rate = float(data.get("lastFundingRate", 0))
                    self.indicators.funding_rate = self.funding_rate
                    logger.debug(f"Funding rate: {self.funding_rate:.6f}")
            except Exception as e:
                logger.warning(f"Funding rate fetch failed: {e}")

            await asyncio.sleep(self.settings.funding_poll_interval_sec)

    # ── Momentum Burst Detection ──────────────────────────────

    def _check_burst(self, price: float):
        """
        Detect rapid price moves. If price moved > threshold in recent ticks,
        fire burst callbacks (triggers immediate contract scan).
        """
        now = time.time()
        self._recent_prices.append((now, price))

        # Need at least 5 ticks and 30s cooldown between bursts
        if len(self._recent_prices) < 5:
            return
        if now - self._last_burst_time < 30:
            return

        # Check price move over last ~60 seconds of ticks
        oldest_time, oldest_price = self._recent_prices[0]
        if oldest_price == 0 or (now - oldest_time) < 10:
            return

        pct_change = abs(price - oldest_price) / oldest_price * 100

        if pct_change >= self.burst_threshold_pct:
            self._last_burst_time = now
            direction = "UP" if price > oldest_price else "DOWN"
            logger.info(
                f"[{self.symbol}] MOMENTUM BURST: {direction} {pct_change:.3f}% "
                f"(${oldest_price:,.2f} → ${price:,.2f})"
            )
            for cb in self._burst_callbacks:
                try:
                    cb(self.symbol, direction, pct_change)
                except Exception as e:
                    logger.error(f"Burst callback error: {e}")

    # ── Indicator Computation ──────────────────────────────────

    def _compute_indicators(self):
        """Recompute all technical indicators from candle buffer."""
        if len(self.candles) < 2:
            return

        closes = np.array([c.close for c in self.candles])
        volumes = np.array([c.volume for c in self.candles])
        highs = np.array([c.high for c in self.candles])
        lows = np.array([c.low for c in self.candles])

        ind = self.indicators
        ind.price = closes[-1]
        ind.candle_count = len(self.candles)
        ind.last_update = time.time()

        # RSI (14-period)
        if len(closes) >= 15:
            ind.rsi = self._calc_rsi(closes, 14)

        # VWAP (session)
        if len(closes) >= 2:
            typical = (highs + lows + closes) / 3.0
            cum_tp_vol = np.cumsum(typical * volumes)
            cum_vol = np.cumsum(volumes)
            if cum_vol[-1] > 0:
                ind.vwap = cum_tp_vol[-1] / cum_vol[-1]

        # Bollinger Bands (20-period, 2 std)
        if len(closes) >= 20:
            window = closes[-20:]
            ind.bollinger_mid = float(np.mean(window))
            std = float(np.std(window, ddof=1))
            ind.bollinger_upper = ind.bollinger_mid + 2 * std
            ind.bollinger_lower = ind.bollinger_mid - 2 * std

        # Momentum (5-candle % change)
        if len(closes) >= 6:
            prev = closes[-6]
            if prev > 0:
                ind.momentum = ((closes[-1] - prev) / prev) * 100.0

        # EMA 9 and EMA 21
        if len(closes) >= 9:
            ind.ema9 = self._calc_ema(closes, 9)
        if len(closes) >= 21:
            ind.ema21 = self._calc_ema(closes, 21)

        # ── HAR Volatility Model ──────────────────────────────────
        # Heterogeneous Autoregressive: decomposes realized vol into
        # short (5-min), medium (30-min), long (120-min) components.
        # Forecast = w0 + w1*RV_short + w2*RV_medium + w3*RV_long
        #
        # This captures the empirical fact that vol at different
        # timescales has different persistence and predictive power.
        all_returns = np.diff(np.log(closes)) if len(closes) >= 2 else np.array([])

        if len(all_returns) >= 5:
            # Realized variance components (sum of squared log returns)
            rv_5 = float(np.sum(all_returns[-5:] ** 2))
            ind.rv_short = math.sqrt(rv_5) * 100.0  # as %

            rv_30 = float(np.sum(all_returns[-min(30, len(all_returns)):] ** 2))
            n_30 = min(30, len(all_returns))
            ind.rv_medium = math.sqrt(rv_30 * 5 / n_30) * 100.0  # scaled to 5-min equiv

            rv_120 = float(np.sum(all_returns ** 2))
            n_120 = len(all_returns)
            ind.rv_long = math.sqrt(rv_120 * 5 / n_120) * 100.0  # scaled to 5-min equiv

            # HAR forecast: weighted combination
            # Weights tuned for crypto's fast-decaying autocorrelation:
            # - Heavy on short-term (most recent vol is most predictive)
            # - Medium on medium-term (30-min regime)
            # - Light on long-term (baseline level)
            w_short = 0.5
            w_medium = 0.3
            w_long = 0.2
            har_forecast = (w_short * ind.rv_short +
                           w_medium * ind.rv_medium +
                           w_long * ind.rv_long)
            ind.har_vol_forecast = har_forecast

            # Keep old volatility_15m for backward compat / dashboard
            if len(all_returns) >= 15:
                std_1m = float(np.std(all_returns[-15:], ddof=1))
                ind.volatility_15m = std_1m * math.sqrt(15) * 100.0
            else:
                ind.volatility_15m = har_forecast

        elif len(all_returns) >= 2:
            # Fallback: simple vol when not enough data for HAR
            std_1m = float(np.std(all_returns, ddof=1))
            ind.volatility_15m = std_1m * math.sqrt(15) * 100.0
            ind.har_vol_forecast = ind.volatility_15m

        # ── Jump Detection ─────────────────────────────────────────
        # A "jump" = 1-min return > 3 standard deviations.
        # When jumps are present, normal CDF underestimates tail risk,
        # so we inflate the vol estimate.
        if len(all_returns) >= 20:
            rolling_std = float(np.std(all_returns[-20:], ddof=1))
            jump_threshold = 3.0 * rolling_std if rolling_std > 0 else 1e-6

            # Check last candle for jump
            last_return = abs(float(all_returns[-1]))
            if last_return > jump_threshold and rolling_std > 0:
                ind.jump_active = True
                ind.jump_intensity = last_return / rolling_std
            else:
                ind.jump_active = False
                ind.jump_intensity = 0.0

            # Count jumps in available history (up to 60 candles)
            lookback = min(60, len(all_returns))
            recent = all_returns[-lookback:]
            jump_count = int(np.sum(np.abs(recent) > jump_threshold))
            ind.jumps_last_hour = jump_count

            # Vol inflation: more jumps = fatter tails = higher effective vol
            # Base: 1.0 (no jumps). Each recent jump adds ~15% vol.
            # Active jump (just happened) adds extra 30%.
            multiplier = 1.0
            if jump_count > 0:
                multiplier += 0.15 * min(jump_count, 5)  # cap at 5 jumps worth
            if ind.jump_active:
                multiplier += 0.30
            ind.jump_vol_multiplier = min(multiplier, 2.5)  # cap at 2.5x
        else:
            ind.jump_active = False
            ind.jump_intensity = 0.0
            ind.jumps_last_hour = 0
            ind.jump_vol_multiplier = 1.0

        # Funding rate (updated by separate poller)
        ind.funding_rate = self.funding_rate

    @staticmethod
    def _calc_rsi(closes: np.ndarray, period: int = 14) -> float:
        """Wilder's RSI."""
        deltas = np.diff(closes[-(period + 1):])
        gains = np.where(deltas > 0, deltas, 0.0)
        losses = np.where(deltas < 0, -deltas, 0.0)

        avg_gain = np.mean(gains[:period])
        avg_loss = np.mean(losses[:period])

        # Smooth with Wilder's method for remaining
        for i in range(period, len(gains)):
            avg_gain = (avg_gain * (period - 1) + gains[i]) / period
            avg_loss = (avg_loss * (period - 1) + losses[i]) / period

        if avg_loss == 0:
            return 100.0
        rs = avg_gain / avg_loss
        return 100.0 - (100.0 / (1.0 + rs))

    @staticmethod
    def _calc_ema(closes: np.ndarray, period: int) -> float:
        """Exponential Moving Average."""
        multiplier = 2.0 / (period + 1)
        ema = float(closes[0])
        for price in closes[1:]:
            ema = (float(price) - ema) * multiplier + ema
        return ema

    # ── Serialization (for dashboard) ──────────────────────────

    def to_dict(self) -> dict:
        """Snapshot for API/dashboard."""
        ind = self.indicators
        return {
            "price": ind.price,
            "rsi": round(ind.rsi, 1),
            "vwap": round(ind.vwap, 2),
            "bollinger_upper": round(ind.bollinger_upper, 2),
            "bollinger_lower": round(ind.bollinger_lower, 2),
            "bollinger_mid": round(ind.bollinger_mid, 2),
            "bb_position": ind.bb_position,
            "momentum": round(ind.momentum, 3),
            "ema9": round(ind.ema9, 2),
            "ema21": round(ind.ema21, 2),
            "volatility_15m": round(ind.volatility_15m, 4),
            "funding_rate": round(ind.funding_rate, 6),
            "candle_count": ind.candle_count,
            "last_update": ind.last_update,
            "ready": self.ready,
            # HAR volatility model
            "har_vol_forecast": round(ind.har_vol_forecast, 4),
            "rv_short": round(ind.rv_short, 4),
            "rv_medium": round(ind.rv_medium, 4),
            "rv_long": round(ind.rv_long, 4),
            # Jump detection
            "jump_active": ind.jump_active,
            "jump_intensity": round(ind.jump_intensity, 2),
            "jumps_last_hour": ind.jumps_last_hour,
            "jump_vol_multiplier": round(ind.jump_vol_multiplier, 2),
        }
