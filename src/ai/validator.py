from __future__ import annotations

"""
AI Consensus Validator — Triple-model gate for trade signals.

Uses GPT-4o-mini + Claude Haiku + Gemini Flash.
2-of-3 must agree to FOLLOW a signal. Adapts when models fail or keys are missing.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass
from typing import Optional

import httpx

logger = logging.getLogger("ai_validator")


@dataclass
class ModelResponse:
    """Response from a single AI model."""
    model: str
    action: str         # "FOLLOW", "SKIP", "FAILED", "NO KEY"
    confidence: float
    side: str
    reasoning: str
    risk_level: str
    latency_ms: float


@dataclass
class ConsensusResult:
    """Combined result from all models."""
    action: str             # "FOLLOW" or "SKIP"
    side: str
    confidence: float       # average confidence of agreeing models
    models: list[ModelResponse]
    follow_count: int
    skip_count: int
    active_count: int


VALIDATION_PROMPT = """You are a crypto prediction market analyst. Evaluate this BTC contract:

Contract: {ticker} — Will BTC be above ${strike} at {close_time}?
Current BTC: ${current_price}
Time to close: {minutes:.1f} minutes
Our probability estimate: {probability:.1f}%
Kalshi {side} price: {price}¢ (implied {implied:.1f}%)
Edge detected: {edge:.1f}¢

Technical indicators:
- RSI: {rsi} | Momentum: {momentum}% | Volatility: {vol}%
- Bollinger: {bb_position} | Funding: {funding_rate}%
- VWAP: ${vwap} | EMA9: ${ema9} | EMA21: ${ema21}

Should we BUY {side} on this contract?

Respond with ONLY valid JSON (no markdown, no code blocks):
{{"action": "FOLLOW" or "SKIP", "confidence": 0.0-1.0, "side": "{side}", "reasoning": "brief explanation", "risk_level": "low" or "medium" or "high"}}"""


class AIValidator:
    """Manages triple-model consensus validation for trade signals."""

    def __init__(self, settings):
        self.settings = settings
        self._openai_key = settings.openai_api_key
        self._anthropic_key = settings.anthropic_api_key
        self._google_key = settings.google_api_key
        self.total_validations = 0
        self.total_cost_estimate = 0.0

    def _build_prompt(self, signal) -> str:
        """Build the validation prompt from a TradeSignal."""
        ind = signal.indicators
        return VALIDATION_PROMPT.format(
            ticker=signal.ticker,
            strike=f"{signal.strike_price:,.2f}",
            close_time=f"{signal.minutes_to_close:.0f} min from now",
            current_price=f"{ind.get('price', 0):,.2f}",
            minutes=signal.minutes_to_close,
            probability=signal.our_probability * 100,
            side=signal.side.upper(),
            price=int(signal.kalshi_implied * 100),
            implied=signal.kalshi_implied * 100,
            edge=signal.edge_cents,
            rsi=ind.get("rsi", "N/A"),
            momentum=ind.get("momentum", "N/A"),
            vol=ind.get("volatility_15m", "N/A"),
            bb_position=ind.get("bb_position", "N/A"),
            funding_rate=ind.get("funding_rate", "N/A"),
            vwap=f"{ind.get('vwap', 0):,.2f}",
            ema9=f"{ind.get('ema9', 0):,.2f}",
            ema21=f"{ind.get('ema21', 0):,.2f}",
        )

    async def validate(self, signal) -> ConsensusResult:
        """
        Run signal through all available models and return consensus.
        """
        prompt = self._build_prompt(signal)

        # Run all models concurrently
        tasks = []
        if self._openai_key:
            tasks.append(self._query_openai(prompt))
        else:
            tasks.append(self._no_key_response("gpt-4o-mini"))

        if self._anthropic_key:
            tasks.append(self._query_anthropic(prompt))
        else:
            tasks.append(self._no_key_response("claude-haiku-4.5"))

        if self._google_key:
            tasks.append(self._query_gemini(prompt))
        else:
            tasks.append(self._no_key_response("gemini-2.0-flash"))

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Process results
        models = []
        for r in results:
            if isinstance(r, Exception):
                models.append(ModelResponse(
                    model="unknown", action="FAILED", confidence=0,
                    side="", reasoning=str(r), risk_level="high",
                    latency_ms=0,
                ))
            else:
                models.append(r)

        # Calculate consensus
        consensus = self._calculate_consensus(models, signal.side)

        self.total_validations += 1
        # Rough cost estimate: ~500 tokens × 3 models
        self.total_cost_estimate += 0.0002

        logger.info(
            f"AI consensus: {consensus.action} "
            f"({consensus.follow_count}/{consensus.active_count} FOLLOW) "
            f"for {signal.ticker}"
        )

        return consensus

    def _calculate_consensus(self, models: list[ModelResponse], default_side: str) -> ConsensusResult:
        """
        Consensus rules:
        - 3 active → 2 must agree FOLLOW
        - 2 active → both must agree
        - 1 active → needs 65%+ confidence
        - 0 active → SKIP
        """
        active = [m for m in models if m.action in ("FOLLOW", "SKIP")]
        follow = [m for m in active if m.action == "FOLLOW"]
        skip = [m for m in active if m.action == "SKIP"]

        action = "SKIP"
        n = len(active)

        if n >= 3 and len(follow) >= 2:
            action = "FOLLOW"
        elif n == 2 and len(follow) == 2:
            action = "FOLLOW"
        elif n == 1 and len(follow) == 1 and follow[0].confidence >= 0.65:
            action = "FOLLOW"

        # Average confidence of agreeing models
        if follow and action == "FOLLOW":
            avg_conf = sum(m.confidence for m in follow) / len(follow)
        else:
            avg_conf = 0.0

        # Determine side from majority
        side = default_side
        if follow:
            sides = [m.side for m in follow if m.side]
            if sides:
                side = max(set(sides), key=sides.count)

        return ConsensusResult(
            action=action,
            side=side,
            confidence=avg_conf,
            models=models,
            follow_count=len(follow),
            skip_count=len(skip),
            active_count=n,
        )

    # ── Model Queries ──────────────────────────────────────────

    async def _query_openai(self, prompt: str) -> ModelResponse:
        """Query GPT-4o-mini."""
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.openai.com/v1/chat/completions",
                    headers={"Authorization": f"Bearer {self._openai_key}"},
                    json={
                        "model": "gpt-4o-mini",
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                        "max_tokens": 500,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["choices"][0]["message"]["content"]
                parsed = self._parse_response(content)
                return ModelResponse(
                    model="gpt-4o-mini",
                    latency_ms=(time.time() - start) * 1000,
                    **parsed,
                )
        except Exception as e:
            logger.warning(f"OpenAI query failed: {e}")
            return ModelResponse(
                model="gpt-4o-mini", action="FAILED", confidence=0,
                side="", reasoning=str(e), risk_level="high",
                latency_ms=(time.time() - start) * 1000,
            )

    async def _query_anthropic(self, prompt: str) -> ModelResponse:
        """Query Claude Haiku 4.5."""
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    "https://api.anthropic.com/v1/messages",
                    headers={
                        "x-api-key": self._anthropic_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json",
                    },
                    json={
                        "model": "claude-haiku-4-5-20251001",
                        "max_tokens": 500,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.3,
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["content"][0]["text"]
                parsed = self._parse_response(content)
                return ModelResponse(
                    model="claude-haiku-4.5",
                    latency_ms=(time.time() - start) * 1000,
                    **parsed,
                )
        except Exception as e:
            logger.warning(f"Anthropic query failed: {e}")
            return ModelResponse(
                model="claude-haiku-4.5", action="FAILED", confidence=0,
                side="", reasoning=str(e), risk_level="high",
                latency_ms=(time.time() - start) * 1000,
            )

    async def _query_gemini(self, prompt: str) -> ModelResponse:
        """Query Gemini 2.0 Flash."""
        start = time.time()
        try:
            async with httpx.AsyncClient(timeout=15) as client:
                resp = await client.post(
                    f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash-lite:generateContent?key={self._google_key}",
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.3,
                            "maxOutputTokens": 500,
                        },
                    },
                )
                resp.raise_for_status()
                data = resp.json()
                content = data["candidates"][0]["content"]["parts"][0]["text"]
                parsed = self._parse_response(content)
                return ModelResponse(
                    model="gemini-2.5-flash-lite",
                    latency_ms=(time.time() - start) * 1000,
                    **parsed,
                )
        except Exception as e:
            logger.warning(f"Gemini query failed: {e}")
            return ModelResponse(
                model="gemini-2.5-flash-lite", action="FAILED", confidence=0,
                side="", reasoning=str(e), risk_level="high",
                latency_ms=(time.time() - start) * 1000,
            )

    async def _no_key_response(self, model: str) -> ModelResponse:
        """Return a NO KEY response for unconfigured models."""
        return ModelResponse(
            model=model, action="NO KEY", confidence=0,
            side="", reasoning="API key not configured",
            risk_level="high", latency_ms=0,
        )

    @staticmethod
    def _parse_response(content: str) -> dict:
        """Parse JSON response from model, handling markdown code blocks."""
        content = content.strip()

        # Strip markdown code blocks if present
        if content.startswith("```"):
            lines = content.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            content = "\n".join(lines).strip()

        try:
            data = json.loads(content)
            return {
                "action": data.get("action", "SKIP").upper(),
                "confidence": float(data.get("confidence", 0)),
                "side": data.get("side", "").lower(),
                "reasoning": data.get("reasoning", ""),
                "risk_level": data.get("risk_level", "medium"),
            }
        except (json.JSONDecodeError, ValueError) as e:
            logger.warning(f"Failed to parse AI response: {content[:100]}")
            return {
                "action": "FAILED",
                "confidence": 0,
                "side": "",
                "reasoning": f"Parse error: {e}",
                "risk_level": "high",
            }

    # ── Serialization ──────────────────────────────────────────

    def to_dict(self) -> dict:
        return {
            "total_validations": self.total_validations,
            "estimated_cost": round(self.total_cost_estimate, 4),
            "models_available": {
                "openai": bool(self._openai_key),
                "anthropic": bool(self._anthropic_key),
                "google": bool(self._google_key),
            },
        }
