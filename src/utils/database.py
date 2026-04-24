from __future__ import annotations

"""
Database layer — SQLAlchemy + aiosqlite for full audit trail.

Tables: signals, ai_decisions, trades, skipped_signals
"""

import os
import json
import time
import logging
from datetime import datetime
from typing import Optional

from sqlalchemy import (
    create_engine, Column, Integer, Float, Text, String,
    JSON, TIMESTAMP, func, text
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

logger = logging.getLogger("database")


class Base(DeclarativeBase):
    pass


class Signal(Base):
    __tablename__ = "signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(Float)
    ticker = Column(Text)
    strike_price = Column(Float)
    side = Column(String(10))
    our_probability = Column(Float)
    kalshi_implied = Column(Float)
    edge_cents = Column(Float)
    indicators = Column(JSON)
    created_at = Column(TIMESTAMP, server_default=func.now())


class AIDecision(Base):
    __tablename__ = "ai_decisions"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_id = Column(Integer)
    ticker = Column(Text)
    gpt_action = Column(String(10))
    gpt_confidence = Column(Float)
    gpt_reasoning = Column(Text)
    claude_action = Column(String(10))
    claude_confidence = Column(Float)
    claude_reasoning = Column(Text)
    gemini_action = Column(String(10))
    gemini_confidence = Column(Float)
    gemini_reasoning = Column(Text)
    consensus_action = Column(String(10))
    consensus_side = Column(String(10))
    created_at = Column(TIMESTAMP, server_default=func.now())


class Trade(Base):
    __tablename__ = "trades"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(Text)
    side = Column(String(10))
    entry_price_cents = Column(Integer)
    exit_price_cents = Column(Integer, nullable=True)
    count = Column(Integer, default=1)
    cost_usd = Column(Float)
    pnl_usd = Column(Float, nullable=True)
    status = Column(String(20), default="open")
    exit_reason = Column(Text, nullable=True)
    mode = Column(String(10), default="paper")
    follow_count = Column(Integer, nullable=True)   # how many AI models said FOLLOW (2 or 3)
    active_count = Column(Integer, nullable=True)    # how many AI models were active
    avg_confidence = Column(Float, nullable=True)     # average confidence of FOLLOW models
    edge_cents = Column(Float, nullable=True)         # edge at time of entry
    spread_cents = Column(Integer, nullable=True)     # bid-ask spread at entry
    asset = Column(String(10), nullable=True)          # "BTC" or "ETH"
    contract_type = Column(String(10), nullable=True)  # "15m", "hourly", "daily"
    bot_version = Column(String(20), default="3.0.0")
    created_at = Column(TIMESTAMP, server_default=func.now())
    closed_at = Column(TIMESTAMP, nullable=True)


class SkippedSignal(Base):
    __tablename__ = "skipped_signals"

    id = Column(Integer, primary_key=True, autoincrement=True)
    ticker = Column(Text)
    side = Column(String(10))
    edge_cents = Column(Float)
    skip_reason = Column(Text)
    outcome = Column(Text, nullable=True)
    hypothetical_pnl = Column(Float, nullable=True)
    created_at = Column(TIMESTAMP, server_default=func.now())
    checked_at = Column(TIMESTAMP, nullable=True)


class BotState(Base):
    """Persistent key-value store for bot state that must survive restarts."""
    __tablename__ = "bot_state"

    key = Column(String(50), primary_key=True)
    value = Column(Text)
    updated_at = Column(TIMESTAMP, server_default=func.now())


class Database:
    """Synchronous SQLite database manager (runs in thread pool for async)."""

    def __init__(self, db_path: str = "data/crypto_bot.db"):
        os.makedirs(os.path.dirname(db_path) if os.path.dirname(db_path) else ".", exist_ok=True)
        self.engine = create_engine(f"sqlite:///{db_path}", echo=False)
        Base.metadata.create_all(self.engine)
        self._migrate()
        self.SessionLocal = sessionmaker(bind=self.engine)
        logger.info(f"Database initialized at {db_path}")

    def _migrate(self):
        """Add new columns to existing tables if they don't exist yet."""
        new_columns = [
            ("trades", "follow_count", "INTEGER"),
            ("trades", "active_count", "INTEGER"),
            ("trades", "avg_confidence", "REAL"),
            ("trades", "edge_cents", "REAL"),
            ("trades", "spread_cents", "INTEGER"),
            ("trades", "asset", "TEXT"),
            ("trades", "contract_type", "TEXT"),
        ]
        with self.engine.connect() as conn:
            for table, col, col_type in new_columns:
                try:
                    conn.execute(text(f"ALTER TABLE {table} ADD COLUMN {col} {col_type}"))
                    conn.commit()
                    logger.info(f"Migration: added {table}.{col}")
                except Exception:
                    pass  # column already exists

    def _session(self) -> Session:
        return self.SessionLocal()

    # ── Signals ────────────────────────────────────────────────

    def record_signal(self, ticker: str, strike_price: float, side: str,
                      our_prob: float, kalshi_implied: float, edge_cents: float,
                      indicators: dict) -> int:
        with self._session() as s:
            sig = Signal(
                timestamp=time.time(),
                ticker=ticker,
                strike_price=strike_price,
                side=side,
                our_probability=our_prob,
                kalshi_implied=kalshi_implied,
                edge_cents=edge_cents,
                indicators=indicators,
            )
            s.add(sig)
            s.commit()
            return sig.id

    # ── AI Decisions ───────────────────────────────────────────

    def record_ai_decision(self, signal_id: int, ticker: str,
                           gpt: dict, claude: dict, gemini: dict,
                           consensus_action: str, consensus_side: str) -> int:
        with self._session() as s:
            dec = AIDecision(
                signal_id=signal_id,
                ticker=ticker,
                gpt_action=gpt.get("action", "NONE"),
                gpt_confidence=gpt.get("confidence", 0),
                gpt_reasoning=gpt.get("reasoning", ""),
                claude_action=claude.get("action", "NONE"),
                claude_confidence=claude.get("confidence", 0),
                claude_reasoning=claude.get("reasoning", ""),
                gemini_action=gemini.get("action", "NONE"),
                gemini_confidence=gemini.get("confidence", 0),
                gemini_reasoning=gemini.get("reasoning", ""),
                consensus_action=consensus_action,
                consensus_side=consensus_side,
            )
            s.add(dec)
            s.commit()
            return dec.id

    # ── Trades ─────────────────────────────────────────────────

    def record_trade(self, ticker: str, side: str, entry_price_cents: int,
                     count: int, cost_usd: float, mode: str = "paper",
                     follow_count: int = None, active_count: int = None,
                     avg_confidence: float = None, edge_cents: float = None,
                     spread_cents: int = None, asset: str = None,
                     contract_type: str = None) -> int:
        # Auto-detect asset and contract type from ticker
        if asset is None:
            asset = "ETH" if "ETH" in ticker else "BTC"
        if contract_type is None:
            if "15M" in ticker:
                contract_type = "15m"
            elif "HR" in ticker:
                contract_type = "hourly"
            else:
                contract_type = "daily"

        with self._session() as s:
            trade = Trade(
                ticker=ticker,
                side=side,
                entry_price_cents=entry_price_cents,
                count=count,
                cost_usd=cost_usd,
                mode=mode,
                follow_count=follow_count,
                active_count=active_count,
                avg_confidence=avg_confidence,
                edge_cents=edge_cents,
                spread_cents=spread_cents,
                asset=asset,
                contract_type=contract_type,
            )
            s.add(trade)
            s.commit()
            return trade.id

    def close_trade(self, trade_id: int, exit_price_cents: int,
                    pnl_usd: float, exit_reason: str):
        with self._session() as s:
            trade = s.query(Trade).filter_by(id=trade_id).first()
            if trade:
                trade.exit_price_cents = exit_price_cents
                trade.pnl_usd = pnl_usd
                trade.status = "closed"
                trade.exit_reason = exit_reason
                trade.closed_at = datetime.utcnow()
                s.commit()

    def get_open_trades(self, mode: Optional[str] = None) -> list[dict]:
        with self._session() as s:
            q = s.query(Trade).filter_by(status="open")
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.all()
            return [
                {
                    "id": t.id, "ticker": t.ticker, "side": t.side,
                    "entry_price_cents": t.entry_price_cents,
                    "count": t.count, "cost_usd": t.cost_usd,
                    "mode": t.mode, "created_at": str(t.created_at),
                }
                for t in trades
            ]

    def get_trade_stats(self, mode: Optional[str] = None) -> dict:
        with self._session() as s:
            q = s.query(Trade).filter_by(status="closed")
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.all()

            if not trades:
                return {"total": 0, "wins": 0, "losses": 0, "win_rate": 0,
                        "total_pnl": 0, "avg_pnl": 0}

            wins = [t for t in trades if (t.pnl_usd or 0) > 0]
            losses = [t for t in trades if (t.pnl_usd or 0) <= 0]
            total_pnl = sum(t.pnl_usd or 0 for t in trades)

            return {
                "total": len(trades),
                "wins": len(wins),
                "losses": len(losses),
                "win_rate": len(wins) / len(trades) * 100 if trades else 0,
                "total_pnl": round(total_pnl, 2),
                "avg_pnl": round(total_pnl / len(trades), 2) if trades else 0,
            }

    def get_today_stats(self, mode: Optional[str] = None) -> dict:
        """Stats for today only (for daily loss limit checking)."""
        with self._session() as s:
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0)
            q = s.query(Trade).filter(
                Trade.status == "closed",
                Trade.closed_at >= today_start,
            )
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.all()

            today_pnl = sum(t.pnl_usd or 0 for t in trades)
            today_trades = s.query(Trade).filter(
                Trade.created_at >= today_start
            ).count()

            return {
                "today_pnl": round(today_pnl, 2),
                "today_trades": today_trades,
                "today_closed": len(trades),
            }

    def get_closed_trades(self, mode: Optional[str] = None, limit: int = 100) -> list[dict]:
        """Return recently closed trades for the dashboard."""
        with self._session() as s:
            q = s.query(Trade).filter_by(status="closed")
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.order_by(Trade.closed_at.desc()).limit(limit).all()
            return [
                {
                    "id": t.id,
                    "ticker": t.ticker,
                    "side": t.side,
                    "entry_cents": t.entry_price_cents,
                    "exit_cents": t.exit_price_cents,
                    "count": t.count,
                    "cost_usd": round(t.cost_usd or 0, 2),
                    "pnl": round(t.pnl_usd or 0, 2),
                    "exit_reason": t.exit_reason or "",
                    "mode": t.mode,
                    "opened": str(t.created_at)[:19] if t.created_at else "",
                    "closed": str(t.closed_at)[:19] if t.closed_at else "",
                }
                for t in trades
            ]

    # ── Consensus Performance ─────────────────────────────────

    def get_consensus_stats(self) -> dict:
        """Break down win rate and P&L by follow_count (2/3 vs 3/3)."""
        with self._session() as s:
            trades = s.query(Trade).filter(
                Trade.status == "closed",
                Trade.follow_count.isnot(None),
            ).all()

            stats = {}
            for fc in [2, 3]:
                group = [t for t in trades if t.follow_count == fc]
                if not group:
                    stats[f"{fc}_of_3"] = {"trades": 0, "wins": 0, "win_rate": 0,
                                           "total_pnl": 0, "avg_pnl": 0}
                    continue
                wins = [t for t in group if (t.pnl_usd or 0) > 0]
                total_pnl = sum(t.pnl_usd or 0 for t in group)
                stats[f"{fc}_of_3"] = {
                    "trades": len(group),
                    "wins": len(wins),
                    "win_rate": round(len(wins) / len(group) * 100, 1),
                    "total_pnl": round(total_pnl, 2),
                    "avg_pnl": round(total_pnl / len(group), 2),
                }
            return stats

    # ── Skipped Signals ────────────────────────────────────────

    def record_skip(self, ticker: str, side: str, edge_cents: float,
                    skip_reason: str):
        with self._session() as s:
            skip = SkippedSignal(
                ticker=ticker,
                side=side,
                edge_cents=edge_cents,
                skip_reason=skip_reason,
            )
            s.add(skip)
            s.commit()

    # ── Contract Type P&L Breakdown ───────────────────────────

    def get_contract_type_stats(self, mode: Optional[str] = None) -> dict:
        """Break down win rate and P&L by contract type: 15M, hourly, daily."""
        with self._session() as s:
            q = s.query(Trade).filter_by(status="closed")
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.all()

            buckets = {"15m": [], "hourly": [], "daily": []}
            for t in trades:
                ticker = t.ticker or ""
                if "15M" in ticker:
                    buckets["15m"].append(t)
                elif "H" in ticker and "D" not in ticker:
                    buckets["hourly"].append(t)
                else:
                    buckets["daily"].append(t)

            stats = {}
            for name, group in buckets.items():
                if not group:
                    stats[name] = {"trades": 0, "wins": 0, "win_rate": 0,
                                   "total_pnl": 0, "avg_pnl": 0}
                    continue
                wins = [t for t in group if (t.pnl_usd or 0) > 0]
                total_pnl = sum(t.pnl_usd or 0 for t in group)
                stats[name] = {
                    "trades": len(group),
                    "wins": len(wins),
                    "win_rate": round(len(wins) / len(group) * 100, 1),
                    "total_pnl": round(total_pnl, 2),
                    "avg_pnl": round(total_pnl / len(group), 2),
                }
            return stats

    # ── Reset ─────────────────────────────────────────────────

    def reset_all(self):
        """Wipe all trade history for a fresh start."""
        with self._session() as s:
            s.query(Trade).delete()
            s.query(Signal).delete()
            s.query(AIDecision).delete()
            s.query(SkippedSignal).delete()
            s.commit()
        logger.info("Database reset — all history wiped")

    # ── Edge Calibration ─────────────────────────────────────

    def get_edge_calibration(self, mode: Optional[str] = None) -> dict:
        """
        Break down win rate by edge bucket to see if 10¢ edge
        actually wins like 10¢ edge (calibration check).
        """
        with self._session() as s:
            q = s.query(Trade).filter(
                Trade.status == "closed",
                Trade.edge_cents.isnot(None),
            )
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.all()

            buckets = [
                ("6-8", 6, 8),
                ("8-10", 8, 10),
                ("10-12", 10, 12),
                ("12-15", 12, 15),
                ("15-20", 15, 20),
                ("20+", 20, 999),
            ]

            stats = {}
            for name, lo, hi in buckets:
                group = [t for t in trades if lo <= (t.edge_cents or 0) < hi]
                if not group:
                    stats[name] = {"trades": 0, "wins": 0, "win_rate": 0,
                                   "total_pnl": 0, "avg_pnl": 0, "avg_edge": 0}
                    continue
                wins = [t for t in group if (t.pnl_usd or 0) > 0]
                total_pnl = sum(t.pnl_usd or 0 for t in group)
                avg_edge = sum(t.edge_cents or 0 for t in group) / len(group)
                stats[name] = {
                    "trades": len(group),
                    "wins": len(wins),
                    "win_rate": round(len(wins) / len(group) * 100, 1),
                    "total_pnl": round(total_pnl, 2),
                    "avg_pnl": round(total_pnl / len(group), 2),
                    "avg_edge": round(avg_edge, 1),
                }
            return stats

    # ── Asset Breakdown ───────────────────────────────────────

    def get_asset_stats(self, mode: Optional[str] = None) -> dict:
        """Break down win rate and P&L by asset (BTC vs ETH)."""
        with self._session() as s:
            q = s.query(Trade).filter_by(status="closed")
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.all()

            stats = {}
            for asset_name in ["BTC", "ETH"]:
                group = [t for t in trades
                         if (t.asset == asset_name) or
                         (t.asset is None and asset_name in (t.ticker or ""))]
                if not group:
                    stats[asset_name] = {"trades": 0, "wins": 0, "win_rate": 0,
                                         "total_pnl": 0, "avg_pnl": 0}
                    continue
                wins = [t for t in group if (t.pnl_usd or 0) > 0]
                total_pnl = sum(t.pnl_usd or 0 for t in group)
                stats[asset_name] = {
                    "trades": len(group),
                    "wins": len(wins),
                    "win_rate": round(len(wins) / len(group) * 100, 1),
                    "total_pnl": round(total_pnl, 2),
                    "avg_pnl": round(total_pnl / len(group), 2),
                }
            return stats

    # ── Persistent Bot State ────────────────────────────────────

    def get_state(self, key: str, default: str = "") -> str:
        """Get a persistent bot state value."""
        with self._session() as s:
            row = s.query(BotState).filter_by(key=key).first()
            return row.value if row else default

    def set_state(self, key: str, value: str):
        """Set a persistent bot state value."""
        with self._session() as s:
            row = s.query(BotState).filter_by(key=key).first()
            if row:
                row.value = value
                row.updated_at = datetime.utcnow()
            else:
                s.add(BotState(key=key, value=value))
            s.commit()

    # ── ADAPTIVE LEARNING ENGINE ─────────────────────────────────
    # These methods let the bot learn from its own results and adapt
    # in real-time: which edges are real, which hours are profitable,
    # which AI models are accurate, and how to size positions.

    def get_model_accuracy(self) -> dict:
        """
        Track each AI model's historical accuracy.
        Returns {model_name: {trades, correct, accuracy, avg_pnl_when_follow}}.
        Used by validator to weight votes.
        """
        with self._session() as s:
            # Get AI decisions joined with trade outcomes
            decisions = s.query(AIDecision).all()
            trades = s.query(Trade).filter_by(status="closed").all()

            # Build ticker → pnl lookup
            ticker_pnl = {}
            for t in trades:
                ticker_pnl[t.ticker] = t.pnl_usd or 0

            models = {
                "gpt": {"trades": 0, "correct": 0, "total_pnl": 0},
                "claude": {"trades": 0, "correct": 0, "total_pnl": 0},
                "gemini": {"trades": 0, "correct": 0, "total_pnl": 0},
            }

            for d in decisions:
                pnl = ticker_pnl.get(d.ticker)
                if pnl is None:
                    continue  # no trade for this decision

                won = pnl > 0

                for model_key, action_col in [
                    ("gpt", d.gpt_action),
                    ("claude", d.claude_action),
                    ("gemini", d.gemini_action),
                ]:
                    if action_col in ("FOLLOW", "SKIP"):
                        models[model_key]["trades"] += 1
                        # Model was "correct" if:
                        # - It said FOLLOW and the trade won
                        # - It said SKIP and the trade lost
                        if (action_col == "FOLLOW" and won) or (action_col == "SKIP" and not won):
                            models[model_key]["correct"] += 1
                        if action_col == "FOLLOW":
                            models[model_key]["total_pnl"] += pnl

            result = {}
            for name, data in models.items():
                n = data["trades"]
                result[name] = {
                    "trades": n,
                    "correct": data["correct"],
                    "accuracy": round(data["correct"] / n * 100, 1) if n > 0 else 50.0,
                    "avg_pnl_when_follow": round(data["total_pnl"] / max(1, n), 3),
                }
            return result

    def get_hourly_performance(self, mode: Optional[str] = None) -> dict:
        """
        Win rate and P&L by hour of day (UTC).
        Lets bot learn which hours are profitable.
        """
        with self._session() as s:
            q = s.query(Trade).filter_by(status="closed")
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.all()

            hours = {h: {"trades": 0, "wins": 0, "pnl": 0.0} for h in range(24)}
            for t in trades:
                if t.created_at:
                    try:
                        h = t.created_at.hour
                    except Exception:
                        continue
                    hours[h]["trades"] += 1
                    if (t.pnl_usd or 0) > 0:
                        hours[h]["wins"] += 1
                    hours[h]["pnl"] += t.pnl_usd or 0

            result = {}
            for h, data in hours.items():
                n = data["trades"]
                result[h] = {
                    "trades": n,
                    "wins": data["wins"],
                    "win_rate": round(data["wins"] / n * 100, 1) if n > 0 else 0,
                    "pnl": round(data["pnl"], 2),
                    "profitable": data["pnl"] > 0 if n >= 5 else True,  # need 5+ trades to judge
                }
            return result

    def get_adaptive_params(self, mode: Optional[str] = None) -> dict:
        """
        Master learning method — returns adaptive parameters the bot
        should use RIGHT NOW based on all historical data.

        This is the brain of the learning system.
        """
        edge_cal = self.get_edge_calibration(mode=mode)
        hourly = self.get_hourly_performance(mode=mode)
        model_acc = self.get_model_accuracy()
        contract_stats = self.get_contract_type_stats(mode=mode)
        asset_stats = self.get_asset_stats(mode=mode)

        # ── Determine best minimum edge ──
        # Find the lowest edge bucket that's profitable with 10+ trades
        best_min_edge = 10.0  # default
        for bucket_name in ["6-8", "8-10", "10-12", "12-15", "15-20", "20+"]:
            bucket = edge_cal.get(bucket_name, {})
            if bucket.get("trades", 0) >= 10:
                if bucket.get("win_rate", 0) >= 55 and bucket.get("total_pnl", 0) > 0:
                    # This bucket is profitable — can lower min edge
                    ranges = {"6-8": 6, "8-10": 8, "10-12": 10, "12-15": 12, "15-20": 15, "20+": 20}
                    best_min_edge = min(best_min_edge, ranges.get(bucket_name, 10))
                    break  # found the lowest profitable bucket

        # ── Determine unprofitable hours ──
        bad_hours = [h for h, data in hourly.items()
                     if data["trades"] >= 10 and not data["profitable"]]

        # ── Best contract type ──
        best_contract = "15m"  # default
        best_pnl = -999
        for ct, data in contract_stats.items():
            if data.get("trades", 0) >= 10 and data.get("total_pnl", 0) > best_pnl:
                best_pnl = data["total_pnl"]
                best_contract = ct

        # ── Model weights (normalized accuracy) ──
        total_acc = sum(m.get("accuracy", 50) for m in model_acc.values())
        model_weights = {}
        for name, data in model_acc.items():
            if total_acc > 0:
                model_weights[name] = round(data["accuracy"] / total_acc, 3)
            else:
                model_weights[name] = 0.333

        # ── Sizing multiplier based on recent performance ──
        # If last 20 trades are profitable, size up. If losing, size down.
        recent_trades = self._get_recent_trades(20, mode=mode)
        recent_pnl = sum(t.get("pnl", 0) for t in recent_trades)
        recent_wins = sum(1 for t in recent_trades if t.get("pnl", 0) > 0)
        recent_wr = recent_wins / len(recent_trades) * 100 if recent_trades else 50

        if recent_wr >= 65 and recent_pnl > 0:
            sizing_mult = 1.5   # hot streak — size up
        elif recent_wr >= 55:
            sizing_mult = 1.2   # solid — slight bump
        elif recent_wr <= 35:
            sizing_mult = 0.5   # cold streak — cut size
        elif recent_wr <= 45:
            sizing_mult = 0.75  # underperforming — reduce
        else:
            sizing_mult = 1.0

        # ── Asset preference ──
        asset_preference = {}
        for asset_name, data in asset_stats.items():
            if data.get("trades", 0) >= 10:
                asset_preference[asset_name] = {
                    "profitable": data.get("total_pnl", 0) > 0,
                    "win_rate": data.get("win_rate", 50),
                    "size_mult": 1.2 if data.get("win_rate", 50) >= 60 else (0.7 if data.get("win_rate", 50) < 45 else 1.0),
                }
            else:
                asset_preference[asset_name] = {"profitable": True, "win_rate": 50, "size_mult": 1.0}

        return {
            "learned_min_edge": best_min_edge,
            "bad_hours_utc": bad_hours,
            "best_contract_type": best_contract,
            "model_weights": model_weights,
            "model_accuracy": model_acc,
            "sizing_multiplier": sizing_mult,
            "recent_win_rate": round(recent_wr, 1),
            "recent_pnl": round(recent_pnl, 2),
            "asset_preference": asset_preference,
            "total_closed_trades": sum(d.get("trades", 0) for d in edge_cal.values()),
            "learning_active": sum(d.get("trades", 0) for d in edge_cal.values()) >= 20,
        }

    def _get_recent_trades(self, n: int, mode: Optional[str] = None) -> list[dict]:
        """Get the N most recent closed trades."""
        with self._session() as s:
            q = s.query(Trade).filter_by(status="closed")
            if mode:
                q = q.filter_by(mode=mode)
            trades = q.order_by(Trade.closed_at.desc()).limit(n).all()
            return [{"pnl": t.pnl_usd or 0, "ticker": t.ticker, "edge": t.edge_cents or 0}
                    for t in trades]

    # ── P&L History (for charting) ─────────────────────────────

    def get_pnl_history(self, limit: int = 200) -> list[dict]:
        with self._session() as s:
            trades = (
                s.query(Trade)
                .filter_by(status="closed")
                .order_by(Trade.closed_at.asc())
                .limit(limit)
                .all()
            )
            cumulative = 0.0
            history = []
            for t in trades:
                cumulative += t.pnl_usd or 0
                history.append({
                    "timestamp": str(t.closed_at),
                    "pnl": t.pnl_usd or 0,
                    "cumulative": round(cumulative, 2),
                    "ticker": t.ticker,
                })
            return history
