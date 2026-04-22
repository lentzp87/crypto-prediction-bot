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
    bot_version = Column(String(20), default="1.0.0")
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
                     avg_confidence: float = None, edge_cents: float = None) -> int:
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
