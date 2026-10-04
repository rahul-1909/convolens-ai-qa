"""SQLAlchemy ORM models for ConvoLens PostgreSQL / SQLite persistence."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone

from sqlalchemy import (
    Column,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    Boolean,
    JSON,
    create_engine,
)
from sqlalchemy.orm import DeclarativeBase, relationship


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""

    pass


def _uuid() -> str:
    return str(uuid.uuid4())


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


# ── Conversation ─────────────────────────────────────────────────────────────


class ConversationRecord(Base):
    """Persisted conversation metadata."""

    __tablename__ = "conversations"

    id = Column(String(64), primary_key=True)
    customer_id = Column(String(128), nullable=True, index=True)
    agent_version = Column(String(64), nullable=True, index=True)
    prompt_version = Column(String(64), nullable=True)
    conversation_goal = Column(String(256), nullable=True)
    channel = Column(String(16), nullable=True)
    language = Column(String(8), nullable=True)
    started_at = Column(DateTime, nullable=True)
    ended_at = Column(DateTime, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now)

    turns = relationship(
        "TurnRecord", back_populates="conversation", cascade="all, delete-orphan"
    )
    evaluation = relationship(
        "EvaluationRecord",
        back_populates="conversation",
        uselist=False,
        cascade="all, delete-orphan",
    )


# ── Turn ─────────────────────────────────────────────────────────────────────


class TurnRecord(Base):
    """Persisted turn within a conversation."""

    __tablename__ = "turns"

    id = Column(String(64), primary_key=True, default=_uuid)
    conversation_id = Column(
        String(64), ForeignKey("conversations.id", ondelete="CASCADE"), nullable=False
    )
    turn_index = Column(Integer, nullable=False)
    speaker = Column(String(16), nullable=False)  # agent / customer / system
    transcript = Column(Text, nullable=False)
    timestamp = Column(DateTime, nullable=True)
    duration_ms = Column(Float, nullable=True)
    asr_confidence = Column(Float, nullable=True)
    tool_calls_json = Column(JSON, nullable=True)
    kb_snippets_json = Column(JSON, nullable=True)

    conversation = relationship("ConversationRecord", back_populates="turns")
    turn_evaluation = relationship(
        "TurnEvaluationRecord",
        back_populates="turn",
        uselist=False,
        cascade="all, delete-orphan",
    )

    __table_args__ = (
        Index("ix_turns_conv_idx", "conversation_id", "turn_index"),
    )


# ── Evaluation (conversation-level) ─────────────────────────────────────────


class EvaluationRecord(Base):
    """Persisted conversation-level evaluation."""

    __tablename__ = "evaluations"

    id = Column(String(64), primary_key=True, default=_uuid)
    conversation_id = Column(
        String(64),
        ForeignKey("conversations.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    accuracy_score = Column(Integer, nullable=False)
    empathy_score = Column(Integer, nullable=False)
    flow_score = Column(Integer, nullable=False)
    goal_completion_score = Column(Integer, nullable=False)
    composite_score = Column(Float, nullable=False)
    evaluator_model = Column(String(128), nullable=True)
    recommended_fixes_json = Column(JSON, nullable=True)
    metadata_json = Column(JSON, nullable=True)
    evaluated_at = Column(DateTime, default=_utc_now)

    conversation = relationship("ConversationRecord", back_populates="evaluation")
    failures = relationship(
        "FailureRecord_DB",
        back_populates="evaluation",
        cascade="all, delete-orphan",
    )


# ── Turn Evaluation ─────────────────────────────────────────────────────────


class TurnEvaluationRecord(Base):
    """Persisted per-turn evaluation scores."""

    __tablename__ = "turn_evaluations"

    id = Column(String(64), primary_key=True, default=_uuid)
    turn_id = Column(
        String(64),
        ForeignKey("turns.id", ondelete="CASCADE"),
        unique=True,
        nullable=False,
    )
    accuracy_score = Column(Integer, nullable=False)
    empathy_score = Column(Integer, nullable=False)
    flow_score = Column(Integer, nullable=False)
    goal_completion_score = Column(Integer, nullable=False)
    composite_score = Column(Float, nullable=False)
    is_flagged = Column(Boolean, default=False)

    turn = relationship("TurnRecord", back_populates="turn_evaluation")


# ── Failure ──────────────────────────────────────────────────────────────────


class FailureRecord_DB(Base):
    """Persisted failure record linked to an evaluation."""

    __tablename__ = "failures"

    id = Column(String(64), primary_key=True, default=_uuid)
    evaluation_id = Column(
        String(64),
        ForeignKey("evaluations.id", ondelete="CASCADE"),
        nullable=False,
    )
    turn_id = Column(String(64), nullable=True)
    category = Column(String(16), nullable=False, index=True)
    subtype = Column(String(64), nullable=True)
    severity = Column(String(4), nullable=False, index=True)
    evidence_quote = Column(Text, nullable=True)
    reasoning = Column(Text, nullable=True)
    root_cause = Column(String(16), nullable=True, index=True)
    root_cause_confidence = Column(Float, nullable=True)
    root_cause_signals_json = Column(JSON, nullable=True)
    recommended_fix = Column(Text, nullable=True)
    created_at = Column(DateTime, default=_utc_now)

    evaluation = relationship("EvaluationRecord", back_populates="failures")

    __table_args__ = (
        Index("ix_failures_cat_sev", "category", "severity"),
    )


# ── Cluster (pattern mining) ────────────────────────────────────────────────


class ClusterRecord(Base):
    """Persisted failure-pattern cluster."""

    __tablename__ = "clusters"

    id = Column(String(64), primary_key=True, default=_uuid)
    label = Column(String(256), nullable=False)
    description = Column(Text, nullable=True)
    cluster_size = Column(Integer, nullable=False)
    root_cause = Column(String(16), nullable=True)
    recommended_fix = Column(Text, nullable=True)
    estimated_impact = Column(String(16), nullable=True)
    sample_turn_ids_json = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=_utc_now)
