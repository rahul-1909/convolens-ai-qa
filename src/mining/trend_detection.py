"""Trend detection and analytics over evaluation data.

Queries stored evaluation results to produce time-series, distribution,
and comparative reports.
"""

from __future__ import annotations

import logging
from collections import Counter
from datetime import datetime

from sqlalchemy import func
from sqlalchemy.orm import Session

from src.models.database import (
    ConversationRecord,
    EvaluationRecord,
    FailureRecord_DB,
)

logger = logging.getLogger(__name__)


def get_failure_distribution(
    db: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
    agent_version: str | None = None,
    customer_id: str | None = None,
) -> list[dict]:
    """Get failure counts grouped by category and severity.

    Args:
        db: Database session.
        start_date: Optional lower-bound filter.
        end_date: Optional upper-bound filter.
        agent_version: Filter by agent version.
        customer_id: Filter by customer.

    Returns:
        List of dicts with ``category``, ``severity``, ``count``.
    """
    query = db.query(
        FailureRecord_DB.category,
        FailureRecord_DB.severity,
        func.count(FailureRecord_DB.id).label("count"),
    ).join(
        EvaluationRecord, FailureRecord_DB.evaluation_id == EvaluationRecord.id
    )

    if start_date:
        query = query.filter(EvaluationRecord.evaluated_at >= start_date)
    if end_date:
        query = query.filter(EvaluationRecord.evaluated_at <= end_date)

    if agent_version or customer_id:
        query = query.join(
            ConversationRecord,
            EvaluationRecord.conversation_id == ConversationRecord.id,
        )
        if agent_version:
            query = query.filter(
                ConversationRecord.agent_version == agent_version
            )
        if customer_id:
            query = query.filter(
                ConversationRecord.customer_id == customer_id
            )

    query = query.group_by(
        FailureRecord_DB.category, FailureRecord_DB.severity
    )

    rows = query.all()
    return [
        {"category": r.category, "severity": r.severity, "count": r.count}
        for r in rows
    ]


def get_top_root_causes(
    db: Session,
    limit: int = 10,
) -> list[dict]:
    """Get the most common root causes across all evaluations.

    Args:
        db: Database session.
        limit: Max number of results.

    Returns:
        List of dicts with ``root_cause``, ``count``, ``avg_confidence``.
    """
    query = (
        db.query(
            FailureRecord_DB.root_cause,
            func.count(FailureRecord_DB.id).label("count"),
            func.avg(FailureRecord_DB.root_cause_confidence).label("avg_confidence"),
        )
        .filter(FailureRecord_DB.root_cause.isnot(None))
        .group_by(FailureRecord_DB.root_cause)
        .order_by(func.count(FailureRecord_DB.id).desc())
        .limit(limit)
    )

    rows = query.all()
    return [
        {
            "root_cause": r.root_cause,
            "count": r.count,
            "avg_confidence": round(float(r.avg_confidence or 0), 2),
        }
        for r in rows
    ]


def get_score_summary(
    db: Session,
    start_date: datetime | None = None,
    end_date: datetime | None = None,
) -> dict:
    """Get aggregate scoring statistics.

    Args:
        db: Database session.
        start_date: Optional lower-bound.
        end_date: Optional upper-bound.

    Returns:
        Dict with ``total_conversations``, ``avg_composite``,
        ``avg_accuracy``, ``avg_empathy``, ``avg_flow``, ``avg_goal``.
    """
    query = db.query(
        func.count(EvaluationRecord.id).label("total"),
        func.avg(EvaluationRecord.composite_score).label("avg_composite"),
        func.avg(EvaluationRecord.accuracy_score).label("avg_accuracy"),
        func.avg(EvaluationRecord.empathy_score).label("avg_empathy"),
        func.avg(EvaluationRecord.flow_score).label("avg_flow"),
        func.avg(EvaluationRecord.goal_completion_score).label("avg_goal"),
    )

    if start_date:
        query = query.filter(EvaluationRecord.evaluated_at >= start_date)
    if end_date:
        query = query.filter(EvaluationRecord.evaluated_at <= end_date)

    row = query.one()
    return {
        "total_conversations": row.total or 0,
        "avg_composite": round(float(row.avg_composite or 0), 2),
        "avg_accuracy": round(float(row.avg_accuracy or 0), 2),
        "avg_empathy": round(float(row.avg_empathy or 0), 2),
        "avg_flow": round(float(row.avg_flow or 0), 2),
        "avg_goal_completion": round(float(row.avg_goal or 0), 2),
    }
