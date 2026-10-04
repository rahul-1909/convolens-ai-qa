"""Unit tests for trend detection and analytics queries."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
import pytest

from src.mining.trend_detection import (
    get_failure_distribution,
    get_top_root_causes,
    get_score_summary,
)
from src.evaluation.pipeline import EvaluationPipeline
from src.schemas.conversation import Conversation, Turn, SpeakerRole


def test_trend_queries_with_filters(db_session, sample_conversation):
    """Test get_failure_distribution and get_score_summary with all filters."""
    # Ensure at least one conversation is evaluated and in DB
    pipeline = EvaluationPipeline(db=db_session)
    pipeline.evaluate(sample_conversation)

    now = datetime.now(timezone.utc)
    start_date = now - timedelta(days=1)
    end_date = now + timedelta(days=1)

    # 1. Distribution without filters
    dist = get_failure_distribution(db_session)
    assert isinstance(dist, list)

    # 2. Distribution with date filters
    dist_dates = get_failure_distribution(
        db_session, start_date=start_date, end_date=end_date
    )
    assert isinstance(dist_dates, list)

    # 3. Distribution with agent version & customer_id filters
    dist_filtered = get_failure_distribution(
        db_session,
        agent_version=sample_conversation.agent_version,
        customer_id=sample_conversation.customer_id,
    )
    assert isinstance(dist_filtered, list)

    # 4. Top root causes
    top_rc = get_top_root_causes(db_session, limit=5)
    assert isinstance(top_rc, list)

    # 5. Score summary with date filters
    summary = get_score_summary(db_session, start_date=start_date, end_date=end_date)
    assert "total_conversations" in summary
    assert summary["total_conversations"] >= 1
    assert "avg_composite" in summary
    assert "avg_accuracy" in summary
