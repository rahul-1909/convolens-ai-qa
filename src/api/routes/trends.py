"""API routes for trend analysis and reporting."""

from __future__ import annotations

import logging
from datetime import datetime

from fastapi import APIRouter, Depends, Query
from sqlalchemy.orm import Session

from src.db.session import get_db_dependency
from src.schemas.api import FailureDistribution, TrendResponse
from src.mining.trend_detection import (
    get_failure_distribution,
    get_score_summary,
    get_top_root_causes,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/trends", tags=["trends"])


@router.get("", response_model=TrendResponse)
def get_trends(
    start_date: datetime | None = Query(None, description="Filter from this date"),
    end_date: datetime | None = Query(None, description="Filter until this date"),
    agent_version: str | None = Query(None, description="Filter by agent version"),
    customer_id: str | None = Query(None, description="Filter by customer"),
    db: Session = Depends(get_db_dependency),
) -> TrendResponse:
    """Return failure distribution, top root causes, and score summaries.

    Supports filtering by date range, agent version, and customer.

    Args:
        start_date: Optional lower-bound date filter.
        end_date: Optional upper-bound date filter.
        agent_version: Optional agent version filter.
        customer_id: Optional customer filter.
        db: Database session (injected).

    Returns:
        Aggregated trend data.
    """
    distribution_raw = get_failure_distribution(
        db,
        start_date=start_date,
        end_date=end_date,
        agent_version=agent_version,
        customer_id=customer_id,
    )

    total_failures = sum(d["count"] for d in distribution_raw)
    distribution = [
        FailureDistribution(
            category=d["category"],
            severity=d["severity"],
            count=d["count"],
            percentage=round(d["count"] / total_failures * 100, 1)
            if total_failures > 0
            else 0.0,
        )
        for d in distribution_raw
    ]

    score_summary = get_score_summary(
        db, start_date=start_date, end_date=end_date
    )
    top_root_causes = get_top_root_causes(db, limit=10)

    return TrendResponse(
        total_conversations=score_summary.get("total_conversations", 0),
        total_failures=total_failures,
        average_composite_score=score_summary.get("avg_composite", 0.0),
        failure_distribution=distribution,
        top_root_causes=top_root_causes,
    )
