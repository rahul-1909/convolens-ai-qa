"""Pydantic schemas for API request/response payloads."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

from pydantic import BaseModel, Field

from src.schemas.conversation import Conversation
from src.schemas.evaluation import (
    ConversationEvaluationResult,
    FailureCategoryCode,
    Severity,
)


# ── Request Models ───────────────────────────────────────────────────────────


class EvaluateConversationRequest(BaseModel):
    """POST /evaluate/conversation request body."""

    conversation: Conversation
    run_pattern_mining: bool = Field(
        False,
        description="If True, also embed failed turns for clustering",
    )


class TrendQuery(BaseModel):
    """Query parameters for trend analysis."""

    start_date: datetime | None = None
    end_date: datetime | None = None
    customer_id: str | None = None
    agent_version: str | None = None
    category: FailureCategoryCode | None = None
    severity: Severity | None = None
    limit: int = Field(100, ge=1, le=1000)


# ── Response Models ──────────────────────────────────────────────────────────


class EvaluateConversationResponse(BaseModel):
    """POST /evaluate/conversation response body."""

    status: str = "success"
    conversation_id: str
    result: ConversationEvaluationResult


class ConversationResultResponse(BaseModel):
    """GET /results/{conversation_id} response body."""

    conversation_id: str
    result: ConversationEvaluationResult | None = None
    found: bool = True


class FailureDistribution(BaseModel):
    """A single row in the failure distribution report."""

    category: str
    severity: str
    count: int
    percentage: float


class TrendResponse(BaseModel):
    """GET /trends response body."""

    total_conversations: int
    total_failures: int
    average_composite_score: float
    failure_distribution: list[FailureDistribution]
    top_root_causes: list[dict]


class HealthResponse(BaseModel):
    """GET /health response."""

    status: str = "healthy"
    version: str
    database: str = "connected"
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
