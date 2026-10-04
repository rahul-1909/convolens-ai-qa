"""Pydantic schemas for evaluation results, failure categories, and root causes."""

from __future__ import annotations

from datetime import datetime, timezone
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


# ── Enumerations ─────────────────────────────────────────────────────────────


class Severity(str, Enum):
    """Failure severity classification."""

    S1 = "S1"  # Critical — conversation fails
    S2 = "S2"  # Major — goal at risk
    S3 = "S3"  # Moderate — poor experience
    S4 = "S4"  # Minor — cosmetic / low impact


class FailureCategoryCode(str, Enum):
    """Level-1 and Level-2 failure category codes."""

    # Level 1
    HALLUCINATION = "L1-HAL"
    CONTEXT_LOSS = "L1-CTX"
    LOOPS_REPETITION = "L1-LOOP"
    INTENT_MISREAD = "L1-INT"
    WEAK_OBJECTION = "L1-OBJ"
    INSTRUCTION_VIOLATION = "L1-INS"
    GOAL_FAILURE = "L1-GOAL"
    # Level 2
    ASR_ERROR = "L2-ASR"
    TTS_MISPRONUNCIATION = "L2-TTS"
    BARGEIN_INTERRUPTION = "L2-BARG"


class RootCauseCode(str, Enum):
    """Root-cause attribution layer codes."""

    PROMPT = "RC-PROMPT"
    KB_GAP = "RC-KB"
    TOOL_FAILURE = "RC-TOOL"
    CONTEXT_MANAGEMENT = "RC-CONTEXT"
    ASR = "RC-ASR"
    TTS = "RC-TTS"
    WORKFLOW = "RC-WORKFLOW"


# ── Scoring ──────────────────────────────────────────────────────────────────


class DimensionScore(BaseModel):
    """Score for a single rubric dimension."""

    dimension: str = Field(
        ..., description="accuracy | empathy | flow | goal_completion"
    )
    score: int = Field(..., ge=1, le=5, description="1-5 score")
    rationale: str = Field("", description="Brief reasoning for this score")


class RubricScores(BaseModel):
    """Complete rubric score set for a turn or conversation."""

    accuracy: int = Field(..., ge=1, le=5)
    empathy: int = Field(..., ge=1, le=5)
    flow: int = Field(..., ge=1, le=5)
    goal_completion: int = Field(..., ge=1, le=5)
    composite: float = Field(
        ..., ge=1.0, le=5.0, description="Weighted composite score"
    )

    @classmethod
    def compute(
        cls,
        accuracy: int,
        empathy: int,
        flow: int,
        goal_completion: int,
    ) -> "RubricScores":
        """Create scores with auto-computed weighted composite."""
        composite = round(
            accuracy * 0.35
            + empathy * 0.20
            + flow * 0.20
            + goal_completion * 0.25,
            2,
        )
        return cls(
            accuracy=accuracy,
            empathy=empathy,
            flow=flow,
            goal_completion=goal_completion,
            composite=composite,
        )


# ── Failure Records ──────────────────────────────────────────────────────────


class FailureRecord(BaseModel):
    """A single detected failure in a turn."""

    category: FailureCategoryCode
    subtype: str | None = Field(
        None, description="Specific subtype within the category"
    )
    severity: Severity
    evidence_quote: str = Field(
        "", description="Exact quote from transcript as evidence"
    )
    reasoning: str = Field(
        "", description="Chain-of-thought reasoning for this failure"
    )


class RootCauseAttribution(BaseModel):
    """Root-cause analysis for a failure."""

    root_cause: RootCauseCode
    confidence: float = Field(
        ..., ge=0.0, le=1.0, description="Confidence in attribution"
    )
    signals: list[str] = Field(
        default_factory=list,
        description="Evidence signals supporting this attribution",
    )
    recommended_fix: str = Field("", description="Suggested remediation")


# ── Turn-Level Result ────────────────────────────────────────────────────────


class TurnEvaluationResult(BaseModel):
    """Complete evaluation result for a single turn."""

    turn_id: str
    turn_index: int
    scores: RubricScores
    failures: list[FailureRecord] = Field(default_factory=list)
    root_causes: list[RootCauseAttribution] = Field(default_factory=list)
    is_flagged: bool = Field(
        False, description="True if any S1 or S2 failure detected"
    )


# ── Conversation-Level Result ────────────────────────────────────────────────


class ConversationEvaluationResult(BaseModel):
    """Complete evaluation result for an entire conversation."""

    conversation_id: str
    overall_scores: RubricScores
    turn_results: list[TurnEvaluationResult] = Field(default_factory=list)
    conversation_failures: list[FailureRecord] = Field(default_factory=list)
    root_causes: list[RootCauseAttribution] = Field(default_factory=list)
    recommended_fixes: list[str] = Field(default_factory=list)
    evaluated_at: datetime = Field(
        default_factory=lambda: datetime.now(timezone.utc)
    )
    evaluator_model: str = Field("", description="LLM model used for judging")
    metadata: dict | None = None
