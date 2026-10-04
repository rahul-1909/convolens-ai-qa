"""API routes for retrieving evaluation results."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.db.session import get_db_dependency
from src.models.database import (
    ConversationRecord,
    EvaluationRecord,
    FailureRecord_DB,
    TurnRecord,
    TurnEvaluationRecord,
)
from src.schemas.api import ConversationResultResponse
from src.schemas.evaluation import (
    ConversationEvaluationResult,
    FailureRecord,
    FailureCategoryCode,
    RootCauseAttribution,
    RootCauseCode,
    RubricScores,
    Severity,
    TurnEvaluationResult,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/results", tags=["results"])


@router.get("/{conversation_id}", response_model=ConversationResultResponse)
def get_results(
    conversation_id: str,
    db: Session = Depends(get_db_dependency),
) -> ConversationResultResponse:
    """Retrieve evaluation results for a specific conversation.

    Args:
        conversation_id: The conversation to look up.
        db: Database session (injected).

    Returns:
        Scores, failures, root causes, and recommended fixes.
    """
    eval_rec = (
        db.query(EvaluationRecord)
        .filter(EvaluationRecord.conversation_id == conversation_id)
        .first()
    )

    if not eval_rec:
        return ConversationResultResponse(
            conversation_id=conversation_id,
            found=False,
            result=None,
        )

    # Build overall scores
    overall_scores = RubricScores.compute(
        accuracy=eval_rec.accuracy_score,
        empathy=eval_rec.empathy_score,
        flow=eval_rec.flow_score,
        goal_completion=eval_rec.goal_completion_score,
    )

    # Get turn evaluations
    turn_recs = (
        db.query(TurnRecord)
        .filter(TurnRecord.conversation_id == conversation_id)
        .order_by(TurnRecord.turn_index)
        .all()
    )

    turn_results = []
    for tr in turn_recs:
        te = (
            db.query(TurnEvaluationRecord)
            .filter(TurnEvaluationRecord.turn_id == tr.id)
            .first()
        )
        if te:
            turn_results.append(
                TurnEvaluationResult(
                    turn_id=tr.id,
                    turn_index=tr.turn_index,
                    scores=RubricScores.compute(
                        te.accuracy_score,
                        te.empathy_score,
                        te.flow_score,
                        te.goal_completion_score,
                    ),
                    failures=[],
                    is_flagged=te.is_flagged,
                )
            )

    # Get failures
    failure_recs = (
        db.query(FailureRecord_DB)
        .filter(FailureRecord_DB.evaluation_id == eval_rec.id)
        .all()
    )

    failures = []
    root_causes = []
    for fr in failure_recs:
        cat_map = {c.value: c for c in FailureCategoryCode}
        category = cat_map.get(fr.category, FailureCategoryCode.HALLUCINATION)
        sev_map = {s.value: s for s in Severity}
        severity = sev_map.get(fr.severity, Severity.S3)

        failures.append(
            FailureRecord(
                category=category,
                subtype=fr.subtype,
                severity=severity,
                evidence_quote=fr.evidence_quote or "",
                reasoning=fr.reasoning or "",
            )
        )

        if fr.root_cause:
            rc_map = {c.value: c for c in RootCauseCode}
            rc_code = rc_map.get(fr.root_cause, RootCauseCode.PROMPT)
            root_causes.append(
                RootCauseAttribution(
                    root_cause=rc_code,
                    confidence=fr.root_cause_confidence or 0.5,
                    signals=fr.root_cause_signals_json or [],
                    recommended_fix=fr.recommended_fix or "",
                )
            )

    result = ConversationEvaluationResult(
        conversation_id=conversation_id,
        overall_scores=overall_scores,
        turn_results=turn_results,
        conversation_failures=failures,
        root_causes=root_causes,
        recommended_fixes=eval_rec.recommended_fixes_json or [],
        evaluated_at=eval_rec.evaluated_at,
        evaluator_model=eval_rec.evaluator_model or "",
    )

    return ConversationResultResponse(
        conversation_id=conversation_id,
        found=True,
        result=result,
    )
