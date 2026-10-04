"""API routes for conversation evaluation."""

from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from src.db.session import get_db_dependency
from src.schemas.api import (
    EvaluateConversationRequest,
    EvaluateConversationResponse,
)
from src.schemas.evaluation import ConversationEvaluationResult
from src.evaluation.pipeline import EvaluationPipeline
from src.ingestion.normalizer import normalize_raw_transcript

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/evaluate", tags=["evaluation"])


@router.post("/conversation", response_model=EvaluateConversationResponse)
def evaluate_conversation(
    request: EvaluateConversationRequest,
    db: Session = Depends(get_db_dependency),
) -> EvaluateConversationResponse:
    """Ingest a conversation and run the full evaluation pipeline.

    Accepts a conversation payload with turns, runs rule-based detectors,
    LLM judge, hallucination checks, and root-cause attribution. Results
    are persisted to the database.

    Args:
        request: Conversation payload + options.
        db: Database session (injected).

    Returns:
        Complete evaluation results.
    """
    try:
        pipeline = EvaluationPipeline(db=db)
        result = pipeline.evaluate(request.conversation)

        return EvaluateConversationResponse(
            status="success",
            conversation_id=request.conversation.conversation_id,
            result=result,
        )
    except Exception as e:
        logger.exception("Evaluation failed for conversation %s", request.conversation.conversation_id)
        raise HTTPException(
            status_code=500,
            detail=f"Evaluation failed: {str(e)}",
        )


@router.post("/raw", response_model=EvaluateConversationResponse)
def evaluate_raw_transcript(
    raw_data: dict,
    db: Session = Depends(get_db_dependency),
) -> EvaluateConversationResponse:
    """Accept a raw transcript (Deepgram/Whisper/simple JSON) and evaluate it.

    The raw data is first normalised into the ConvoLens schema before
    evaluation.

    Args:
        raw_data: Raw transcript dict in any supported format.
        db: Database session (injected).

    Returns:
        Complete evaluation results.
    """
    try:
        conversation = normalize_raw_transcript(raw_data)
        pipeline = EvaluationPipeline(db=db)
        result = pipeline.evaluate(conversation)

        return EvaluateConversationResponse(
            status="success",
            conversation_id=conversation.conversation_id,
            result=result,
        )
    except Exception as e:
        logger.exception("Raw evaluation failed")
        raise HTTPException(status_code=500, detail=str(e))
