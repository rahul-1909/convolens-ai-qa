"""Unit and integration tests for the evaluation pipeline."""

from __future__ import annotations

import json
from pathlib import Path
import pytest

from src.db.session import init_db
from src.evaluation.pipeline import EvaluationPipeline
from src.ingestion.normalizer import normalize_raw_transcript
from src.schemas.evaluation import FailureCategoryCode, Severity


def test_pipeline_evaluates_sample_file(db_session):
    """Pipeline should evaluate real-world sample conversation JSON end-to-end."""
    sample_file = (
        Path(__file__).resolve().parent.parent
        / "data"
        / "samples"
        / "sample_conversation.json"
    )
    with open(sample_file, "r", encoding="utf-8") as f:
        raw_data = json.load(f)

    conversation = normalize_raw_transcript(raw_data)
    assert len(conversation.turns) == 15

    pipeline = EvaluationPipeline(db=db_session)
    result = pipeline.evaluate(conversation)

    assert result.conversation_id == "sample-loan-collection-001"
    assert 1.0 <= result.overall_scores.composite <= 5.0
    assert len(result.turn_results) == 15
    assert len(result.conversation_failures) >= 0
    assert isinstance(result.root_causes, list)


def test_pipeline_detects_repetition_in_conversation(db_session, sample_conversation_with_repetition):
    """Pipeline should identify loops and repetition failures and assign root causes."""
    pipeline = EvaluationPipeline(db=db_session)
    result = pipeline.evaluate(sample_conversation_with_repetition)

    categories = [f.category for f in result.conversation_failures]
    assert FailureCategoryCode.LOOPS_REPETITION in categories
