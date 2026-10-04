"""Extended tests for RootCauseAnalyzer and HallucinationDetector edge cases."""

from __future__ import annotations

import pytest

from src.evaluation.root_cause import RootCauseAnalyzer
from src.evaluation.hallucination import HallucinationDetector
from src.schemas.conversation import Turn, SpeakerRole, ToolCall
from src.schemas.evaluation import (
    FailureCategoryCode,
    FailureRecord,
    RootCauseCode,
    Severity,
)


def test_root_cause_weak_objection():
    """Weak objection failures should map to RC-PROMPT."""
    analyzer = RootCauseAnalyzer()
    failure = FailureRecord(
        category=FailureCategoryCode.WEAK_OBJECTION,
        severity=Severity.S3,
        evidence_quote="Customer said fees are too high",
    )
    result = analyzer.analyze(failure)
    assert result.root_cause == RootCauseCode.PROMPT


def test_root_cause_tts_and_bargein():
    """TTS mispronunciation and barge-in should map to TTS and WORKFLOW respectively."""
    analyzer = RootCauseAnalyzer()

    tts_fail = FailureRecord(
        category=FailureCategoryCode.TTS_MISPRONUNCIATION,
        severity=Severity.S3,
    )
    assert analyzer.analyze(tts_fail).root_cause == RootCauseCode.TTS

    bargein_fail = FailureRecord(
        category=FailureCategoryCode.BARGEIN_INTERRUPTION,
        severity=Severity.S3,
    )
    assert analyzer.analyze(bargein_fail).root_cause == RootCauseCode.WORKFLOW


def test_root_cause_loops_with_tool_failure():
    """Loops occurring alongside tool failures should attribute to RC-TOOL."""
    analyzer = RootCauseAnalyzer()
    failure = FailureRecord(
        category=FailureCategoryCode.LOOPS_REPETITION,
        severity=Severity.S2,
    )
    turns = [
        Turn(
            turn_id="t1",
            turn_index=0,
            speaker=SpeakerRole.AGENT,
            transcript="Retrying operation...",
            tool_calls=[
                ToolCall(tool_name="api_call", success=False, output_payload={"error": "timeout"})
            ],
        ),
        Turn(
            turn_id="t2",
            turn_index=1,
            speaker=SpeakerRole.AGENT,
            transcript="Retrying operation...",
        ),
    ]
    result = analyzer.analyze(failure, turns[1], turns)
    assert result.root_cause == RootCauseCode.TOOL_FAILURE


def test_root_cause_goal_failure_with_tool_failure():
    """Goal failure with tool failures should attribute to RC-TOOL."""
    analyzer = RootCauseAnalyzer()
    failure = FailureRecord(
        category=FailureCategoryCode.GOAL_FAILURE,
        severity=Severity.S1,
    )
    turns = [
        Turn(
            turn_id="t1",
            turn_index=0,
            speaker=SpeakerRole.AGENT,
            transcript="Connecting...",
            tool_calls=[ToolCall(tool_name="verify", success=False)],
        )
    ]
    result = analyzer.analyze(failure, turns[0], turns)
    assert result.root_cause == RootCauseCode.TOOL_FAILURE


def test_hallucination_with_and_without_tool_calls():
    """Test hallucination heuristics for authority phrases with and without tool calls."""
    detector = HallucinationDetector()

    # Agent claims authority WITHOUT tool calls -> flagged
    turn_no_tool = Turn(
        turn_id="t-auth-1",
        turn_index=0,
        speaker=SpeakerRole.AGENT,
        transcript="As per our records, you are eligible for Rs. 50,000 credit limit.",
        tool_calls=[],
    )
    failures = detector.check_turn(turn_no_tool)
    assert len(failures) >= 1
    assert any(f.category == FailureCategoryCode.HALLUCINATION for f in failures)

    # Customer speaker -> never flagged
    cust_turn = Turn(
        turn_id="t-cust",
        turn_index=1,
        speaker=SpeakerRole.CUSTOMER,
        transcript="As per our records I paid 50%",
    )
    assert len(detector.check_turn(cust_turn)) == 0
