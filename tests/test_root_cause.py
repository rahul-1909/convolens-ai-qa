"""Tests for root-cause analyzer."""

from __future__ import annotations

from src.evaluation.root_cause import RootCauseAnalyzer
from src.schemas.conversation import Turn, SpeakerRole, ToolCall
from src.schemas.evaluation import (
    FailureCategoryCode,
    FailureRecord,
    RootCauseCode,
    Severity,
)


class TestRootCauseAnalyzer:
    """Tests for the deterministic root-cause attribution engine."""

    def setup_method(self):
        self.analyzer = RootCauseAnalyzer()

    def test_hallucination_with_failed_tool(self):
        """Hallucination + failed tool → RC-TOOL."""
        failure = FailureRecord(
            category=FailureCategoryCode.HALLUCINATION,
            severity=Severity.S1,
            evidence_quote="Agent stated wrong amount",
        )
        turn = Turn(
            turn_id="t1",
            turn_index=0,
            speaker=SpeakerRole.AGENT,
            transcript="Your balance is Rs. 50,000",
            tool_calls=[
                ToolCall(
                    tool_name="get_balance",
                    success=False,
                    output_payload={"error": "timeout"},
                )
            ],
        )
        result = self.analyzer.analyze(failure, turn)
        assert result.root_cause == RootCauseCode.TOOL_FAILURE
        assert result.confidence > 0.7

    def test_hallucination_no_kb_no_tools(self):
        """Hallucination with no grounding sources → RC-KB."""
        failure = FailureRecord(
            category=FailureCategoryCode.HALLUCINATION,
            severity=Severity.S2,
        )
        turn = Turn(
            turn_id="t1",
            turn_index=0,
            speaker=SpeakerRole.AGENT,
            transcript="The interest rate is 8.5%",
        )
        result = self.analyzer.analyze(failure, turn)
        assert result.root_cause == RootCauseCode.KB_GAP

    def test_hallucination_kb_present(self):
        """Hallucination despite KB → RC-PROMPT."""
        failure = FailureRecord(
            category=FailureCategoryCode.HALLUCINATION,
            severity=Severity.S1,
        )
        turn = Turn(
            turn_id="t1",
            turn_index=0,
            speaker=SpeakerRole.AGENT,
            transcript="Your rate is 10%",
            kb_snippets=["Interest rate is 11.5%"],
        )
        result = self.analyzer.analyze(failure, turn)
        assert result.root_cause == RootCauseCode.PROMPT

    def test_context_loss_long_conversation(self):
        """Long conversations → context management issue."""
        failure = FailureRecord(
            category=FailureCategoryCode.CONTEXT_LOSS,
            severity=Severity.S2,
        )
        # Create 20 turns to trigger long-conversation signal
        turns = [
            Turn(turn_id=f"t{i}", turn_index=i, speaker=SpeakerRole.AGENT, transcript=f"Turn {i}")
            for i in range(20)
        ]
        result = self.analyzer.analyze(failure, turns[10], turns)
        assert result.root_cause == RootCauseCode.CONTEXT_MANAGEMENT

    def test_asr_error_direct(self):
        """ASR errors map directly to RC-ASR."""
        failure = FailureRecord(
            category=FailureCategoryCode.ASR_ERROR,
            severity=Severity.S3,
        )
        result = self.analyzer.analyze(failure)
        assert result.root_cause == RootCauseCode.ASR
        assert result.confidence == 0.90

    def test_instruction_violation(self):
        """Instruction violations → RC-PROMPT."""
        failure = FailureRecord(
            category=FailureCategoryCode.INSTRUCTION_VIOLATION,
            severity=Severity.S1,
        )
        result = self.analyzer.analyze(failure)
        assert result.root_cause == RootCauseCode.PROMPT

    def test_intent_misread_low_asr(self):
        """Intent misread + low ASR → RC-ASR."""
        failure = FailureRecord(
            category=FailureCategoryCode.INTENT_MISREAD,
            severity=Severity.S2,
        )
        turn = Turn(
            turn_id="t1",
            turn_index=0,
            speaker=SpeakerRole.CUSTOMER,
            transcript="I want to close my account",
            asr_confidence=0.45,
        )
        result = self.analyzer.analyze(failure, turn)
        assert result.root_cause == RootCauseCode.ASR

    def test_analyze_all(self, sample_conversation):
        """analyze_all should produce one attribution per failure."""
        failures = [
            FailureRecord(
                category=FailureCategoryCode.HALLUCINATION,
                severity=Severity.S2,
                evidence_quote="As per our records",
            ),
            FailureRecord(
                category=FailureCategoryCode.ASR_ERROR,
                severity=Severity.S3,
                evidence_quote="Low confidence turn",
            ),
        ]
        results = self.analyzer.analyze_all(failures, sample_conversation.turns)
        assert len(results) == 2
