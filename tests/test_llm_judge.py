"""Tests for LLM Judge (fallback mode — no API key needed)."""

from __future__ import annotations

import os

from src.evaluation.llm_judge import LLMJudge


class TestLLMJudgeFallback:
    """Tests for the LLM judge operating in heuristic fallback mode."""

    def setup_method(self):
        # Ensure no API key so we test the fallback path
        self.judge = LLMJudge(api_key="")

    def test_evaluate_turn_fallback(self, sample_conversation):
        """Without an API key, judge should return fallback neutral scores."""
        turn = sample_conversation.turns[0]  # Agent greeting
        result = self.judge.evaluate_turn(turn, sample_conversation)
        assert result.turn_id == "t1"
        assert 1 <= result.scores.accuracy <= 5
        assert 1 <= result.scores.composite <= 5

    def test_evaluate_conversation_fallback(self, sample_conversation):
        """Conversation-level fallback should aggregate from turns."""
        # First get turn results
        turn_results = []
        for turn in sample_conversation.turns:
            result = self.judge.evaluate_turn(turn, sample_conversation)
            turn_results.append(result)

        conv_result = self.judge.evaluate_conversation(
            sample_conversation, turn_results
        )
        assert "accuracy" in conv_result
        assert "empathy" in conv_result

    def test_parse_json_with_markdown_fences(self):
        """JSON parser should handle markdown code fences."""
        raw = '```json\n{"accuracy": 4, "empathy": 3}\n```'
        parsed = LLMJudge._parse_json(raw)
        assert parsed["accuracy"] == 4

    def test_parse_failures_from_raw(self):
        """Failure parser should handle both code and name formats."""
        raw_failures = [
            {
                "category": "hallucination",
                "severity": "S1",
                "evidence_quote": "Agent said wrong rate",
                "reasoning": "Rate was 11.5 not 10",
            },
            {
                "category": "L1-CTX",
                "severity": "S2",
                "reasoning": "Lost customer name",
            },
        ]
        failures = LLMJudge._parse_failures(raw_failures)
        assert len(failures) == 2
        assert failures[0].category.value == "L1-HAL"
        assert failures[1].category.value == "L1-CTX"

    def test_clamp_values(self):
        """Clamp should restrict to [1, 5] range."""
        assert LLMJudge._clamp(0) == 1
        assert LLMJudge._clamp(6) == 5
        assert LLMJudge._clamp(3) == 3
        assert LLMJudge._clamp("4") == 4
        assert LLMJudge._clamp(None) == 3
