"""Hallucination detection — compares agent claims against grounded sources.

Works in two modes:
1. **Standalone (rule-based)**: Simple keyword/entity checks.
2. **LLM-assisted**: Invoked as part of the LLM judge prompt (see ``llm_judge.py``).
"""

from __future__ import annotations

from dataclasses import dataclass

from src.schemas.conversation import Turn, SpeakerRole
from src.schemas.evaluation import FailureRecord, FailureCategoryCode, Severity


@dataclass
class HallucinationDetector:
    """Checks agent turns for claims not grounded in KB or tool outputs.

    This is the *rule-based* layer. The LLM judge performs deeper semantic
    grounding checks in parallel.
    """

    def check_turn(self, turn: Turn) -> list[FailureRecord]:
        """Run lightweight hallucination heuristics on a single agent turn.

        Args:
            turn: The turn to analyse.

        Returns:
            List of potential hallucination flags (may be empty).
        """
        if turn.speaker != SpeakerRole.AGENT:
            return []

        failures: list[FailureRecord] = []

        # Heuristic 1: Agent mentions specific numbers/percentages but
        # no KB snippet or tool output is present to ground them.
        if self._has_specific_numbers(turn.transcript) and not (
            turn.kb_snippets or turn.tool_calls
        ):
            failures.append(
                FailureRecord(
                    category=FailureCategoryCode.HALLUCINATION,
                    subtype="numerical_error",
                    severity=Severity.S2,
                    evidence_quote=turn.transcript[:200],
                    reasoning=(
                        "Agent cites specific numbers/percentages but no "
                        "KB snippets or tool outputs are available this turn "
                        "to ground the claim."
                    ),
                )
            )

        # Heuristic 2: Agent says "as per our records" or similar authority
        # phrases without any tool call backing.
        authority_phrases = [
            "as per our records",
            "according to our system",
            "our records show",
            "i can confirm that",
            "the system shows",
        ]
        lower = turn.transcript.lower()
        for phrase in authority_phrases:
            if phrase in lower and not turn.tool_calls:
                failures.append(
                    FailureRecord(
                        category=FailureCategoryCode.HALLUCINATION,
                        subtype="factual_fabrication",
                        severity=Severity.S2,
                        evidence_quote=(
                            f'Agent used authority phrase "{phrase}" without '
                            f"any tool call to verify the claim."
                        ),
                        reasoning=(
                            "Agent asserts system-backed information but made "
                            "no tool/API call this turn to retrieve it."
                        ),
                    )
                )
                break  # one flag per turn is sufficient

        return failures

    # ── Private helpers ──────────────────────────────────────────────────

    @staticmethod
    def _has_specific_numbers(text: str) -> bool:
        """Return True if text contains currency amounts, percentages, or dates."""
        import re

        patterns = [
            r"\b\d+(?:,\d{3})*(?:\.\d+)?\s*%",           # percentages
            r"(?:Rs\.?|₹|INR|\$|USD)\s*\d+",              # currency
            r"\b\d{1,2}[/\-]\d{1,2}[/\-]\d{2,4}\b",      # dates
            r"\b\d{4,}\b",                                 # large numbers (IDs, amounts)
        ]
        return any(re.search(p, text) for p in patterns)
