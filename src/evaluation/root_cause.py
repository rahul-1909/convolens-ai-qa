"""Root-cause attribution engine.

Maps detected failures to their most likely originating layer:
  prompt | kb_gap | tool_failure | context_management | asr | tts | workflow

Uses a signal-matching decision tree rather than ML so results are
deterministic and auditable.
"""

from __future__ import annotations

from src.schemas.conversation import Turn, SpeakerRole
from src.schemas.evaluation import (
    FailureCategoryCode,
    FailureRecord,
    RootCauseAttribution,
    RootCauseCode,
    Severity,
)


class RootCauseAnalyzer:
    """Deterministic root-cause classifier for conversation failures.

    For each failure, the analyser checks a series of contextual signals
    (KB availability, tool success/failure, ASR confidence, …) and selects
    the most likely layer.
    """

    def analyze(
        self,
        failure: FailureRecord,
        turn: Turn | None = None,
        all_turns: list[Turn] | None = None,
    ) -> RootCauseAttribution:
        """Attribute a single failure to its root-cause layer.

        Args:
            failure: The detected failure record.
            turn: The specific turn where the failure occurred (if known).
            all_turns: Full conversation turns for context analysis.

        Returns:
            A ``RootCauseAttribution`` with the layer, confidence, and signals.
        """
        # Dispatch based on failure category
        category = failure.category

        if category == FailureCategoryCode.HALLUCINATION:
            return self._analyze_hallucination(failure, turn)

        if category == FailureCategoryCode.CONTEXT_LOSS:
            return self._analyze_context_loss(failure, turn, all_turns)

        if category == FailureCategoryCode.LOOPS_REPETITION:
            return self._analyze_loops(failure, turn, all_turns)

        if category == FailureCategoryCode.INTENT_MISREAD:
            return self._analyze_intent_misread(failure, turn)

        if category == FailureCategoryCode.WEAK_OBJECTION:
            return RootCauseAttribution(
                root_cause=RootCauseCode.PROMPT,
                confidence=0.7,
                signals=["Objection handling is typically governed by prompt instructions"],
                recommended_fix="Add explicit objection-handling scripts to the system prompt.",
            )

        if category == FailureCategoryCode.INSTRUCTION_VIOLATION:
            return RootCauseAttribution(
                root_cause=RootCauseCode.PROMPT,
                confidence=0.85,
                signals=["Agent violated its own instructions — prompt clarity issue"],
                recommended_fix="Clarify or strengthen the violated instruction in the system prompt.",
            )

        if category == FailureCategoryCode.GOAL_FAILURE:
            return self._analyze_goal_failure(failure, turn, all_turns)

        if category == FailureCategoryCode.ASR_ERROR:
            return RootCauseAttribution(
                root_cause=RootCauseCode.ASR,
                confidence=0.90,
                signals=["ASR-layer failure detected directly"],
                recommended_fix="Review ASR model configuration; consider domain-specific fine-tuning.",
            )

        if category == FailureCategoryCode.TTS_MISPRONUNCIATION:
            return RootCauseAttribution(
                root_cause=RootCauseCode.TTS,
                confidence=0.90,
                signals=["TTS-layer failure detected directly"],
                recommended_fix="Add custom pronunciation lexicon entries for affected terms.",
            )

        if category == FailureCategoryCode.BARGEIN_INTERRUPTION:
            return RootCauseAttribution(
                root_cause=RootCauseCode.WORKFLOW,
                confidence=0.70,
                signals=["Turn-taking / VAD configuration issue"],
                recommended_fix="Tune voice activity detection and barge-in sensitivity thresholds.",
            )

        # Fallback
        return RootCauseAttribution(
            root_cause=RootCauseCode.PROMPT,
            confidence=0.50,
            signals=["No specific signal matched; defaulting to prompt layer"],
            recommended_fix="Review conversation logs manually for this failure type.",
        )

    def analyze_all(
        self,
        failures: list[FailureRecord],
        turns: list[Turn],
    ) -> list[RootCauseAttribution]:
        """Attribute root causes for a batch of failures.

        Args:
            failures: All detected failures in the conversation.
            turns: Full list of conversation turns.

        Returns:
            One ``RootCauseAttribution`` per failure (same order).
        """
        turn_map = {t.turn_id: t for t in turns}
        results = []
        for failure in failures:
            # Try to find the originating turn from evidence
            related_turn = None
            for t in turns:
                if t.transcript and t.transcript[:80] in failure.evidence_quote:
                    related_turn = t
                    break
            results.append(self.analyze(failure, related_turn, turns))
        return results

    # ── Private Analysers ────────────────────────────────────────────────

    def _analyze_hallucination(
        self, failure: FailureRecord, turn: Turn | None
    ) -> RootCauseAttribution:
        """Determine whether hallucination stems from KB gap, tool failure, or prompt."""
        if turn is None:
            return RootCauseAttribution(
                root_cause=RootCauseCode.PROMPT,
                confidence=0.60,
                signals=["No turn context available for deeper analysis"],
                recommended_fix="Add grounding instructions to the system prompt.",
            )

        # Signal 1: Tool called but failed → tool_failure
        failed_tools = [tc for tc in turn.tool_calls if not tc.success]
        if failed_tools:
            return RootCauseAttribution(
                root_cause=RootCauseCode.TOOL_FAILURE,
                confidence=0.85,
                signals=[
                    f"Tool '{failed_tools[0].tool_name}' returned failure",
                    "Agent may have fabricated data because tool data was unavailable",
                ],
                recommended_fix=(
                    f"Fix the '{failed_tools[0].tool_name}' tool reliability. "
                    f"Add fallback behaviour when tools fail."
                ),
            )

        # Signal 2: No KB snippets and no tools → KB gap
        if not turn.kb_snippets and not turn.tool_calls:
            return RootCauseAttribution(
                root_cause=RootCauseCode.KB_GAP,
                confidence=0.75,
                signals=[
                    "No knowledge-base snippets were retrieved for this turn",
                    "No tool calls made — agent had no grounding source",
                ],
                recommended_fix=(
                    "Add the missing information to the knowledge base. "
                    "Ensure retrieval pipeline covers this topic."
                ),
            )

        # Signal 3: KB present but agent still hallucinated → prompt issue
        return RootCauseAttribution(
            root_cause=RootCauseCode.PROMPT,
            confidence=0.70,
            signals=[
                "KB snippets were available but agent still made an ungrounded claim",
                "Prompt may lack explicit grounding instructions",
            ],
            recommended_fix=(
                "Strengthen grounding instructions: 'Only state facts from the "
                "provided context. If unsure, say you will verify.'"
            ),
        )

    def _analyze_context_loss(
        self,
        failure: FailureRecord,
        turn: Turn | None,
        all_turns: list[Turn] | None,
    ) -> RootCauseAttribution:
        """Context loss is usually context-management or workflow."""
        signals = []
        if all_turns and len(all_turns) > 15:
            signals.append(
                f"Conversation is long ({len(all_turns)} turns) — "
                f"possible context window overflow"
            )
            return RootCauseAttribution(
                root_cause=RootCauseCode.CONTEXT_MANAGEMENT,
                confidence=0.80,
                signals=signals,
                recommended_fix=(
                    "Implement conversation summarisation for long dialogues. "
                    "Consider slot-filling to persist key values."
                ),
            )

        return RootCauseAttribution(
            root_cause=RootCauseCode.CONTEXT_MANAGEMENT,
            confidence=0.70,
            signals=["Agent lost track of previously stated information"],
            recommended_fix=(
                "Add explicit slot-tracking to the prompt. List collected "
                "values at each turn."
            ),
        )

    def _analyze_loops(
        self,
        failure: FailureRecord,
        turn: Turn | None,
        all_turns: list[Turn] | None,
    ) -> RootCauseAttribution:
        """Loops can be prompt, workflow, or context-management."""
        if all_turns:
            # Check if any tool call failed near the loop
            for t in (all_turns or []):
                if any(not tc.success for tc in t.tool_calls):
                    return RootCauseAttribution(
                        root_cause=RootCauseCode.TOOL_FAILURE,
                        confidence=0.75,
                        signals=["A tool failure near the loop may have caused retry behaviour"],
                        recommended_fix="Add retry limits and fallback paths when tools fail.",
                    )

        return RootCauseAttribution(
            root_cause=RootCauseCode.WORKFLOW,
            confidence=0.70,
            signals=["Agent repeated itself without making progress"],
            recommended_fix=(
                "Add loop detection to the orchestration layer. "
                "Set a max-retry count and escalation path."
            ),
        )

    def _analyze_intent_misread(
        self, failure: FailureRecord, turn: Turn | None
    ) -> RootCauseAttribution:
        """Intent misread: ASR (if voice) or prompt (if chat)."""
        if turn and turn.asr_confidence is not None and turn.asr_confidence < 0.65:
            return RootCauseAttribution(
                root_cause=RootCauseCode.ASR,
                confidence=0.80,
                signals=[
                    f"Low ASR confidence ({turn.asr_confidence:.2f}) on user turn",
                    "Agent may have misread intent because transcription was poor",
                ],
                recommended_fix="Improve ASR accuracy; ask user to repeat when confidence is low.",
            )

        return RootCauseAttribution(
            root_cause=RootCauseCode.PROMPT,
            confidence=0.65,
            signals=["Agent misclassified user intent despite clear input"],
            recommended_fix=(
                "Add more intent examples to the prompt. "
                "Consider adding a confirmation step for ambiguous inputs."
            ),
        )

    def _analyze_goal_failure(
        self,
        failure: FailureRecord,
        turn: Turn | None,
        all_turns: list[Turn] | None,
    ) -> RootCauseAttribution:
        """Goal failure — check all layers."""
        # Check for tool failures across conversation
        if all_turns:
            tool_failures = sum(
                1 for t in all_turns for tc in t.tool_calls if not tc.success
            )
            if tool_failures > 0:
                return RootCauseAttribution(
                    root_cause=RootCauseCode.TOOL_FAILURE,
                    confidence=0.75,
                    signals=[f"{tool_failures} tool failure(s) during conversation"],
                    recommended_fix="Fix tool reliability; the goal may have been unachievable.",
                )

        return RootCauseAttribution(
            root_cause=RootCauseCode.WORKFLOW,
            confidence=0.65,
            signals=["Conversation ended without achieving its stated goal"],
            recommended_fix=(
                "Review the conversation flow design. Ensure goal-completion "
                "checks exist before ending the conversation."
            ),
        )
