"""Evaluation pipeline — orchestrates the full conversation scoring flow.

Steps:
  1. PII-mask transcripts (for LLM calls only; originals are stored).
  2. Run rule-based detectors.
  3. Run hallucination heuristics.
  4. Run LLM judge on each agent turn + conversation-level.
  5. Run root-cause attribution on all failures.
  6. Merge and deduplicate results.
  7. Persist to database.
"""

from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

from sqlalchemy.orm import Session

from src.config import get_settings
from src.schemas.conversation import Conversation, SpeakerRole
from src.schemas.evaluation import (
    ConversationEvaluationResult,
    FailureRecord,
    RootCauseAttribution,
    RubricScores,
    Severity,
    TurnEvaluationResult,
)
from src.evaluation.llm_judge import LLMJudge
from src.evaluation.rule_detectors import RuleBasedDetectors
from src.evaluation.hallucination import HallucinationDetector
from src.evaluation.root_cause import RootCauseAnalyzer
from src.models.database import (
    ConversationRecord,
    TurnRecord,
    EvaluationRecord,
    TurnEvaluationRecord,
    FailureRecord_DB,
)

logger = logging.getLogger(__name__)
_SETTINGS = get_settings()


class EvaluationPipeline:
    """End-to-end evaluation orchestrator.

    Args:
        db: SQLAlchemy session for persistence.
        judge: LLM judge instance (optional; creates default if missing).
    """

    def __init__(
        self,
        db: Session | None = None,
        judge: LLMJudge | None = None,
    ) -> None:
        self.db = db
        self.judge = judge or LLMJudge()
        self.rule_detectors = RuleBasedDetectors()
        self.hallucination_detector = HallucinationDetector()
        self.root_cause_analyzer = RootCauseAnalyzer()

    def evaluate(self, conversation: Conversation) -> ConversationEvaluationResult:
        """Run the complete evaluation pipeline on a conversation.

        Args:
            conversation: The conversation to evaluate.

        Returns:
            Full ``ConversationEvaluationResult`` with turn-level and
            conversation-level scores, failures, and root causes.
        """
        logger.info(
            "Starting evaluation for conversation %s (%d turns)",
            conversation.conversation_id,
            len(conversation.turns),
        )

        # Step 1: Rule-based detection
        rule_failures = self.rule_detectors.run_all(conversation.turns)
        logger.info("Rule detectors found %d issues", len(rule_failures))

        # Step 2: Hallucination heuristics
        hal_failures: list[FailureRecord] = []
        for turn in conversation.turns:
            hal_failures.extend(self.hallucination_detector.check_turn(turn))
        logger.info("Hallucination heuristics found %d issues", len(hal_failures))

        # Step 3: LLM judge — per-turn
        turn_results: list[TurnEvaluationResult] = []
        for turn in conversation.turns:
            if turn.speaker == SpeakerRole.AGENT:
                result = self.judge.evaluate_turn(turn, conversation)
                turn_results.append(result)
            else:
                # Score customer turns neutrally (they are inputs, not judged)
                turn_results.append(
                    TurnEvaluationResult(
                        turn_id=turn.turn_id,
                        turn_index=turn.turn_index,
                        scores=RubricScores.compute(3, 3, 3, 3),
                        failures=[],
                        is_flagged=False,
                    )
                )

        # Step 4: LLM judge — conversation-level
        conv_eval_raw = self.judge.evaluate_conversation(
            conversation, turn_results
        )

        # Step 5: Merge all failures
        all_failures: list[FailureRecord] = list(rule_failures) + list(hal_failures)
        for tr in turn_results:
            all_failures.extend(tr.failures)
        all_failures.extend(
            self.judge._parse_failures(conv_eval_raw.get("failures", []))
        )

        # Step 6: Root-cause attribution
        root_causes = self.root_cause_analyzer.analyze_all(
            all_failures, conversation.turns
        )

        # Also include LLM-suggested root causes
        llm_root_causes: list[RootCauseAttribution] = []
        for rc_raw in conv_eval_raw.get("root_causes", []):
            try:
                from src.schemas.evaluation import RootCauseCode

                code_map = {c.value: c for c in RootCauseCode}
                code_map.update({
                    "prompt": RootCauseCode.PROMPT,
                    "kb_gap": RootCauseCode.KB_GAP,
                    "tool_failure": RootCauseCode.TOOL_FAILURE,
                    "context_management": RootCauseCode.CONTEXT_MANAGEMENT,
                    "asr": RootCauseCode.ASR,
                    "tts": RootCauseCode.TTS,
                    "workflow": RootCauseCode.WORKFLOW,
                })
                rc_code = code_map.get(
                    rc_raw.get("root_cause", ""), RootCauseCode.PROMPT
                )
                llm_root_causes.append(
                    RootCauseAttribution(
                        root_cause=rc_code,
                        confidence=rc_raw.get("confidence", 0.5),
                        signals=rc_raw.get("signals", []),
                        recommended_fix=rc_raw.get("recommended_fix", ""),
                    )
                )
            except Exception:
                pass

        combined_root_causes = root_causes + llm_root_causes

        # Step 7: Build final scores
        overall_scores = RubricScores.compute(
            accuracy=self._clamp(conv_eval_raw.get("accuracy", 3)),
            empathy=self._clamp(conv_eval_raw.get("empathy", 3)),
            flow=self._clamp(conv_eval_raw.get("flow", 3)),
            goal_completion=self._clamp(conv_eval_raw.get("goal_completion", 3)),
        )

        result = ConversationEvaluationResult(
            conversation_id=conversation.conversation_id,
            overall_scores=overall_scores,
            turn_results=turn_results,
            conversation_failures=all_failures,
            root_causes=combined_root_causes,
            recommended_fixes=conv_eval_raw.get("recommended_fixes", []),
            evaluated_at=datetime.now(timezone.utc),
            evaluator_model=self.judge.model,
        )

        # Step 8: Persist
        if self.db is not None:
            self._persist(conversation, result)

        logger.info(
            "Evaluation complete for %s: composite=%.2f, failures=%d",
            conversation.conversation_id,
            result.overall_scores.composite,
            len(result.conversation_failures),
        )
        return result

    # ── Persistence ──────────────────────────────────────────────────────

    def _persist(
        self,
        conversation: Conversation,
        result: ConversationEvaluationResult,
    ) -> None:
        """Save conversation, turns, and evaluation results to the database."""
        try:
            # Clean up prior evaluation if re-evaluating the same conversation
            existing_eval = (
                self.db.query(EvaluationRecord)
                .filter(EvaluationRecord.conversation_id == conversation.conversation_id)
                .first()
            )
            if existing_eval:
                self.db.delete(existing_eval)
                self.db.flush()

            # Clean up prior turn evaluations if re-evaluating
            turn_ids = [t.turn_id for t in conversation.turns]
            if turn_ids:
                (
                    self.db.query(TurnEvaluationRecord)
                    .filter(TurnEvaluationRecord.turn_id.in_(turn_ids))
                    .delete(synchronize_session=False)
                )
                self.db.flush()

            # Conversation record
            conv_rec = ConversationRecord(
                id=conversation.conversation_id,
                customer_id=conversation.customer_id,
                agent_version=conversation.agent_version,
                prompt_version=conversation.prompt_version,
                conversation_goal=conversation.conversation_goal,
                channel=conversation.channel,
                language=conversation.language,
                started_at=conversation.started_at,
                ended_at=conversation.ended_at,
                metadata_json=conversation.metadata,
            )
            self.db.merge(conv_rec)

            # Turn records
            for turn in conversation.turns:
                turn_rec = TurnRecord(
                    id=turn.turn_id,
                    conversation_id=conversation.conversation_id,
                    turn_index=turn.turn_index,
                    speaker=turn.speaker.value,
                    transcript=turn.transcript,
                    timestamp=turn.timestamp,
                    duration_ms=turn.duration_ms,
                    asr_confidence=turn.asr_confidence,
                    tool_calls_json=(
                        [tc.model_dump() for tc in turn.tool_calls]
                        if turn.tool_calls
                        else None
                    ),
                    kb_snippets_json=turn.kb_snippets or None,
                )
                self.db.merge(turn_rec)

            # Evaluation record
            eval_id = str(uuid.uuid4())
            eval_rec = EvaluationRecord(
                id=eval_id,
                conversation_id=conversation.conversation_id,
                accuracy_score=result.overall_scores.accuracy,
                empathy_score=result.overall_scores.empathy,
                flow_score=result.overall_scores.flow,
                goal_completion_score=result.overall_scores.goal_completion,
                composite_score=result.overall_scores.composite,
                evaluator_model=result.evaluator_model,
                recommended_fixes_json=result.recommended_fixes,
                evaluated_at=result.evaluated_at,
            )
            self.db.add(eval_rec)
            self.db.flush()

            # Turn evaluation records
            for tr in result.turn_results:
                te_rec = TurnEvaluationRecord(
                    id=str(uuid.uuid4()),
                    turn_id=tr.turn_id,
                    accuracy_score=tr.scores.accuracy,
                    empathy_score=tr.scores.empathy,
                    flow_score=tr.scores.flow,
                    goal_completion_score=tr.scores.goal_completion,
                    composite_score=tr.scores.composite,
                    is_flagged=tr.is_flagged,
                )
                self.db.add(te_rec)

            # Failure records
            for i, failure in enumerate(result.conversation_failures):
                rc = (
                    result.root_causes[i]
                    if i < len(result.root_causes)
                    else None
                )
                fail_rec = FailureRecord_DB(
                    id=str(uuid.uuid4()),
                    evaluation_id=eval_id,
                    category=failure.category.value,
                    subtype=failure.subtype,
                    severity=failure.severity.value,
                    evidence_quote=failure.evidence_quote,
                    reasoning=failure.reasoning,
                    root_cause=rc.root_cause.value if rc else None,
                    root_cause_confidence=rc.confidence if rc else None,
                    root_cause_signals_json=rc.signals if rc else None,
                    recommended_fix=rc.recommended_fix if rc else None,
                )
                self.db.add(fail_rec)

            self.db.flush()
            logger.info("Persisted evaluation for %s", conversation.conversation_id)
        except Exception:
            logger.exception("Failed to persist evaluation results")
            raise

    @staticmethod
    def _clamp(value, low: int = 1, high: int = 5) -> int:
        try:
            return max(low, min(high, int(value)))
        except (ValueError, TypeError):
            return 3
