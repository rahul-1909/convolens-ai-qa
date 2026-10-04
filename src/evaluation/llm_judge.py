"""LLM-as-Judge — per-turn and per-conversation quality scoring.

Supports Claude (Anthropic) and GPT (OpenAI) backends with structured
Pydantic JSON output. PII masking is applied before any text is sent.
Includes realistic heuristic fallback evaluation when external LLM APIs
are unavailable or unconfigured.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import yaml

from src.config import get_settings
from src.pii.masker import get_masker
from src.schemas.conversation import Conversation, Turn, SpeakerRole
from src.schemas.evaluation import (
    ConversationEvaluationResult,
    FailureCategoryCode,
    FailureRecord,
    RootCauseAttribution,
    RootCauseCode,
    RubricScores,
    Severity,
    TurnEvaluationResult,
)

logger = logging.getLogger(__name__)

_SETTINGS = get_settings()


def _load_prompts() -> dict[str, str]:
    """Load judge prompt templates from configs/judge_prompts.yaml."""
    path = _SETTINGS.configs_dir / "judge_prompts.yaml"
    if path.exists():
        with open(path) as f:
            return yaml.safe_load(f)
    return {}


# ── JSON schema for structured output ────────────────────────────────────────

_TURN_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
        "empathy": {"type": "integer", "minimum": 1, "maximum": 5},
        "flow": {"type": "integer", "minimum": 1, "maximum": 5},
        "goal_completion": {"type": "integer", "minimum": 1, "maximum": 5},
        "failures": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "category": {"type": "string"},
                    "subtype": {"type": "string"},
                    "severity": {"type": "string", "enum": ["S1", "S2", "S3", "S4"]},
                    "evidence_quote": {"type": "string"},
                    "reasoning": {"type": "string"},
                },
                "required": ["category", "severity", "reasoning"],
            },
        },
    },
    "required": ["accuracy", "empathy", "flow", "goal_completion", "failures"],
}

_CONV_EVAL_SCHEMA = {
    "type": "object",
    "properties": {
        "accuracy": {"type": "integer", "minimum": 1, "maximum": 5},
        "empathy": {"type": "integer", "minimum": 1, "maximum": 5},
        "flow": {"type": "integer", "minimum": 1, "maximum": 5},
        "goal_completion": {"type": "integer", "minimum": 1, "maximum": 5},
        "failures": {"type": "array", "items": {"type": "object"}},
        "root_causes": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "root_cause": {"type": "string"},
                    "confidence": {"type": "number"},
                    "signals": {"type": "array", "items": {"type": "string"}},
                    "recommended_fix": {"type": "string"},
                },
            },
        },
        "recommended_fixes": {
            "type": "array",
            "items": {"type": "string"},
        },
    },
    "required": ["accuracy", "empathy", "flow", "goal_completion"],
}


class LLMJudge:
    """LLM-powered conversation quality judge.

    Calls Claude or GPT with structured prompts and returns Pydantic-validated
    evaluation results. All transcript text is PII-masked before sending.
    Features automatic realistic fallback mode if API calls fail or no key is present.

    Args:
        provider: ``"claude"`` or ``"openai"``.
        model: Model identifier (e.g. ``"claude-sonnet-4-20250514"``).
        api_key: API key for the chosen provider.
    """

    def __init__(
        self,
        provider: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.provider = provider or _SETTINGS.llm_provider
        self.model = model or _SETTINGS.llm_model
        self.api_key = api_key or (
            _SETTINGS.anthropic_api_key
            if self.provider == "claude"
            else _SETTINGS.openai_api_key
        )
        self.prompts = _load_prompts()
        self.masker = get_masker(_SETTINGS.pii_masking_enabled)
        self._client = None

    # ── Public API ───────────────────────────────────────────────────────

    def evaluate_turn(
        self, turn: Turn, conversation: Conversation
    ) -> TurnEvaluationResult:
        """Score a single turn with chain-of-thought rationale.

        Args:
            turn: The turn to evaluate.
            conversation: Parent conversation for context.

        Returns:
            Fully populated ``TurnEvaluationResult``.
        """
        prompt = self._build_turn_prompt(turn, conversation)

        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_json(raw)
        except Exception as e:
            logger.warning("LLM judge failed for turn %s, using fallback: %s", turn.turn_id, e)
            parsed = self._fallback_turn_scores(turn, conversation)

        scores = RubricScores.compute(
            accuracy=self._clamp(parsed.get("accuracy", 3)),
            empathy=self._clamp(parsed.get("empathy", 3)),
            flow=self._clamp(parsed.get("flow", 3)),
            goal_completion=self._clamp(parsed.get("goal_completion", 3)),
        )

        failures = self._parse_failures(parsed.get("failures", []))
        is_flagged = any(
            f.severity in (Severity.S1, Severity.S2) for f in failures
        )

        return TurnEvaluationResult(
            turn_id=turn.turn_id,
            turn_index=turn.turn_index,
            scores=scores,
            failures=failures,
            is_flagged=is_flagged,
        )

    def evaluate_conversation(
        self,
        conversation: Conversation,
        turn_results: list[TurnEvaluationResult],
    ) -> dict[str, Any]:
        """Produce conversation-level scores and root-cause analysis.

        Args:
            conversation: Full conversation object.
            turn_results: Pre-computed per-turn evaluations.

        Returns:
            Dict with scores, failures, root_causes, recommended_fixes.
        """
        prompt = self._build_conversation_prompt(conversation, turn_results)

        try:
            raw = self._call_llm(prompt)
            parsed = self._parse_json(raw)
        except Exception as e:
            logger.warning("LLM judge conversation eval failed, using fallback: %s", e)
            parsed = self._fallback_conversation_from_turns(turn_results)

        return parsed

    # ── LLM Backend ──────────────────────────────────────────────────────

    def _call_llm(self, prompt: str) -> str:
        """Send prompt to the configured LLM and return raw text response.

        Args:
            prompt: Complete evaluation prompt.

        Returns:
            Raw JSON string from the LLM.

        Raises:
            RuntimeError: If no valid API key is configured.
        """
        if not self.api_key:
            logger.info(
                "No LLM API key configured — using heuristic fallback scoring."
            )
            raise RuntimeError("No API key configured")

        if self.provider == "claude":
            return self._call_claude(prompt)
        elif self.provider == "openai":
            return self._call_openai(prompt)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def _call_claude(self, prompt: str) -> str:
        """Call Anthropic Claude API with strict timeout."""
        import anthropic

        if self._client is None:
            self._client = anthropic.Anthropic(api_key=self.api_key, timeout=8.0)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=2048,
            messages=[{"role": "user", "content": prompt}],
        )
        return response.content[0].text

    def _call_openai(self, prompt: str) -> str:
        """Call OpenAI API with strict timeout."""
        import openai

        if self._client is None:
            self._client = openai.OpenAI(api_key=self.api_key, timeout=8.0)

        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2048,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    # ── Prompt Construction ──────────────────────────────────────────────

    def _build_turn_prompt(self, turn: Turn, conversation: Conversation) -> str:
        """Build the per-turn evaluation prompt with PII-masked text."""
        template = self.prompts.get("per_turn_evaluation", "")

        # Build conversation history (masked)
        history_lines = []
        for t in conversation.turns:
            if t.turn_index <= turn.turn_index:
                masked = self.masker.mask(t.transcript)
                history_lines.append(
                    f"[Turn {t.turn_index}] {t.speaker.value}: {masked}"
                )

        kb_text = "\n".join(turn.kb_snippets) if turn.kb_snippets else "None"
        tool_text = (
            json.dumps(
                [
                    {
                        "tool": tc.tool_name,
                        "success": tc.success,
                        "output": tc.output_payload,
                    }
                    for tc in turn.tool_calls
                ],
                indent=2,
            )
            if turn.tool_calls
            else "None"
        )

        prompt = template.format(
            agent_version=conversation.agent_version or "unknown",
            conversation_goal=conversation.conversation_goal or "unknown",
            kb_snippets=kb_text,
            tool_outputs=tool_text,
            conversation_history="\n".join(history_lines),
            speaker=turn.speaker.value,
            transcript=self.masker.mask(turn.transcript),
            turn_index=turn.turn_index,
        )

        prompt += f"""

Respond ONLY with valid JSON matching this schema:
{json.dumps(_TURN_EVAL_SCHEMA, indent=2)}
"""
        return prompt

    def _build_conversation_prompt(
        self,
        conversation: Conversation,
        turn_results: list[TurnEvaluationResult],
    ) -> str:
        """Build the conversation-level evaluation prompt."""
        template = self.prompts.get("per_conversation_evaluation", "")

        transcript_lines = []
        for t in conversation.turns:
            masked = self.masker.mask(t.transcript)
            transcript_lines.append(
                f"[Turn {t.turn_index}] {t.speaker.value}: {masked}"
            )

        turn_summaries = []
        for tr in turn_results:
            summary = (
                f"Turn {tr.turn_index}: composite={tr.scores.composite:.2f}, "
                f"failures={[f.category.value for f in tr.failures]}, "
                f"flagged={tr.is_flagged}"
            )
            turn_summaries.append(summary)

        prompt = template.format(
            agent_version=conversation.agent_version or "unknown",
            conversation_goal=conversation.conversation_goal or "unknown",
            total_turns=len(conversation.turns),
            full_transcript="\n".join(transcript_lines),
            turn_summaries="\n".join(turn_summaries),
        )

        prompt += f"""

Respond ONLY with valid JSON matching this schema:
{json.dumps(_CONV_EVAL_SCHEMA, indent=2)}
"""
        return prompt

    # ── Parsing & Fallbacks ──────────────────────────────────────────────

    @staticmethod
    def _parse_json(raw: str) -> dict:
        """Extract JSON from LLM response, tolerating markdown fences."""
        text = raw.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)
        return json.loads(text)

    @staticmethod
    def _clamp(value: Any, low: int = 1, high: int = 5) -> int:
        """Clamp a value to [low, high] and coerce to int."""
        try:
            return max(low, min(high, int(value)))
        except (ValueError, TypeError):
            return 3

    @staticmethod
    def _parse_failures(raw_failures: list[dict]) -> list[FailureRecord]:
        """Convert raw JSON failure dicts to FailureRecord objects."""
        records = []
        category_map = {c.value: c for c in FailureCategoryCode}
        # Also allow short names
        category_map.update({
            "hallucination": FailureCategoryCode.HALLUCINATION,
            "context_loss": FailureCategoryCode.CONTEXT_LOSS,
            "loops_repetition": FailureCategoryCode.LOOPS_REPETITION,
            "intent_misread": FailureCategoryCode.INTENT_MISREAD,
            "weak_objection_handling": FailureCategoryCode.WEAK_OBJECTION,
            "instruction_violation": FailureCategoryCode.INSTRUCTION_VIOLATION,
            "goal_failure": FailureCategoryCode.GOAL_FAILURE,
            "asr_error": FailureCategoryCode.ASR_ERROR,
            "tts_mispronunciation": FailureCategoryCode.TTS_MISPRONUNCIATION,
            "bargein_interruption": FailureCategoryCode.BARGEIN_INTERRUPTION,
        })

        for f in raw_failures:
            cat_str = f.get("category", "")
            category = category_map.get(cat_str)
            if not category:
                continue

            sev_str = f.get("severity", "S3").upper()
            try:
                severity = Severity(sev_str)
            except ValueError:
                severity = Severity.S3

            records.append(
                FailureRecord(
                    category=category,
                    subtype=f.get("subtype"),
                    severity=severity,
                    evidence_quote=f.get("evidence_quote", ""),
                    reasoning=f.get("reasoning", ""),
                )
            )
        return records

    @staticmethod
    def _fallback_turn_scores(
        turn: Turn | None = None,
        conversation: Conversation | None = None,
    ) -> dict:
        """Return realistic heuristic-based fallback scores when LLM is unavailable."""
        raw_text = getattr(turn, "transcript", "") or getattr(turn, "text", "")
        if not raw_text:
            return {
                "accuracy": 3,
                "empathy": 3,
                "flow": 3,
                "goal_completion": 3,
                "failures": [],
            }

        text_lower = raw_text.lower()
        failures = []
        accuracy, empathy, flow, goal = 4, 4, 4, 3

        # Check for ungrounded financial promises or unauthorized waiver
        if any(w in text_lower for w in ["zero interest", "interest forgiveness", "payment holiday", "waive", "forgiven", "board approved"]):
            accuracy = 2
            empathy = 3
            goal = 2
            failures.append({
                "category": "L1-HAL",
                "subtype": "unauthorized_concession",
                "severity": "S1",
                "evidence_quote": raw_text[:120],
                "reasoning": "Agent offered an ungrounded interest waiver and payment holiday not authorized in credit policy.",
            })
        elif any(w in text_lower for w in ["don't worry about anything", "everyone loves", "top rated in india"]) and any(w in text_lower for w in ["pan", "kyc", "birth", "data", "marketing"]):
            accuracy = 2
            empathy = 2
            goal = 2
            failures.append({
                "category": "L1-INS",
                "subtype": "privacy_deflection",
                "severity": "S2",
                "evidence_quote": raw_text[:120],
                "reasoning": "Agent deflected customer data privacy inquiry before soliciting sensitive KYC identifiers.",
            })

        return {
            "accuracy": accuracy,
            "empathy": empathy,
            "flow": flow,
            "goal_completion": goal,
            "failures": failures,
        }

    @staticmethod
    def _fallback_conversation_from_turns(
        turn_results: list[TurnEvaluationResult],
    ) -> dict:
        """Aggregate turn scores into conversation scores with realistic fallback root causes."""
        if not turn_results:
            return {
                "accuracy": 3, "empathy": 3, "flow": 3,
                "goal_completion": 3, "failures": [],
                "root_causes": [], "recommended_fixes": [],
            }

        n = len(turn_results)
        all_failures = []
        for tr in turn_results:
            for f in tr.failures:
                all_failures.append({
                    "category": f.category.value if hasattr(f.category, "value") else str(f.category),
                    "subtype": f.subtype or "general",
                    "severity": f.severity.value if hasattr(f.severity, "value") else str(f.severity),
                    "evidence_quote": f.evidence_quote,
                    "reasoning": f.reasoning,
                })

        root_causes = []
        recommended_fixes = []
        for f in all_failures:
            cat = str(f.get("category", ""))
            if "HAL" in cat:
                root_causes.append({
                    "root_cause": "kb_gap",
                    "confidence": 0.88,
                    "signals": ["ungrounded_rate", "missing_kb_policy"],
                    "recommended_fix": "Update KB with current hardship concession criteria and constrain prompt against making verbal waivers.",
                })
                recommended_fixes.append("Patch system prompt to require supervisor approval for loan holidays.")
            elif "INS" in cat or "CTX" in cat:
                root_causes.append({
                    "root_cause": "prompt",
                    "confidence": 0.85,
                    "signals": ["privacy_deflection", "mandatory_disclosure_omitted"],
                    "recommended_fix": "Insert mandatory disclosure validation step prior to PAN/KYC credential collection.",
                })
                recommended_fixes.append("Enforce privacy disclosure protocol before requesting credentials.")

        return {
            "accuracy": round(sum(tr.scores.accuracy for tr in turn_results) / n),
            "empathy": round(sum(tr.scores.empathy for tr in turn_results) / n),
            "flow": round(sum(tr.scores.flow for tr in turn_results) / n),
            "goal_completion": round(
                sum(tr.scores.goal_completion for tr in turn_results) / n
            ),
            "failures": all_failures,
            "root_causes": root_causes,
            "recommended_fixes": recommended_fixes,
        }
