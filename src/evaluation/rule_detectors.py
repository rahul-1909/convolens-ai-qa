"""Rule-based detectors for conversation quality issues.

These run BEFORE the LLM judge and catch deterministic patterns:
- Repeated agent utterances (n-gram / embedding similarity)
- Long silences between turns
- Low ASR confidence spans
- Overlapping speech / barge-in indicators
- Skipped mandatory compliance disclosures
- Weak objection deflection
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from difflib import SequenceMatcher

from src.schemas.conversation import Turn, SpeakerRole
from src.schemas.evaluation import FailureRecord, FailureCategoryCode, Severity


# ── Helpers ──────────────────────────────────────────────────────────────────


def _normalize(text: str) -> str:
    """Lowercase, strip punctuation, collapse whitespace."""
    text = text.lower().strip()
    text = re.sub(r"[^\w\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _similarity(a: str, b: str) -> float:
    """Sequence-matcher ratio between two normalised strings."""
    return SequenceMatcher(None, _normalize(a), _normalize(b)).ratio()


# ── Detectors ────────────────────────────────────────────────────────────────


@dataclass
class RuleDetectorConfig:
    """Tunable thresholds for every rule-based detector.

    Attributes:
        repetition_similarity_threshold: Min ratio to flag two agent
            utterances as near-duplicates (0-1).
        repetition_window: Number of preceding agent turns to compare.
        silence_threshold_ms: Gap between turns above which a long silence
            is flagged.
        low_asr_confidence_threshold: ASR confidence below this triggers
            an ``L2-ASR`` flag.
        min_turn_length_for_repetition: Ignore very short turns (e.g.
            "yes", "okay") when checking repetition.
    """

    repetition_similarity_threshold: float = 0.80
    repetition_window: int = 5
    silence_threshold_ms: float = 8_000.0
    low_asr_confidence_threshold: float = 0.60
    min_turn_length_for_repetition: int = 20


class RuleBasedDetectors:
    """Stateless collection of deterministic quality detectors.

    Args:
        config: Detector thresholds. Uses defaults if not supplied.
    """

    def __init__(self, config: RuleDetectorConfig | None = None) -> None:
        self.cfg = config or RuleDetectorConfig()

    # ── Public API ───────────────────────────────────────────────────────

    def run_all(self, turns: list[Turn]) -> list[FailureRecord]:
        """Execute every detector and return combined failure list.

        Args:
            turns: Ordered conversation turns.

        Returns:
            Deduplicated list of ``FailureRecord`` objects.
        """
        failures: list[FailureRecord] = []
        failures.extend(self.detect_repeated_utterances(turns))
        failures.extend(self.detect_long_silences(turns))
        failures.extend(self.detect_low_asr_confidence(turns))
        failures.extend(self.detect_mandatory_disclosure_breach(turns))
        failures.extend(self.detect_weak_objection_deflection(turns))
        return failures

    # ── Individual Detectors ─────────────────────────────────────────────

    def detect_repeated_utterances(
        self, turns: list[Turn]
    ) -> list[FailureRecord]:
        """Flag near-duplicate agent utterances within a sliding window.

        Args:
            turns: Ordered conversation turns.

        Returns:
            List of ``L1-LOOP`` failures with evidence quotes.
        """
        failures: list[FailureRecord] = []
        agent_turns = [
            t for t in turns if t.speaker == SpeakerRole.AGENT
        ]

        for i, current in enumerate(agent_turns):
            if len(current.transcript) < self.cfg.min_turn_length_for_repetition:
                continue
            window_start = max(0, i - self.cfg.repetition_window)
            for prev in agent_turns[window_start:i]:
                if len(prev.transcript) < self.cfg.min_turn_length_for_repetition:
                    continue
                sim = _similarity(current.transcript, prev.transcript)
                if sim >= self.cfg.repetition_similarity_threshold:
                    failures.append(
                        FailureRecord(
                            category=FailureCategoryCode.LOOPS_REPETITION,
                            subtype="phrase_repeat",
                            severity=Severity.S2,
                            evidence_quote=(
                                f"Turn {current.turn_index} repeats turn "
                                f"{prev.turn_index} (similarity={sim:.0%}): "
                                f'"{current.transcript[:120]}..."'
                            ),
                            reasoning=(
                                f"Agent utterance at turn {current.turn_index} "
                                f"has {sim:.0%} textual similarity with turn "
                                f"{prev.turn_index}, exceeding the "
                                f"{self.cfg.repetition_similarity_threshold:.0%} "
                                f"threshold."
                            ),
                        )
                    )
                    break  # one match per current turn is enough
        return failures

    def detect_long_silences(self, turns: list[Turn]) -> list[FailureRecord]:
        """Flag gaps between consecutive turns that exceed the threshold.

        Args:
            turns: Ordered conversation turns with timestamps.

        Returns:
            List of ``L2-BARG`` failures for silence timeouts.
        """
        failures: list[FailureRecord] = []
        for i in range(1, len(turns)):
            prev, curr = turns[i - 1], turns[i]
            if prev.timestamp and curr.timestamp:
                gap_ms = (
                    curr.timestamp - prev.timestamp
                ).total_seconds() * 1000
                if prev.duration_ms:
                    gap_ms -= prev.duration_ms
                if gap_ms > self.cfg.silence_threshold_ms:
                    failures.append(
                        FailureRecord(
                            category=FailureCategoryCode.BARGEIN_INTERRUPTION,
                            subtype="silence_timeout",
                            severity=Severity.S3,
                            evidence_quote=(
                                f"Gap of {gap_ms/1000:.1f}s between turn "
                                f"{prev.turn_index} and turn {curr.turn_index}."
                            ),
                            reasoning=(
                                f"Silence of {gap_ms:.0f}ms exceeds the "
                                f"{self.cfg.silence_threshold_ms:.0f}ms threshold."
                            ),
                        )
                    )
        return failures

    def detect_low_asr_confidence(
        self, turns: list[Turn]
    ) -> list[FailureRecord]:
        """Flag turns where ASR confidence is below the threshold.

        Args:
            turns: Ordered conversation turns.

        Returns:
            List of ``L2-ASR`` failures.
        """
        failures: list[FailureRecord] = []
        for t in turns:
            if (
                t.asr_confidence is not None
                and t.asr_confidence < self.cfg.low_asr_confidence_threshold
            ):
                failures.append(
                    FailureRecord(
                        category=FailureCategoryCode.ASR_ERROR,
                        subtype="low_confidence",
                        severity=Severity.S3,
                        evidence_quote=(
                            f'Turn {t.turn_index} ({t.speaker.value}): '
                            f'"{t.transcript[:100]}" '
                            f'[confidence={t.asr_confidence:.2f}]'
                        ),
                        reasoning=(
                            f"ASR confidence {t.asr_confidence:.2f} is below "
                            f"the {self.cfg.low_asr_confidence_threshold} "
                            f"threshold — transcript may be unreliable."
                        ),
                    )
                )
        return failures

    def detect_mandatory_disclosure_breach(
        self, turns: list[Turn]
    ) -> list[FailureRecord]:
        """Flag agent turns where mandatory disclosures in KB snippets were skipped.

        Args:
            turns: Ordered conversation turns.

        Returns:
            List of ``L1-INS`` failures.
        """
        failures: list[FailureRecord] = []
        for t in turns:
            if t.speaker == SpeakerRole.AGENT and t.kb_snippets:
                for kb in t.kb_snippets:
                    if "MUST state" in kb or "Mandatory" in kb:
                        lower_tx = t.transcript.lower()
                        terms = []
                        if "fee" in kb.lower() and "fee" not in lower_tx:
                            terms.append("fee")
                        if "apr" in kb.lower() and "apr" not in lower_tx:
                            terms.append("APR / interest rate")
                        if (
                            "accrue" in kb.lower()
                            and "accrue" not in lower_tx
                            and "interest" not in lower_tx
                        ):
                            terms.append("accruing interest")
                        if terms:
                            failures.append(
                                FailureRecord(
                                    category=FailureCategoryCode.INSTRUCTION_VIOLATION,
                                    subtype="skipped_disclosure",
                                    severity=Severity.S1,
                                    evidence_quote=(
                                        f'Turn {t.turn_index}: "{t.transcript[:100]}..." '
                                        f'(Omitted: {", ".join(terms)})'
                                    ),
                                    reasoning=(
                                        f"Agent KB mandated: '{kb[:80]}...', "
                                        f"but agent omitted mandatory terms: {', '.join(terms)}."
                                    ),
                                )
                            )
        return failures

    def detect_weak_objection_deflection(
        self, turns: list[Turn]
    ) -> list[FailureRecord]:
        """Flag agent turns that deflect customer concerns with generic platitudes.

        Args:
            turns: Ordered conversation turns.

        Returns:
            List of ``L1-OBJ`` failures.
        """
        failures: list[FailureRecord] = []
        for i in range(1, len(turns)):
            prev, curr = turns[i - 1], turns[i]
            if (
                prev.speaker == SpeakerRole.CUSTOMER
                and curr.speaker == SpeakerRole.AGENT
            ):
                customer_text = prev.transcript.lower()
                agent_text = curr.transcript.lower()
                has_objection = any(
                    w in customer_text
                    for w in [
                        "fee",
                        "charge",
                        "privacy",
                        "data",
                        "share",
                        "why",
                        "expensive",
                        "cost",
                    ]
                )
                has_platitude = any(
                    p in agent_text
                    for p in [
                        "don't worry",
                        "dont worry",
                        "completely safe and secure",
                        "everyone loves",
                        "trust us",
                    ]
                )
                if has_objection and has_platitude:
                    failures.append(
                        FailureRecord(
                            category=FailureCategoryCode.WEAK_OBJECTION,
                            subtype="generic_response",
                            severity=Severity.S3,
                            evidence_quote=(
                                f'Customer turn {prev.turn_index}: "{prev.transcript[:70]}" -> '
                                f'Agent deflection turn {curr.turn_index}: "{curr.transcript[:70]}"'
                            ),
                            reasoning=(
                                "Agent delivered a cookie-cutter deflection "
                                "instead of addressing customer's specific question."
                            ),
                        )
                    )
        return failures
