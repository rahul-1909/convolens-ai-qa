"""PII masking — redacts names, phone numbers, emails, PAN, Aadhaar before LLM calls.

All patterns are applied BEFORE any text is sent to an external LLM. The original
text is stored in the database; masked text is only used for evaluation prompts.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class PIIMasker:
    """Regex-based PII detector and redactor.

    Attributes:
        enabled: Global toggle from settings.
        patterns: Ordered list of (label, compiled regex) tuples.
    """

    enabled: bool = True
    patterns: list[tuple[str, re.Pattern]] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.patterns:
            self.patterns = self._default_patterns()

    # ── Public API ───────────────────────────────────────────────────────

    def mask(self, text: str) -> str:
        """Replace all detected PII spans with bracketed placeholders.

        Args:
            text: Raw transcript or turn content.

        Returns:
            Text with PII replaced, e.g. ``[PHONE]``, ``[EMAIL]``, ``[PAN]``.
        """
        if not self.enabled or not text:
            return text

        masked = text
        for label, pattern in self.patterns:
            masked = pattern.sub(f"[{label}]", masked)
        return masked

    def mask_turns(self, turns: list[dict]) -> list[dict]:
        """Mask the ``transcript`` field in a list of turn dicts (in-place copy).

        Args:
            turns: List of turn dictionaries, each with a ``transcript`` key.

        Returns:
            New list with masked transcripts (originals are not mutated).
        """
        masked_turns = []
        for turn in turns:
            t = dict(turn)
            t["transcript"] = self.mask(t.get("transcript", ""))
            masked_turns.append(t)
        return masked_turns

    # ── Defaults ─────────────────────────────────────────────────────────

    @staticmethod
    def _default_patterns() -> list[tuple[str, re.Pattern]]:
        """India-focused PII patterns + universal email / credit-card."""
        return [
            # Aadhaar (12 digits, often space/dash separated in groups of 4)
            ("AADHAAR", re.compile(r"\b\d{4}[\s\-]?\d{4}[\s\-]?\d{4}\b")),
            # PAN (Indian Permanent Account Number: ABCDE1234F)
            ("PAN", re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b")),
            # Indian phone numbers (+91, 0-prefix, or raw 10-digit)
            (
                "PHONE",
                re.compile(
                    r"(?:\+91[\s\-]?|0)?[6-9]\d{4}[\s\-]?\d{5}\b"
                ),
            ),
            # Email addresses
            (
                "EMAIL",
                re.compile(
                    r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Z|a-z]{2,}\b"
                ),
            ),
            # Credit/debit card numbers (13-19 digits, grouped)
            (
                "CARD",
                re.compile(
                    r"\b(?:\d{4}[\s\-]?){3,4}\d{1,4}\b"
                ),
            ),
            # Date of birth patterns (DD/MM/YYYY, DD-MM-YYYY)
            (
                "DOB",
                re.compile(
                    r"\b\d{1,2}[\-/]\d{1,2}[\-/]\d{2,4}\b"
                ),
            ),
        ]


# Module-level convenience instance
_default_masker: PIIMasker | None = None


def get_masker(enabled: bool = True) -> PIIMasker:
    """Return a module-level singleton masker.

    Args:
        enabled: Whether masking is active.

    Returns:
        Configured PIIMasker instance.
    """
    global _default_masker
    if _default_masker is None:
        _default_masker = PIIMasker(enabled=enabled)
    return _default_masker
