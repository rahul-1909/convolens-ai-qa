"""Transcript normalizer — converts raw transcripts (from multiple ASR
providers) into the unified ConvoLens schema.
"""

from __future__ import annotations

import uuid
from datetime import datetime

from src.schemas.conversation import Conversation, Turn, SpeakerRole, ToolCall


def normalize_raw_transcript(raw: dict) -> Conversation:
    """Convert a raw transcript dict into a ``Conversation`` object.

    Supports two input shapes:
      1. **Simple format**: ``{"conversation_id": ..., "turns": [...]}``
      2. **Deepgram / Whisper format**: With ``utterances`` and ``words``
         arrays including speaker labels and timestamps.

    Args:
        raw: Raw transcript dictionary.

    Returns:
        Normalised ``Conversation`` object.
    """
    conv_id = raw.get("conversation_id", str(uuid.uuid4()))

    # Detect format
    if "turns" in raw:
        turns = _parse_simple_turns(raw["turns"])
    elif "utterances" in raw:
        turns = _parse_utterance_format(raw["utterances"])
    elif "results" in raw and "utterances" in raw.get("results", {}):
        turns = _parse_utterance_format(raw["results"]["utterances"])
    else:
        turns = []

    return Conversation(
        conversation_id=conv_id,
        customer_id=raw.get("customer_id"),
        agent_version=raw.get("agent_version"),
        prompt_version=raw.get("prompt_version"),
        conversation_goal=raw.get("conversation_goal"),
        channel=raw.get("channel"),
        language=raw.get("language"),
        started_at=_parse_dt(raw.get("started_at")),
        ended_at=_parse_dt(raw.get("ended_at")),
        turns=turns,
        metadata=raw.get("metadata"),
    )


def _parse_simple_turns(raw_turns: list[dict]) -> list[Turn]:
    """Parse turns from the simple JSON format."""
    turns = []
    for i, t in enumerate(raw_turns):
        speaker_str = t.get("speaker", "agent").lower()
        speaker = (
            SpeakerRole.AGENT
            if speaker_str == "agent"
            else SpeakerRole.CUSTOMER
            if speaker_str in ("customer", "user", "human")
            else SpeakerRole.SYSTEM
        )

        tool_calls = []
        for tc in t.get("tool_calls", []):
            tool_calls.append(
                ToolCall(
                    tool_name=tc.get("tool_name", "unknown"),
                    input_payload=tc.get("input_payload"),
                    output_payload=tc.get("output_payload"),
                    success=tc.get("success", True),
                    latency_ms=tc.get("latency_ms"),
                )
            )

        turns.append(
            Turn(
                turn_id=t.get("turn_id", f"{i}"),
                turn_index=t.get("turn_index", i),
                speaker=speaker,
                transcript=t.get("transcript", ""),
                timestamp=_parse_dt(t.get("timestamp")),
                duration_ms=t.get("duration_ms"),
                asr_confidence=t.get("asr_confidence"),
                tool_calls=tool_calls,
                kb_snippets=t.get("kb_snippets", []),
            )
        )
    return turns


def _parse_utterance_format(utterances: list[dict]) -> list[Turn]:
    """Parse Deepgram/Whisper-style utterance arrays."""
    turns = []
    for i, u in enumerate(utterances):
        speaker_label = str(u.get("speaker", u.get("channel", i)))
        # Even speakers are agents, odd are customers (convention)
        try:
            speaker_idx = int(speaker_label)
            speaker = SpeakerRole.AGENT if speaker_idx % 2 == 0 else SpeakerRole.CUSTOMER
        except ValueError:
            speaker = (
                SpeakerRole.AGENT if "agent" in speaker_label.lower()
                else SpeakerRole.CUSTOMER
            )

        start = u.get("start")
        end = u.get("end")
        confidence = u.get("confidence", u.get("avg_confidence"))

        duration = None
        if start is not None and end is not None and end >= start:
            duration = float(end - start) * 1000.0

        turns.append(
            Turn(
                turn_id=str(i),
                turn_index=i,
                speaker=speaker,
                transcript=u.get("transcript", u.get("text", "")),
                duration_ms=duration,
                asr_confidence=confidence,
            )
        )
    return turns


def _parse_dt(value) -> datetime | None:
    """Best-effort datetime parser."""
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None
