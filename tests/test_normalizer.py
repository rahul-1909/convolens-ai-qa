"""Unit tests for transcript normalizer."""

from __future__ import annotations

from datetime import datetime
import pytest

from src.ingestion.normalizer import normalize_raw_transcript, _parse_dt
from src.schemas.conversation import SpeakerRole


def test_normalize_simple_format():
    """Test normalizing simple turns format."""
    raw = {
        "conversation_id": "conv-simple",
        "customer_id": "cust-1",
        "agent_version": "v1.0",
        "prompt_version": "p1.0",
        "conversation_goal": "test_goal",
        "channel": "chat",
        "language": "en",
        "started_at": "2026-10-01T10:00:00",
        "ended_at": "2026-10-01T10:05:00",
        "turns": [
            {
                "turn_id": "t-1",
                "turn_index": 0,
                "speaker": "agent",
                "transcript": "Hello customer",
                "timestamp": "2026-10-01T10:00:00",
                "duration_ms": 2000,
                "asr_confidence": 0.99,
                "tool_calls": [
                    {
                        "tool_name": "lookup",
                        "input_payload": {"id": 1},
                        "output_payload": {"name": "Alice"},
                        "success": True,
                        "latency_ms": 50,
                    }
                ],
                "kb_snippets": ["greeting policy"],
            },
            {
                "turn_id": "t-2",
                "turn_index": 1,
                "speaker": "user",
                "transcript": "Hi there",
            },
            {
                "turn_id": "t-3",
                "turn_index": 2,
                "speaker": "system",
                "transcript": "Call connected",
            },
        ],
    }
    conv = normalize_raw_transcript(raw)
    assert conv.conversation_id == "conv-simple"
    assert len(conv.turns) == 3
    assert conv.turns[0].speaker == SpeakerRole.AGENT
    assert conv.turns[1].speaker == SpeakerRole.CUSTOMER
    assert conv.turns[2].speaker == SpeakerRole.SYSTEM
    assert len(conv.turns[0].tool_calls) == 1
    assert conv.turns[0].tool_calls[0].tool_name == "lookup"


def test_normalize_utterances_format():
    """Test normalizing Deepgram-style utterances list."""
    raw = {
        "conversation_id": "conv-utterances",
        "utterances": [
            {
                "speaker": 0,
                "transcript": "Hello this is the agent",
                "start": 0.0,
                "end": 2.5,
                "confidence": 0.95,
            },
            {
                "speaker": 1,
                "transcript": "I need help",
                "start": 2.6,
                "end": 4.0,
                "confidence": 0.88,
            },
        ],
    }
    conv = normalize_raw_transcript(raw)
    assert len(conv.turns) == 2
    assert conv.turns[0].speaker == SpeakerRole.AGENT
    assert conv.turns[1].speaker == SpeakerRole.CUSTOMER
    assert conv.turns[0].duration_ms == 2500.0
    assert conv.turns[0].asr_confidence == 0.95


def test_normalize_results_utterances_format():
    """Test normalizing Deepgram nested results.utterances structure."""
    raw = {
        "conversation_id": "conv-deepgram",
        "results": {
            "utterances": [
                {
                    "speaker": "agent_channel",
                    "text": "How can I help you?",
                    "start": 1.0,
                    "end": 3.0,
                    "avg_confidence": 0.97,
                },
                {
                    "speaker": "caller",
                    "text": "My payment is stuck",
                    "start": 3.2,
                    "end": 5.0,
                    "avg_confidence": 0.92,
                },
            ]
        },
    }
    conv = normalize_raw_transcript(raw)
    assert len(conv.turns) == 2
    assert conv.turns[0].speaker == SpeakerRole.AGENT
    assert conv.turns[1].speaker == SpeakerRole.CUSTOMER
    assert conv.turns[0].transcript == "How can I help you?"


def test_normalize_empty_format():
    """Test normalizing an empty dictionary fallback."""
    conv = normalize_raw_transcript({})
    assert conv.conversation_id is not None
    assert len(conv.turns) == 0


def test_parse_dt_helper():
    """Test date parsing helper with various formats."""
    assert _parse_dt(None) is None
    now = datetime.now()
    assert _parse_dt(now) == now
    parsed = _parse_dt("2026-10-01T12:00:00")
    assert parsed.year == 2026 and parsed.month == 10
    assert _parse_dt("invalid-date-string") is None
