"""Pydantic schemas for conversations, turns, and speaker data."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class SpeakerRole(str, Enum):
    """Who is speaking in a given turn."""

    AGENT = "agent"
    CUSTOMER = "customer"
    SYSTEM = "system"


class ToolCall(BaseModel):
    """Record of an external tool invocation made during a turn."""

    tool_name: str = Field(..., description="Name of the tool / API called")
    input_payload: dict | None = Field(
        None, description="Arguments sent to the tool"
    )
    output_payload: dict | str | None = Field(
        None, description="Response returned by the tool"
    )
    success: bool = Field(True, description="Whether the call succeeded")
    latency_ms: float | None = Field(
        None, description="Round-trip time in milliseconds"
    )


class Turn(BaseModel):
    """A single conversational turn (one speaker utterance)."""

    turn_id: str = Field(..., description="Unique turn identifier")
    turn_index: int = Field(..., description="0-based position in conversation")
    speaker: SpeakerRole
    transcript: str = Field(..., description="Text content of this turn")
    timestamp: datetime | None = Field(
        None, description="Absolute timestamp of the turn"
    )
    duration_ms: float | None = Field(
        None, description="Duration of audio for this turn"
    )
    asr_confidence: float | None = Field(
        None,
        ge=0.0,
        le=1.0,
        description="ASR engine confidence (0-1) if from speech",
    )
    tool_calls: list[ToolCall] = Field(
        default_factory=list, description="Tools invoked during this turn"
    )
    kb_snippets: list[str] = Field(
        default_factory=list,
        description="Knowledge-base passages retrieved for this turn",
    )


class Conversation(BaseModel):
    """Full conversation with metadata and ordered turns."""

    conversation_id: str = Field(..., description="Unique conversation ID")
    customer_id: str | None = Field(
        None, description="Customer / account identifier"
    )
    agent_version: str | None = Field(
        None, description="Version tag of the AI agent"
    )
    prompt_version: str | None = Field(
        None, description="Version of the system prompt used"
    )
    conversation_goal: str | None = Field(
        None,
        description="Intended goal (e.g. 'loan_collection', 'kyc_onboarding')",
    )
    channel: str | None = Field(
        None, description="'voice' or 'chat'"
    )
    language: str | None = Field(None, description="ISO 639-1 language code")
    started_at: datetime | None = None
    ended_at: datetime | None = None
    turns: list[Turn] = Field(
        default_factory=list, description="Ordered list of conversation turns"
    )
    metadata: dict | None = Field(
        None, description="Arbitrary additional metadata"
    )
