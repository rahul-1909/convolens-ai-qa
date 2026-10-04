"""Shared pytest fixtures and test configuration."""

from __future__ import annotations

import json
import os
import sys
from datetime import datetime
from pathlib import Path

import pytest

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

# Force SQLite for tests
os.environ["DATABASE_URL"] = "sqlite:///./test_convolens.db"
os.environ["APP_DEBUG"] = "false"
os.environ["PII_MASKING_ENABLED"] = "true"

from src.db.session import init_db, engine, SessionLocal
from src.models.database import Base
from src.schemas.conversation import Conversation, Turn, SpeakerRole, ToolCall


@pytest.fixture(scope="session", autouse=True)
def setup_test_db():
    """Create all tables before test session, drop after."""
    Base.metadata.create_all(bind=engine)
    yield
    try:
        Base.metadata.drop_all(bind=engine)
    except Exception:
        pass
    finally:
        engine.dispose()
        db_file = Path("test_convolens.db")
        if db_file.exists():
            try:
                db_file.unlink()
            except OSError:
                pass


@pytest.fixture
def db_session():
    """Yield a fresh DB session per test, rolled back after."""
    session = SessionLocal()
    try:
        yield session
    finally:
        session.rollback()
        session.close()


@pytest.fixture
def sample_conversation() -> Conversation:
    """Build a minimal but realistic test conversation."""
    return Conversation(
        conversation_id="test-conv-001",
        customer_id="CUST-TEST",
        agent_version="v1.0.0",
        conversation_goal="loan_collection",
        channel="voice",
        language="en",
        started_at=datetime(2026, 10, 1, 10, 0, 0),
        turns=[
            Turn(
                turn_id="t1",
                turn_index=0,
                speaker=SpeakerRole.AGENT,
                transcript="Good morning! Am I speaking with Mr. Kumar?",
                timestamp=datetime(2026, 10, 1, 10, 0, 0),
                duration_ms=3000,
                asr_confidence=0.95,
                kb_snippets=["Greeting script: Confirm customer identity."],
            ),
            Turn(
                turn_id="t2",
                turn_index=1,
                speaker=SpeakerRole.CUSTOMER,
                transcript="Yes, this is Kumar.",
                timestamp=datetime(2026, 10, 1, 10, 0, 4),
                duration_ms=1500,
                asr_confidence=0.92,
            ),
            Turn(
                turn_id="t3",
                turn_index=2,
                speaker=SpeakerRole.AGENT,
                transcript=(
                    "I'm calling about your personal loan. Our records show "
                    "your EMI of Rs. 12,000 was due on September 20th."
                ),
                timestamp=datetime(2026, 10, 1, 10, 0, 6),
                duration_ms=5500,
                asr_confidence=0.97,
                tool_calls=[
                    ToolCall(
                        tool_name="fetch_account",
                        input_payload={"customer_id": "CUST-TEST"},
                        output_payload={"emi_amount": 12000, "due_date": "2026-09-20"},
                        success=True,
                        latency_ms=100,
                    )
                ],
                kb_snippets=["Collection script: State overdue amount and date."],
            ),
            Turn(
                turn_id="t4",
                turn_index=3,
                speaker=SpeakerRole.CUSTOMER,
                transcript="I had some financial issues but I can pay next week.",
                timestamp=datetime(2026, 10, 1, 10, 0, 12),
                duration_ms=3200,
                asr_confidence=0.55,  # Low confidence — should trigger ASR flag
            ),
            Turn(
                turn_id="t5",
                turn_index=4,
                speaker=SpeakerRole.AGENT,
                transcript=(
                    "I understand. As per our records, you have an excellent "
                    "payment history. Can we set a payment date for next Friday?"
                ),
                timestamp=datetime(2026, 10, 1, 10, 0, 16),
                duration_ms=6000,
                asr_confidence=0.96,
                # No tool call but says "as per our records" — hallucination flag
            ),
        ],
    )


@pytest.fixture
def sample_conversation_with_repetition() -> Conversation:
    """Conversation where the agent repeats itself."""
    return Conversation(
        conversation_id="test-conv-repeat",
        agent_version="v1.0.0",
        conversation_goal="kyc_onboarding",
        turns=[
            Turn(
                turn_id="r1",
                turn_index=0,
                speaker=SpeakerRole.AGENT,
                transcript="Could you please provide your PAN card number for verification purposes?",
                asr_confidence=0.95,
            ),
            Turn(
                turn_id="r2",
                turn_index=1,
                speaker=SpeakerRole.CUSTOMER,
                transcript="I don't have it with me right now.",
                asr_confidence=0.90,
            ),
            Turn(
                turn_id="r3",
                turn_index=2,
                speaker=SpeakerRole.AGENT,
                transcript="Can you please provide your PAN card number for verification purposes?",
                asr_confidence=0.95,
            ),
        ],
    )
