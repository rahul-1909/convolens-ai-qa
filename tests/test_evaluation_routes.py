"""Integration tests for evaluation routes error handling and raw ingestion."""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.db.session import init_db


@pytest.fixture(scope="module")
def client():
    init_db()
    with TestClient(app) as c:
        yield c


def test_evaluate_raw_endpoint(client):
    """POST /evaluate/raw should normalize and evaluate raw utterance payloads."""
    raw_payload = {
        "conversation_id": "raw-conv-test-1",
        "agent_version": "v3.0.1",
        "conversation_goal": "payment_reminder",
        "utterances": [
            {
                "speaker": 0,
                "transcript": "Hello, this is an automated reminder regarding your loan.",
                "start": 0.0,
                "end": 3.0,
                "confidence": 0.98,
            },
            {
                "speaker": 1,
                "transcript": "Yes I will pay today.",
                "start": 3.2,
                "end": 5.0,
                "confidence": 0.92,
            },
        ],
    }
    response = client.post("/evaluate/raw", json=raw_payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["conversation_id"] == "raw-conv-test-1"
    assert "result" in data
    assert len(data["result"]["turn_results"]) == 2


def test_evaluate_conversation_invalid_payload(client):
    """POST /evaluate/conversation with malformed data should return 422 Unprocessable Entity."""
    response = client.post("/evaluate/conversation", json={"invalid": "payload"})
    assert response.status_code == 422
