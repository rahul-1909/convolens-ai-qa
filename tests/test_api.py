"""Integration tests for FastAPI endpoints."""

from __future__ import annotations

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from src.main import app
from src.db.session import init_db


@pytest.fixture(scope="module")
def client():
    """Create test client with initialized database."""
    init_db()
    with TestClient(app) as c:
        yield c


def test_health_check(client):
    """GET /health should return 200 OK and healthy status."""
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert "version" in data
    assert data["database"] == "connected"


def test_root_endpoint(client):
    """GET / should return basic platform info."""
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert data["app"] == "ConvoLens"
    assert data["docs"] == "/docs"


def test_evaluate_conversation_endpoint(client, sample_conversation):
    """POST /evaluate/conversation should accept conversation and return evaluation."""
    payload = {
        "conversation": json.loads(sample_conversation.model_dump_json()),
        "run_pattern_mining": False,
    }
    response = client.post("/evaluate/conversation", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "success"
    assert data["conversation_id"] == sample_conversation.conversation_id
    assert "result" in data
    res = data["result"]
    assert "overall_scores" in res
    assert 1.0 <= res["overall_scores"]["composite"] <= 5.0
    assert len(res["turn_results"]) == len(sample_conversation.turns)


def test_get_results_endpoint(client, sample_conversation):
    """GET /results/{id} should return persisted evaluation data."""
    # First evaluate to ensure it is in DB
    payload = {
        "conversation": json.loads(sample_conversation.model_dump_json()),
        "run_pattern_mining": False,
    }
    eval_resp = client.post("/evaluate/conversation", json=payload)
    assert eval_resp.status_code == 200

    # Retrieve
    conv_id = sample_conversation.conversation_id
    get_resp = client.get(f"/results/{conv_id}")
    assert get_resp.status_code == 200
    data = get_resp.json()
    assert data["found"] is True
    assert data["conversation_id"] == conv_id
    assert data["result"]["overall_scores"]["composite"] > 0


def test_get_results_not_found(client):
    """GET /results/non-existent-id should return found=False."""
    response = client.get("/results/non-existent-conv-id-999")
    assert response.status_code == 200
    data = response.json()
    assert data["found"] is False
    assert data["result"] is None


def test_trends_endpoint(client):
    """GET /trends should return aggregated quality and failure metrics."""
    response = client.get("/trends")
    assert response.status_code == 200
    data = response.json()
    assert "total_conversations" in data
    assert "total_failures" in data
    assert "average_composite_score" in data
    assert "failure_distribution" in data
    assert "top_root_causes" in data
