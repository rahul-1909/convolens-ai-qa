"""Unit tests for database session context manager."""

import pytest
from src.db.session import get_db, get_db_dependency
from src.models.database import ConversationRecord


def test_get_db_commit(db_session):
    """get_db context manager should commit on clean exit."""
    with get_db() as session:
        rec = ConversationRecord(id="db-test-commit")
        session.merge(rec)

    with get_db() as session:
        found = session.query(ConversationRecord).filter_by(id="db-test-commit").first()
        assert found is not None
        session.delete(found)


def test_get_db_rollback():
    """get_db context manager should rollback on exception."""
    with pytest.raises(ValueError):
        with get_db() as session:
            rec = ConversationRecord(id="db-test-rollback")
            session.add(rec)
            raise ValueError("Forced error for rollback testing")

    with get_db() as session:
        found = session.query(ConversationRecord).filter_by(id="db-test-rollback").first()
        assert found is None


def test_get_db_dependency():
    """Test FastAPI dependency generator."""
    gen = get_db_dependency()
    session = next(gen)
    assert session is not None
    try:
        next(gen)
    except StopIteration:
        pass
