"""Database session management — works with SQLite (dev/serverless) and PostgreSQL (prod)."""

from __future__ import annotations

import json
import logging
import os
import tempfile
from contextlib import contextmanager
from pathlib import Path
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.models.database import Base

logger = logging.getLogger(__name__)
_settings = get_settings()

# ── Dynamic Database URL (Serverless / Read-Only Filesystem Safe) ──────────────

_db_url = _settings.database_url
# On Vercel / AWS Lambda the project directory is read-only; use /tmp for SQLite
if os.environ.get("VERCEL") or os.environ.get("AWS_LAMBDA_FUNCTION_NAME"):
    if _db_url.startswith("sqlite:///."):
        temp_db = Path(tempfile.gettempdir()) / "convolens.db"
        _db_url = f"sqlite:///{temp_db.as_posix()}"

_connect_args: dict = {}
if _db_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(
    _db_url,
    connect_args=_connect_args,
    echo=(_settings.app_debug and _settings.app_env == "development"),
    pool_pre_ping=True,
)

# Enable foreign keys for SQLite
if _db_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        try:
            cursor = dbapi_conn.cursor()
            cursor.execute("PRAGMA foreign_keys=ON")
            cursor.close()
        except Exception:
            pass


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Helpers ──────────────────────────────────────────────────────────────────


def init_db() -> None:
    """Create all tables and auto-seed sample benchmark calls on first launch."""
    Base.metadata.create_all(bind=engine)

    # Auto-seed if newly created database
    try:
        from src.models.database import ConversationRecord
        from src.ingestion.normalizer import normalize_raw_transcript
        from src.evaluation.pipeline import EvaluationPipeline

        with get_db() as session:
            count = session.query(ConversationRecord).count()
            if count == 0:
                sample_files = [
                    "wealth_credit_onboarding_call.json",
                    "bfsi_hardship_collection_call.json",
                ]
                sample_dir = (
                    Path(__file__).resolve().parent.parent.parent
                    / "data"
                    / "samples"
                )
                for sf in sample_files:
                    path = sample_dir / sf
                    if path.exists():
                        try:
                            with open(path, "r", encoding="utf-8") as f:
                                data = json.load(f)
                            conv = normalize_raw_transcript(data)
                            pipeline = EvaluationPipeline(db=session)
                            pipeline.evaluate(conv)
                        except Exception as e:
                            logger.warning("Could not auto-seed %s: %s", sf, e)
    except Exception as e:
        logger.warning("Auto-seed skipped: %s", e)


@contextmanager
def get_db() -> Generator[Session, None, None]:
    """Yield a transactional DB session and handle commit/rollback."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def get_db_dependency() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a DB session per request."""
    session = SessionLocal()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
