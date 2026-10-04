"""Database session management — works with SQLite (dev) and PostgreSQL (prod)."""

from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

from sqlalchemy import create_engine, event
from sqlalchemy.orm import Session, sessionmaker

from src.config import get_settings
from src.models.database import Base

_settings = get_settings()

# ── Engine ───────────────────────────────────────────────────────────────────

_connect_args: dict = {}
if _settings.database_url.startswith("sqlite"):
    _connect_args["check_same_thread"] = False

engine = create_engine(
    _settings.database_url,
    connect_args=_connect_args,
    echo=(_settings.app_debug and _settings.app_env == "development"),
    pool_pre_ping=True,
)

# Enable WAL mode for SQLite for better concurrent read performance
if _settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_conn, connection_record):
        cursor = dbapi_conn.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)


# ── Helpers ──────────────────────────────────────────────────────────────────


def init_db() -> None:
    """Create all tables. Safe to call repeatedly (uses CREATE IF NOT EXISTS)."""
    Base.metadata.create_all(bind=engine)


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
