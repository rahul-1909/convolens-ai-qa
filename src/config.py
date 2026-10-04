"""Application configuration loaded from environment variables and .env file."""

from __future__ import annotations

import os
from pathlib import Path
from functools import lru_cache

from pydantic_settings import BaseSettings
from pydantic import Field


_PROJECT_ROOT = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    """Central configuration — reads from .env then environment variables."""

    # ── App ──────────────────────────────────────────────────────────────
    app_env: str = Field("development", alias="APP_ENV")
    app_debug: bool = Field(True, alias="APP_DEBUG")
    log_level: str = Field("INFO", alias="LOG_LEVEL")
    api_host: str = Field("0.0.0.0", alias="API_HOST")
    api_port: int = Field(8000, alias="API_PORT")

    # ── Database ─────────────────────────────────────────────────────────
    database_url: str = Field(
        "sqlite:///./convolens.db", alias="DATABASE_URL"
    )

    # ── LLM ──────────────────────────────────────────────────────────────
    anthropic_api_key: str = Field("", alias="ANTHROPIC_API_KEY")
    openai_api_key: str = Field("", alias="OPENAI_API_KEY")
    llm_provider: str = Field("claude", alias="LLM_PROVIDER")
    llm_model: str = Field("claude-sonnet-4-20250514", alias="LLM_MODEL")

    # ── Observability ────────────────────────────────────────────────────
    langfuse_public_key: str = Field("", alias="LANGFUSE_PUBLIC_KEY")
    langfuse_secret_key: str = Field("", alias="LANGFUSE_SECRET_KEY")
    langfuse_host: str = Field(
        "https://cloud.langfuse.com", alias="LANGFUSE_HOST"
    )

    # ── Embeddings ───────────────────────────────────────────────────────
    embedding_model: str = Field(
        "all-MiniLM-L6-v2", alias="EMBEDDING_MODEL"
    )

    # ── PII ──────────────────────────────────────────────────────────────
    pii_masking_enabled: bool = Field(True, alias="PII_MASKING_ENABLED")

    # ── Paths ────────────────────────────────────────────────────────────
    project_root: Path = _PROJECT_ROOT
    configs_dir: Path = _PROJECT_ROOT / "configs"

    model_config = {
        "env_file": str(_PROJECT_ROOT / ".env"),
        "env_file_encoding": "utf-8",
        "extra": "ignore",
        "populate_by_name": True,
    }


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return a cached singleton of application settings."""
    return Settings()
