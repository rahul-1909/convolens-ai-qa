"""ConvoLens — FastAPI application entry point.

Run with:
    uvicorn src.main:app --reload --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import HTMLResponse

from src import __version__
from src.config import get_settings
from src.db.session import init_db
from src.utils.logging import setup_logging
from src.api.middleware import add_middleware
from src.api.routes.evaluation import router as evaluation_router
from src.api.routes.results import router as results_router
from src.api.routes.trends import router as trends_router
from src.schemas.api import HealthResponse

# ── Setup ────────────────────────────────────────────────────────────────────

setup_logging()
_settings = get_settings()
_STATIC_INDEX = Path(__file__).resolve().parent / "static" / "index.html"


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown lifecycle."""
    init_db()
    yield


# ── App Factory ──────────────────────────────────────────────────────────────

app = FastAPI(
    title="ConvoLens",
    description=(
        "Automated Quality Evaluation & Failure-Taxonomy Platform "
        "for Voice and Chat AI Agents"
    ),
    version=__version__,
    lifespan=lifespan,
    docs_url="/docs",
    redoc_url="/redoc",
)

# Attach middleware
add_middleware(app)

# Register route modules
app.include_router(evaluation_router)
app.include_router(results_router)
app.include_router(trends_router)


# ── Root / Health ────────────────────────────────────────────────────────────


@app.get("/", response_class=HTMLResponse, tags=["dashboard"])
def root():
    """Single unified showcase dashboard with live interactive evaluation."""
    if _STATIC_INDEX.exists():
        return HTMLResponse(content=_STATIC_INDEX.read_text(encoding="utf-8"))
    return HTMLResponse(
        f"<h1>ConvoLens v{__version__}</h1><p><a href='/docs'>Swagger API Docs</a></p>"
    )


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health_check():
    """Health check endpoint for load balancers and monitoring."""
    return HealthResponse(
        status="healthy",
        version=__version__,
        database="connected",
        timestamp=datetime.now(timezone.utc),
    )
