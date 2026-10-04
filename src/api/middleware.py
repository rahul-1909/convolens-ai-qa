"""API middleware — CORS, request logging, Vercel routing, and global exception handling."""

from __future__ import annotations

import logging
import time

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

logger = logging.getLogger(__name__)


def add_middleware(app: FastAPI) -> None:
    """Attach middleware and exception handlers to the FastAPI app.

    Args:
        app: The FastAPI application instance.
    """
    # CORS — allow all for development; restrict in production
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    @app.middleware("http")
    async def vercel_path_rewrite(request: Request, call_next):
        """Map Vercel internal rewrite headers/params back to the original client request path."""
        target_path = request.query_params.get("__path")
        if not target_path:
            raw = (
                request.headers.get("x-matched-path")
                or request.headers.get("x-forwarded-uri")
                or request.headers.get("x-vercel-matched-path")
                or request.headers.get("x-original-uri")
                or request.headers.get("x-rewrite-url")
            )
            if raw and raw not in ("/api/index.py", "/api/index", "/api"):
                target_path = raw

        if target_path:
            clean_path = "/" + target_path.lstrip("/").split("?")[0]
            request.scope["path"] = clean_path
        elif request.scope.get("path") in ("/api/index.py", "/api/index", "/api"):
            request.scope["path"] = "/"

        return await call_next(request)

    @app.middleware("http")
    async def log_requests(request: Request, call_next):
        """Log request method, path, and response time."""
        start = time.perf_counter()
        response = await call_next(request)
        elapsed = (time.perf_counter() - start) * 1000
        logger.info(
            "%s %s -> %d (%.1fms)",
            request.method,
            request.url.path,
            response.status_code,
            elapsed,
        )
        return response

    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception):
        """Catch unhandled exceptions and return a clean JSON error."""
        logger.error(
            "Unhandled error on %s %s: %s",
            request.method,
            request.url.path,
            exc,
            exc_info=True,
        )
        return JSONResponse(
            status_code=500,
            content={
                "error": "Internal server error",
                "detail": str(exc),
                "path": request.url.path,
            },
        )
