"""Vercel serverless entrypoint for ConvoLens FastAPI."""

import sys
import traceback
from pathlib import Path

# Add project root to sys.path so 'src' is always importable on Vercel / AWS Lambda
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

try:
    from src.main import app
except Exception as e:
    tb = traceback.format_exc()
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse

    app = FastAPI(title="ConvoLens Diagnostics")

    @app.api_route("/{path:path}", methods=["GET", "POST", "PUT", "DELETE"], response_class=HTMLResponse)
    def catch_all(path: str = ""):
        return HTMLResponse(
            f"""
            <!DOCTYPE html>
            <html>
            <head><title>ConvoLens Serverless Diagnostic</title></head>
            <body style="background:#090d16;color:#f87171;font-family:monospace;padding:2.5rem;line-height:1.6;">
                <h2 style="color:#fff;">ConvoLens Initialization Diagnostic</h2>
                <p style="color:#94a3b8;">The serverless function caught an exception during app boot:</p>
                <pre style="background:#111827;border:1px solid #1f2937;padding:1.5rem;border-radius:8px;color:#e2e8f0;overflow:auto;font-size:0.9rem;">{tb}</pre>
                <p style="color:#64748b;margin-top:1.5rem;">Path attempted: /{path}</p>
            </body>
            </html>
            """,
            status_code=500,
        )
