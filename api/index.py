"""Vercel serverless entrypoint for ConvoLens FastAPI."""

import sys
from pathlib import Path

# Add project root to sys.path so 'src' is always importable on Vercel / AWS Lambda
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src.main import app
