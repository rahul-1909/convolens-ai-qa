"""Sentence-transformer embedding generation for failed turns.

Embeds turn transcripts so they can be clustered by HDBSCAN to surface
recurring failure patterns.
"""

from __future__ import annotations

import logging
from functools import lru_cache

import numpy as np

from src.config import get_settings

logger = logging.getLogger(__name__)
_SETTINGS = get_settings()


@lru_cache(maxsize=1)
def _load_model():
    """Lazily load the sentence-transformer model (cached singleton)."""
    try:
        from sentence_transformers import SentenceTransformer

        model_name = _SETTINGS.embedding_model
        logger.info("Loading embedding model: %s", model_name)
        return SentenceTransformer(model_name)
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Install with: pip install sentence-transformers"
        )
        return None


def embed_texts(texts: list[str]) -> np.ndarray:
    """Generate embeddings for a list of text strings.

    Args:
        texts: List of strings to embed.

    Returns:
        2-D numpy array of shape ``(len(texts), embedding_dim)``.

    Raises:
        RuntimeError: If sentence-transformers is not installed.
    """
    model = _load_model()
    if model is None:
        raise RuntimeError(
            "sentence-transformers is required for embedding generation."
        )
    embeddings = model.encode(texts, show_progress_bar=False, convert_to_numpy=True)
    return embeddings


def embed_single(text: str) -> np.ndarray:
    """Embed a single text string.

    Args:
        text: String to embed.

    Returns:
        1-D numpy array of the embedding vector.
    """
    return embed_texts([text])[0]
