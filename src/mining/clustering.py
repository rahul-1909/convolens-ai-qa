"""HDBSCAN clustering for failure pattern discovery.

Embeds failed turns, clusters them, reduces dimensions with UMAP for
visualisation, and (optionally) auto-labels clusters with an LLM.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ClusterResult:
    """Output of a clustering run."""

    labels: np.ndarray                       # cluster label per input (-1 = noise)
    n_clusters: int                          # number of distinct clusters found
    cluster_sizes: dict[int, int]            # cluster_id -> size
    reduced_2d: np.ndarray | None = None     # UMAP 2-D coords for plotting
    sample_indices: dict[int, list[int]] = field(default_factory=dict)


def cluster_failed_turns(
    texts: list[str],
    min_cluster_size: int = 5,
    min_samples: int = 3,
) -> ClusterResult:
    """Embed, reduce, and cluster a list of failed-turn transcripts.

    Args:
        texts: Raw transcript strings of failed turns.
        min_cluster_size: HDBSCAN minimum cluster size.
        min_samples: HDBSCAN minimum samples parameter.

    Returns:
        ``ClusterResult`` with labels, sizes, and optional 2-D reduction.

    Raises:
        RuntimeError: If required libraries are not installed.
    """
    if len(texts) < min_cluster_size:
        logger.warning(
            "Too few texts (%d) to cluster (need >= %d). Returning empty.",
            len(texts),
            min_cluster_size,
        )
        return ClusterResult(
            labels=np.full(len(texts), -1),
            n_clusters=0,
            cluster_sizes={},
        )

    try:
        from src.mining.embeddings import embed_texts
        import hdbscan
    except ImportError as e:
        raise RuntimeError(
            f"Missing dependency for clustering: {e}. "
            "Install with: pip install hdbscan sentence-transformers"
        )

    # Step 1: Embed
    logger.info("Embedding %d failed turns...", len(texts))
    embeddings = embed_texts(texts)

    # Step 2: Reduce dimensions (UMAP) — improves HDBSCAN and gives viz coords
    reduced = embeddings
    reduced_2d = None
    try:
        import umap

        if embeddings.shape[0] > 15:
            n_components = min(50, embeddings.shape[1])
            reducer = umap.UMAP(
                n_components=n_components, random_state=42, metric="cosine"
            )
            reduced = reducer.fit_transform(embeddings)

            # Also make a 2-D reduction for visualisation
            viz_reducer = umap.UMAP(
                n_components=2, random_state=42, metric="cosine"
            )
            reduced_2d = viz_reducer.fit_transform(embeddings)
    except ImportError:
        logger.info("umap-learn not installed; clustering on raw embeddings.")

    # Step 3: Cluster with HDBSCAN
    logger.info("Running HDBSCAN (min_cluster_size=%d)...", min_cluster_size)
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        metric="euclidean",
    )
    labels = clusterer.fit_predict(reduced)

    unique_labels = set(labels)
    unique_labels.discard(-1)
    n_clusters = len(unique_labels)

    cluster_sizes = {}
    sample_indices: dict[int, list[int]] = {}
    for cid in unique_labels:
        indices = [i for i, l in enumerate(labels) if l == cid]
        cluster_sizes[cid] = len(indices)
        sample_indices[cid] = indices[:10]  # up to 10 samples for labelling

    noise_count = int(np.sum(labels == -1))
    logger.info(
        "Found %d clusters (%d noise points out of %d total)",
        n_clusters,
        noise_count,
        len(texts),
    )

    return ClusterResult(
        labels=labels,
        n_clusters=n_clusters,
        cluster_sizes=cluster_sizes,
        reduced_2d=reduced_2d,
        sample_indices=sample_indices,
    )
