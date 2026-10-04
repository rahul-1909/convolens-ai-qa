"""Unit tests for pattern mining and clustering fallback behaviors."""

from unittest.mock import patch
import numpy as np
import pytest

from src.mining.clustering import cluster_failed_turns, ClusterResult


def test_cluster_failed_turns_too_few_samples():
    """Clustering with fewer samples than min_cluster_size should return noise labels safely."""
    texts = ["failure sample 1", "failure sample 2"]
    res = cluster_failed_turns(texts, min_cluster_size=5)
    assert isinstance(res, ClusterResult)
    assert res.n_clusters == 0
    assert len(res.labels) == 2
    assert all(l == -1 for l in res.labels)


def test_cluster_failed_turns_mocked():
    """Test clustering execution with mocked embeddings and HDBSCAN."""
    texts = [f"sample failure text {i}" for i in range(10)]
    mock_embeddings = np.random.randn(10, 8)

    with patch("src.mining.embeddings.embed_texts", return_value=mock_embeddings):
        # Even if hdbscan is not installed or runs locally, test the import error handling or execution
        try:
            res = cluster_failed_turns(texts, min_cluster_size=2, min_samples=1)
            assert isinstance(res, ClusterResult)
        except RuntimeError as e:
            assert "Missing dependency" in str(e)
