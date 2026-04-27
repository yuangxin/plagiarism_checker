"""Tests for plagiarism_checker.similarity."""

import numpy as np
import pytest

from plagiarism_checker.corpus import SentenceRecord
from plagiarism_checker.similarity import (
    detect_pairs,
    aggregate_pairs,
    build_pair_details,
)


def _make_rows():
    """Create sample rows with known structure."""
    return [
        SentenceRecord(sid="A", did="A.txt", sent_id=0, text="The cat sat on the mat.", lang="en"),
        SentenceRecord(sid="A", did="A.txt", sent_id=1, text="Dogs bark loudly at night.", lang="en"),
        SentenceRecord(sid="B", did="B.txt", sent_id=0, text="The cat sat on the mat.", lang="en"),
        SentenceRecord(sid="B", did="B.txt", sent_id=1, text="Completely unrelated sentence.", lang="en"),
    ]


def _make_embeddings(rows):
    """Create simple embeddings where identical text gets identical vectors."""
    np.random.seed(42)
    dim = 8
    emb = np.random.randn(len(rows), dim).astype("float32")
    # Make rows 0 and 2 identical (same text)
    emb[2] = emb[0]
    # Normalize
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return emb


class TestDetectPairs:
    def test_finds_identical_pairs(self):
        rows = _make_rows()
        emb = _make_embeddings(rows)
        index = __import__("faiss").IndexFlatIP(emb.shape[1])
        index.add(emb)

        hits = detect_pairs(rows, emb, index, k=5, threshold=0.5)
        # Should find A↔B pairs (rows 0 and 2 are identical)
        assert len(hits) > 0

    def test_filters_below_threshold(self):
        rows = _make_rows()
        emb = _make_embeddings(rows)
        index = __import__("faiss").IndexFlatIP(emb.shape[1])
        index.add(emb)

        hits = detect_pairs(rows, emb, index, k=5, threshold=0.999)
        # Very high threshold should find fewer pairs
        total_hits = sum(len(v) for v in hits.values())
        # With threshold 0.999, only nearly-identical pairs pass
        assert total_hits <= 2

    def test_skips_same_student(self):
        rows = _make_rows()
        emb = _make_embeddings(rows)
        index = __import__("faiss").IndexFlatIP(emb.shape[1])
        index.add(emb)

        hits = detect_pairs(rows, emb, index, k=5, threshold=0.0)
        # No pair should have same sid on both sides
        for (sid_a, sid_b), pair_hits in hits.items():
            assert sid_a != sid_b


class TestAggregatePairs:
    def test_returns_sorted(self):
        rows = _make_rows()
        emb = _make_embeddings(rows)
        index = __import__("faiss").IndexFlatIP(emb.shape[1])
        index.add(emb)

        hits = detect_pairs(rows, emb, index, k=5, threshold=0.5)
        stats = aggregate_pairs(rows, hits, use_citation_penalty=False)
        scores = [s["score"] for s in stats]
        assert scores == sorted(scores, reverse=True)

    def test_has_required_fields(self):
        rows = _make_rows()
        emb = _make_embeddings(rows)
        index = __import__("faiss").IndexFlatIP(emb.shape[1])
        index.add(emb)

        hits = detect_pairs(rows, emb, index, k=5, threshold=0.5)
        stats = aggregate_pairs(rows, hits, use_citation_penalty=False)
        if stats:
            s = stats[0]
            for key in ["pair", "count", "mean_sim", "max_sim", "score", "coverage_min"]:
                assert key in s


class TestBuildPairDetails:
    def test_includes_citation_fields(self):
        rows = _make_rows()
        emb = _make_embeddings(rows)
        index = __import__("faiss").IndexFlatIP(emb.shape[1])
        index.add(emb)

        hits = detect_pairs(rows, emb, index, k=5, threshold=0.5)
        stats = aggregate_pairs(rows, hits, use_citation_penalty=True)
        details = build_pair_details(rows, stats, hits, max_hits=50)

        if details:
            for d in details:
                assert "hits" in d
                for h in d["hits"]:
                    assert "citation_penalty" in h
                    assert "citation_label" in h
                    assert "adjusted_sim" in h
