"""Tests for plagiarism_checker.crosslingual."""

import numpy as np
import pytest

from plagiarism_checker.corpus import SentenceRecord
from plagiarism_checker.crosslingual import (
    get_pair_languages,
    detect_crosslingual_pairs,
    merge_crosslingual_hits,
)


def _make_crosslingual_rows():
    return [
        SentenceRecord(sid="A", did="A.txt", sent_id=0, text="Hello world.", lang="en"),
        SentenceRecord(sid="A", did="A.txt", sent_id=1, text="Good morning.", lang="en"),
        SentenceRecord(sid="B", did="B.txt", sent_id=0, text="你好世界。", lang="zh"),
        SentenceRecord(sid="B", did="B.txt", sent_id=1, text="早上好。", lang="zh"),
        SentenceRecord(sid="C", did="C.txt", sent_id=0, text="Another English text.", lang="en"),
    ]


def _make_embeddings(rows):
    np.random.seed(42)
    dim = 8
    emb = np.random.randn(len(rows), dim).astype("float32")
    # Make rows 0 and 2 somewhat similar (cross-lingual pair)
    emb[2] = emb[0] * 0.8 + np.random.randn(dim).astype("float32") * 0.2
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / norms
    return emb


class TestGetPairLanguages:
    def test_identifies_crosslingual(self):
        rows = _make_crosslingual_rows()
        pair_hits = {("A", "B"): [(0, 2, 0.75)]}
        result = get_pair_languages(rows, pair_hits)
        assert result[("A", "B")]["cross_lingual"] is True
        assert result[("A", "B")]["lang_a"] == "en"
        assert result[("A", "B")]["lang_b"] == "zh"

    def test_monolingual_pair(self):
        rows = _make_crosslingual_rows()
        pair_hits = {("A", "C"): [(0, 4, 0.85)]}
        result = get_pair_languages(rows, pair_hits)
        assert result[("A", "C")]["cross_lingual"] is False


class TestDetectCrosslingualPairs:
    def test_uses_lower_threshold(self):
        rows = _make_crosslingual_rows()
        emb = _make_embeddings(rows)
        import faiss
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        # With very low threshold, should find cross-lingual pairs
        hits = detect_crosslingual_pairs(rows, emb, index, k=5, threshold=0.0)
        # Should find some cross-lingual hits
        total = sum(len(v) for v in hits.values())
        assert total > 0

    def test_high_threshold_finds_fewer(self):
        rows = _make_crosslingual_rows()
        emb = _make_embeddings(rows)
        import faiss
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)

        hits_low = detect_crosslingual_pairs(rows, emb, index, k=5, threshold=0.0)
        hits_high = detect_crosslingual_pairs(rows, emb, index, k=5, threshold=0.99)
        total_low = sum(len(v) for v in hits_low.values())
        total_high = sum(len(v) for v in hits_high.values())
        assert total_low >= total_high


class TestMergeCrosslingualHits:
    def test_merges_and_deduplicates(self):
        regular = {("A", "B"): [(0, 2, 0.85)]}
        cross = {("A", "B"): [(1, 3, 0.70)]}
        merged = merge_crosslingual_hits(regular, cross)
        assert len(merged[("A", "B")]) == 2

    def test_dedup_same_hit(self):
        regular = {("A", "B"): [(0, 2, 0.85)]}
        cross = {("A", "B"): [(0, 2, 0.85)]}
        merged = merge_crosslingual_hits(regular, cross)
        assert len(merged[("A", "B")]) == 1
