"""Tests for plagiarism_checker.citation_analyzer."""

import pytest
from unittest.mock import MagicMock, patch

from plagiarism_checker.citation_analyzer import (
    CitationAssessment,
    CitationAnalyzer,
    _CITATION_PROMPT_TEMPLATE,
)


class TestCitationAssessment:
    def test_defaults(self):
        a = CitationAssessment()
        assert a.is_properly_cited is False
        assert a.citation_quality == 0.0
        assert a.paraphrase_level == "verbatim"
        assert a.is_common_knowledge is False
        assert a.adjusted_penalty == 1.0
        assert a.explanation == ""

    def test_custom_values(self):
        a = CitationAssessment(
            is_properly_cited=True,
            citation_quality=0.85,
            paraphrase_level="paraphrase",
            is_common_knowledge=False,
            adjusted_penalty=0.3,
            explanation="Properly cited with attribution.",
        )
        assert a.is_properly_cited is True
        assert a.citation_quality == 0.85
        assert a.adjusted_penalty == 0.3


class TestCitationAnalyzer:
    def test_fast_path_no_markers(self):
        """assess_single should return quickly when no citation markers are present."""
        analyzer = CitationAnalyzer()
        result = analyzer.assess_single(
            text_a="The sky is blue and grass is green.",
            text_b="Blue skies and green grass are common.",
            similarity=0.85,
        )
        assert not result.is_properly_cited
        assert result.adjusted_penalty == 1.0

    def test_cache_works(self):
        """Second call with same args should return cached result."""
        analyzer = CitationAnalyzer()
        r1 = analyzer.assess_single(
            text_a="Plain text without markers.",
            text_b="Another plain text.",
            similarity=0.80,
        )
        r2 = analyzer.assess_single(
            text_a="Plain text without markers.",
            text_b="Another plain text.",
            similarity=0.80,
        )
        assert r1 is r2  # Same object from cache

    def test_assess_batch_respects_max_items(self):
        """assess_batch should not process more than max_items."""
        analyzer = CitationAnalyzer()
        hits = [
            {"i": i, "j": i + 1, "sim": 0.9,
             "text_i": f"Sentence {i} without markers.",
             "text_j": f"Sentence {i+1} without markers."}
            for i in range(10)
        ]
        rows = [
            MagicMock(text=f"Sentence {i}.", lang="en")
            for i in range(12)
        ]
        result = analyzer.assess_batch(hits, rows, max_items=3)
        assert len(result) <= 3


class TestParseRawResponse:
    def test_parse_valid_json(self):
        analyzer = CitationAnalyzer()
        raw = '{"is_properly_cited": true, "citation_quality": 0.8, "paraphrase_level": "paraphrase", "is_common_knowledge": false, "adjusted_penalty": 0.4, "explanation": "test"}'
        result = analyzer._parse_raw_response(raw)
        assert result.is_properly_cited is True
        assert result.adjusted_penalty == 0.4

    def test_parse_json_with_markdown_fences(self):
        analyzer = CitationAnalyzer()
        raw = '```json\n{"is_properly_cited": false, "citation_quality": 0.0, "paraphrase_level": "verbatim", "is_common_knowledge": true, "adjusted_penalty": 0.2, "explanation": "common knowledge"}\n```'
        result = analyzer._parse_raw_response(raw)
        assert result.is_common_knowledge is True

    def test_parse_invalid_json(self):
        analyzer = CitationAnalyzer()
        raw = "not valid json at all"
        result = analyzer._parse_raw_response(raw)
        assert result.adjusted_penalty == 1.0
        assert "Failed" in result.explanation


class TestCitationAnalyzerWithMock:
    """Test the full flow with mocked LLM."""

    def test_llm_assessment_flow(self):
        analyzer = CitationAnalyzer()
        mock_result = {
            "is_properly_cited": True,
            "citation_quality": 0.9,
            "paraphrase_level": "digest",
            "is_common_knowledge": False,
            "adjusted_penalty": 0.3,
            "explanation": "Properly cited with full attribution.",
        }
        analyzer._ensure_agent = MagicMock()
        analyzer._agent = MagicMock()
        analyzer._agent._call_llm.return_value = mock_result

        result = analyzer.assess_single(
            text_a="As Smith (2020) notes, deep learning is transformative.",
            text_b="Deep learning has transformed the field significantly.",
            similarity=0.88,
        )
        assert result.is_properly_cited is True
        assert result.adjusted_penalty == 0.3
        assert result.paraphrase_level == "digest"
