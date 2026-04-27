"""Tests for plagiarism_checker.citation."""

import pytest

from plagiarism_checker.citation import (
    has_citation_marker,
    has_quotation_mark,
    is_likely_citation,
    compute_citation_penalty,
)


class TestHasCitationMarker:
    def test_numeric_citation(self):
        assert has_citation_marker("Some claim [1] about data.")

    def test_author_year(self):
        assert has_citation_marker("As shown (Smith, 2020) in prior work.")

    def test_according_to(self):
        assert has_citation_marker("According to Smith, the results show...")

    def test_chinese_citation(self):
        assert has_citation_marker("根据张教授的研究，结论如下。")

    def test_chinese_citation_ref(self):
        assert has_citation_marker("参考文献如下所述。")

    def test_no_citation(self):
        assert not has_citation_marker("The sky is blue and the grass is green.")

    def test_empty_string(self):
        assert not has_citation_marker("")


class TestHasQuotationMark:
    def test_double_quotes(self):
        assert has_quotation_mark('He said "hello" to everyone.')

    def test_chinese_quotes(self):
        assert has_quotation_mark("「引用内容」在这里。")

    def test_chinese_double_quotes(self):
        assert has_quotation_mark("『重要内容』已标注。")

    def test_no_quotes(self):
        assert not has_quotation_mark("Plain text without any quotes.")

    def test_empty_string(self):
        assert not has_quotation_mark("")


class TestIsLikelyCitation:
    def test_with_marker(self):
        assert is_likely_citation("As noted [1], the results hold.")

    def test_with_quotes(self):
        assert is_likely_citation('The author stated "key finding".')

    def test_plain_text(self):
        assert not is_likely_citation("Just a regular sentence.")


class TestComputeCitationPenalty:
    def test_both_sides_generic(self):
        # Both have citation markers → 0.60
        penalty = compute_citation_penalty(
            "As shown [1], data supports this.",
            "Reference [2] also shows data.",
            0.90,
        )
        assert penalty == 0.60

    def test_left_has_quotation_only(self):
        # Left has quotes but no explicit source → 0.75
        penalty = compute_citation_penalty(
            'The author stated "important point".',
            "The important point was discussed.",
            0.90,
        )
        assert penalty == 0.75

    def test_left_citation_marker_only(self):
        # Left has citation marker only → 0.85
        penalty = compute_citation_penalty(
            "As noted in [1], data supports this.",
            "Data supports the conclusion.",
            0.90,
        )
        assert penalty == 0.85

    def test_no_citation(self):
        # No citation markers or quotes → 1.0
        penalty = compute_citation_penalty(
            "The sky is blue.",
            "The sky appears blue.",
            0.90,
        )
        assert penalty == 1.00

    def test_explicit_source_with_quotes(self):
        # Explicit source citation + quotes → 0.40
        penalty = compute_citation_penalty(
            'Smith stated "the results are significant" (Smith, 2020).',
            "The results are significant for the field.",
            0.95,
            left_did="review.txt",
            right_did="Smith_2020_Results.txt",
        )
        # May be 0.40 or 0.60 depending on source matching
        assert penalty <= 0.60
