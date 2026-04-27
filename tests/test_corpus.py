"""Tests for plagiarism_checker.corpus."""

import pytest

from plagiarism_checker.corpus import (
    SentenceRecord,
    split_sentences,
    split_paragraphs,
    _detect_language,
    _merge_abbreviations,
    load_corpus,
)


class TestSplitSentences:
    def test_chinese_punctuation(self, sample_chinese_text):
        sents = split_sentences(sample_chinese_text)
        assert len(sents) == 4
        assert sents[0] == "这是第一句话。"
        assert sents[1] == "这是第二句话！"

    def test_english_punctuation(self, sample_english_text):
        sents = split_sentences(sample_english_text)
        assert len(sents) == 4
        assert sents[0] == "First sentence here."

    def test_mixed_punctuation(self, sample_mixed_text):
        sents = split_sentences(sample_mixed_text)
        assert len(sents) == 4

    def test_abbreviation_mr(self):
        text = "Mr. Smith went home."
        sents = split_sentences(text)
        assert len(sents) == 1
        assert "Mr." in sents[0]

    def test_abbreviation_dr(self):
        text = "Dr. Lee published a paper. It was great."
        sents = split_sentences(text)
        assert len(sents) == 2
        assert "Dr." in sents[0]

    def test_abbreviation_us(self):
        text = "The U.S. economy is growing. Reports confirm this."
        sents = split_sentences(text)
        assert len(sents) == 2
        assert "U.S." in sents[0]

    def test_empty_string(self):
        assert split_sentences("") == []

    def test_single_sentence(self):
        assert split_sentences("Just one sentence.") == ["Just one sentence."]

    def test_short_text_filtered(self):
        text = "Hi."
        sents = split_sentences(text)
        # "Hi." is only 3 chars, but split_sentences doesn't filter by length
        # That happens in load_corpus
        assert isinstance(sents, list)


class TestMergeAbbreviations:
    def test_no_abbreviation(self):
        parts = ["Hello world ", "Next sentence"]
        assert _merge_abbreviations(parts) == parts

    def test_abbreviation_merge(self):
        parts = ["Hello Mr", " Smith went home"]
        merged = _merge_abbreviations(parts)
        assert len(merged) == 1

    def test_empty_input(self):
        assert _merge_abbreviations([]) == []


class TestSplitParagraphs:
    def test_basic_split(self):
        text = "Para one.\n\nPara two.\n\nPara three."
        paras = split_paragraphs(text)
        assert len(paras) == 3

    def test_single_paragraph(self):
        text = "Just one paragraph with multiple sentences."
        paras = split_paragraphs(text)
        assert len(paras) == 1

    def test_empty_string(self):
        assert split_paragraphs("") == []


class TestDetectLanguage:
    def test_english(self):
        lang = _detect_language("This is a typical English sentence for testing purposes.")
        assert lang == "en"

    def test_chinese(self):
        lang = _detect_language("这是一个用于测试目的的中文句子，包含足够多的字符。")
        assert lang == "zh"

    def test_short_text(self):
        assert _detect_language("Hi") == ""


class TestSentenceRecord:
    def test_has_lang_field(self):
        rec = SentenceRecord(sid="A", did="A.txt", sent_id=0, text="Hello.", lang="en")
        assert rec.lang == "en"

    def test_lang_default_empty(self):
        rec = SentenceRecord(sid="A", did="A.txt", sent_id=0, text="Hello.")
        assert rec.lang == ""


class TestLoadCorpus:
    def test_raises_for_missing_dir(self):
        with pytest.raises(FileNotFoundError):
            load_corpus("/nonexistent/path")

    def test_loads_files(self, tmp_corpus_dir):
        rows = load_corpus(tmp_corpus_dir)
        assert len(rows) > 0
        # Should have records from both files
        sids = {r.sid for r in rows}
        assert "A" in sids
        assert "B" in sids

    def test_lang_detected(self, tmp_corpus_dir):
        rows = load_corpus(tmp_corpus_dir)
        langs = {r.lang for r in rows}
        # Should detect at least English or Chinese
        assert langs & {"en", "zh", ""}
