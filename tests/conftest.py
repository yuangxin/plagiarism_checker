"""Shared test fixtures."""

import os
import tempfile
from pathlib import Path

import pytest

from plagiarism_checker.corpus import SentenceRecord


@pytest.fixture
def sample_chinese_text():
    return "这是第一句话。这是第二句话！这是第三句？最后一句；结束。"


@pytest.fixture
def sample_english_text():
    return "First sentence here. Second sentence there! Third one? Done."


@pytest.fixture
def sample_mixed_text():
    return "Hello world.这是中文句子。Another English sentence！混合文本结束。"


@pytest.fixture
def tmp_corpus_dir(tmp_path):
    """Create a temp directory with sample .txt files for corpus testing."""
    (tmp_path / "A.txt").write_text(
        "Dr. Smith went to the U.S. for research. He published papers on AI. "
        "According to Lee (2020), deep learning is powerful.\n\n"
        "这是中文段落。包含多个句子。用于测试目的。",
        encoding="utf-8",
    )
    (tmp_path / "B.txt").write_text(
        "Machine learning transforms industries. "
        "Mr. Brown and Mrs. Davis co-authored the study [1]. "
        "正如张教授所指出的，人工智能正在改变世界。",
        encoding="utf-8",
    )
    return tmp_path


@pytest.fixture
def sample_rows():
    return [
        SentenceRecord(sid="A", did="A.txt", sent_id=0, text="Hello world.", para_id=0, lang="en"),
        SentenceRecord(sid="A", did="A.txt", sent_id=1, text="This is a test.", para_id=0, lang="en"),
        SentenceRecord(sid="B", did="B.txt", sent_id=0, text="Hello world.", para_id=0, lang="en"),
        SentenceRecord(sid="B", did="B.txt", sent_id=1, text="Different text here.", para_id=0, lang="en"),
        SentenceRecord(sid="C", did="C.txt", sent_id=0, text="你好世界。", para_id=0, lang="zh"),
    ]
