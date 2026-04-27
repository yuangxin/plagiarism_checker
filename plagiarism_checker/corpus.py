"""
学生提交文本的加载与预处理工具：遍历目录、切分句子与段落、构建记录。
支持两种目录结构（学生文件夹/平铺文件）。
"""

from __future__ import annotations

import logging
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Iterator

logger = logging.getLogger(__name__)

# 常见英文缩写，不应作为句子边界
_ABBREVIATIONS = frozenset({
    "Mr", "Mrs", "Ms", "Dr", "Prof", "Sr", "Jr", "vs", "etc",
    "e.g", "i.e", "U.S", "U.K", "Inc", "Ltd", "Co", "Corp",
    "Univ", "Rev", "Gen", "Col", "Fig", "Eq", "Vol", "No",
    "al", "ed", "eds", "dept", "approx", "appt", "apt",
})

_CHINESE_BOUNDARY = re.compile(r"(?<=[。！？；])")
_ENGLISH_BOUNDARY = re.compile(r"(?<=[.!?;])")


@dataclass(frozen=True)
class SentenceRecord:
    """单个句子的记录"""
    sid: str           # 学生ID
    did: str           # 文档名
    sent_id: int       # 句子编号
    text: str          # 句子内容
    para_id: int = 0   # 所属段落编号
    lang: str = ""     # 语言标签 (ISO 639-1)


@dataclass(frozen=True)
class ParagraphRecord:
    """段落级别的记录"""
    sid: str
    did: str
    para_id: int
    text: str
    sent_count: int    # 该段落包含的句子数


def _merge_abbreviations(parts: list[str]) -> list[str]:
    """合并因缩写中的句号而错误分割的片段。"""
    if not parts:
        return parts
    merged = [parts[0]]
    for part in parts[1:]:
        prev = merged[-1].rstrip()
        last_word = prev.rsplit(None, 1)[-1].rstrip('.') if prev and not prev.endswith('\n') else ''
        if last_word and last_word.lower() in _ABBREVIATIONS:
            merged[-1] = merged[-1] + part
        else:
            merged.append(part)
    return merged


def split_sentences(text: str) -> list[str]:
    """
    切分文本为句子列表（中英文标点支持），正确处理常见缩写。

    Args:
        text: 原始文本。

    Returns:
        句子列表，已去除空白。
    """
    # 先按中文标点分割（总是安全的）
    segments = _CHINESE_BOUNDARY.split(text)

    results: list[str] = []
    for segment in segments:
        if not segment.strip():
            continue
        # 对每个中文片段再按英文标点分割
        parts = _ENGLISH_BOUNDARY.split(segment)
        merged = _merge_abbreviations(parts)
        results.extend(s.strip() for s in merged if s.strip())

    return results


def _detect_language(text: str) -> str:
    """检测文本语言，返回 ISO 639-1 代码。"""
    if len(text) < 10:
        return ""
    try:
        from langdetect import detect, LangDetectException
        return detect(text)
    except Exception:
        return ""


def split_paragraphs(text: str) -> List[str]:
    """
    按空行切分段落。

    Args:
        text: 原始文本。

    Returns:
        段落列表，已去除空白。
    """
    paragraphs = re.split(r'\n\s*\n', text)
    return [p.strip() for p in paragraphs if p.strip()]


def iter_documents(folder: Path) -> Iterator[tuple[str, Path]]:
    """
    遍历所有文档，支持两种结构：
    1) 每个学生一个文件夹，里面有多个文档；
    2) 所有文档平铺在一个文件夹。

    Args:
        folder: 根目录路径。

    Yields:
        (sid, doc_path): 学生ID与文档路径。
    """
    for entry in sorted(folder.iterdir()):
        if entry.is_dir():
            sid = entry.name
            for doc in sorted(entry.iterdir()):
                if doc.suffix.lower() in {".txt", ".md"} and doc.is_file():
                    yield sid, doc
        elif entry.suffix.lower() in {".txt", ".md"} and entry.is_file():
            yield entry.stem, entry


def load_corpus(folder: str | os.PathLike[str]) -> List[SentenceRecord]:
    """
    加载目录下所有文档的句子记录。

    Args:
        folder: 根目录路径。

    Returns:
        句子记录列表。
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到目录: {root}")

    rows: List[SentenceRecord] = []
    for sid, doc_path in iter_documents(root):
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = split_paragraphs(text)

        sent_counter = 0
        for para_id, para_text in enumerate(paragraphs):
            sentences = split_sentences(para_text)
            for sentence in sentences:
                if len(sentence) < 5:
                    continue
                lang = _detect_language(sentence)
                rows.append(
                    SentenceRecord(
                        sid=sid,
                        did=doc_path.name,
                        sent_id=sent_counter,
                        text=sentence,
                        para_id=para_id,
                        lang=lang,
                    )
                )
                sent_counter += 1
    logger.info("Loaded %d sentences from %s", len(rows), folder)
    return rows


def load_paragraphs(folder: str | os.PathLike[str]) -> List[ParagraphRecord]:
    """
    加载目录下所有文档的段落记录。

    Args:
        folder: 根目录路径。

    Returns:
        段落记录列表。
    """
    root = Path(folder)
    if not root.is_dir():
        raise FileNotFoundError(f"找不到目录: {root}")

    paras: List[ParagraphRecord] = []
    for sid, doc_path in iter_documents(root):
        text = doc_path.read_text(encoding="utf-8", errors="ignore")
        paragraphs = split_paragraphs(text)
        
        for para_id, para_text in enumerate(paragraphs):
            sentences = split_sentences(para_text)
            # 过滤太短的段落
            if len(para_text) < 20 or len(sentences) < 2:
                continue
            paras.append(
                ParagraphRecord(
                    sid=sid,
                    did=doc_path.name,
                    para_id=para_id,
                    text=para_text,
                    sent_count=len(sentences),
                )
            )
    return paras
