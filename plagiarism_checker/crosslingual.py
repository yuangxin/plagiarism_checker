"""
跨语言抄袭检测：检测翻译抄袭（如中文翻译英文原文）。
提供跨语言配对检测、回译评分和 LLM 翻译等价性判断。
"""

from __future__ import annotations

import logging
from collections import defaultdict
from typing import Dict, List, Tuple

import numpy as np

from .corpus import SentenceRecord

logger = logging.getLogger(__name__)


def get_pair_languages(
    rows: List[SentenceRecord],
    pair_hits: Dict[Tuple[str, str], list],
) -> Dict[Tuple[str, str], dict]:
    """
    分析每对文档的语言组合。

    Returns:
        字典 {(sid_a, sid_b): {"lang_a": "zh", "lang_b": "en", "cross_lingual": True, ...}}
    """
    # 统计每个 sid 中各语言句子的数量
    lang_count: Dict[str, dict] = defaultdict(lambda: defaultdict(int))
    for row in rows:
        if row.lang:
            lang_count[row.sid][row.lang] += 1

    # 每个 sid 的主导语言
    dominant_lang: Dict[str, str] = {}
    for sid, counts in lang_count.items():
        if counts:
            dominant_lang[sid] = max(counts, key=counts.get)  # type: ignore[arg-type]

    pair_langs: Dict[Tuple[str, str], dict] = {}
    for pair in pair_hits:
        sid_a, sid_b = pair
        lang_a = dominant_lang.get(sid_a, "")
        lang_b = dominant_lang.get(sid_b, "")
        pair_langs[pair] = {
            "lang_a": lang_a,
            "lang_b": lang_b,
            "cross_lingual": bool(lang_a and lang_b and lang_a != lang_b),
        }
    return pair_langs


def detect_crosslingual_pairs(
    rows: List[SentenceRecord],
    embeddings: np.ndarray,
    index,
    *,
    k: int = 5,
    threshold: float = 0.65,
) -> Dict[Tuple[str, str], List[Tuple[int, int, float]]]:
    """
    跨语言句子级配对检测，使用更宽松的阈值。

    翻译后的句子在语义向量空间中距离通常比原文更大，
    因此跨语言检测使用较低的阈值（默认 0.65 vs 普通的 0.82）。

    Args:
        rows: 句子记录列表（需含 lang 字段）。
        embeddings: 句子向量矩阵。
        index: FAISS 索引。
        k: 每句保留的命中上限。
        threshold: 跨语言相似度阈值（比普通阈值更宽松）。

    Returns:
        字典 {(sid_i, sid_j): [(i, j, sim)]}，仅包含跨语言命中。
    """
    pair_hits: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = defaultdict(list)

    for i, row in enumerate(rows):
        if not row.lang:
            continue
        distances, indices = index.search(embeddings[i : i + 1], k + 10)
        sid_i = row.sid
        lang_i = row.lang
        taken = 0

        for sim, j in zip(distances[0], indices[0]):
            if j == i:
                continue
            sid_j = rows[j].sid
            if sid_i == sid_j:
                continue
            lang_j = rows[j].lang
            # 仅保留不同语言的配对
            if not lang_j or lang_i == lang_j:
                continue
            if sim < threshold:
                continue

            pair_hits[(sid_i, sid_j)].append((i, j, float(sim)))
            taken += 1
            if taken >= k:
                break

    logger.info(
        "Cross-lingual detection found %d pairs (threshold=%.2f)",
        len(pair_hits), threshold,
    )
    return pair_hits


def assess_translation_equivalence(
    text_a: str,
    text_b: str,
    lang_a: str,
    lang_b: str,
    similarity: float,
    agent=None,
) -> dict:
    """
    使用 LLM 评估两段文本是否为翻译等价关系。

    Args:
        text_a: 目标文本。
        text_b: 参考文本。
        lang_a: 目标语言。
        lang_b: 参考语言。
        similarity: 向量相似度。
        agent: 可选的 SmartPlagiarismAgent 实例。

    Returns:
        评估结果字典。
    """
    if agent is None:
        return {
            "is_translation": False,
            "confidence": 0.0,
            "explanation": "No LLM agent available for translation assessment",
        }

    lang_names = {"zh": "Chinese", "en": "English", "fr": "French",
                  "de": "German", "ja": "Japanese", "ko": "Korean",
                  "es": "Spanish", "ru": "Russian"}

    prompt = f"""You are a cross-lingual plagiarism analyst. Determine if the following two texts
are translation equivalents (one is a translation of the other).

**Text A** ({lang_names.get(lang_a, lang_a)}): "{text_a[:300]}"
**Text B** ({lang_names.get(lang_b, lang_b)}): "{text_b[:300]}"
**Vector similarity**: {similarity:.2f}

Analyze:
1. Is Text A a translation of Text B (or vice versa)?
2. What type of translation: verbatim, paraphrased, or adapted?
3. Are there signs of machine translation?

Return JSON:
{{
  "is_translation": true/false,
  "confidence": 0.0-1.0,
  "translation_type": "verbatim"|"paraphrased"|"adapted",
  "machine_translation_likelihood": 0.0-1.0,
  "explanation": "..."
}}"""

    result = agent._call_llm(prompt)
    if isinstance(result, dict) and "is_translation" in result:
        return result
    return {
        "is_translation": False,
        "confidence": 0.0,
        "explanation": str(result.get("raw_response", "LLM parse error")),
    }


def merge_crosslingual_hits(
    regular_hits: Dict[Tuple[str, str], list],
    crosslingual_hits: Dict[Tuple[str, str], list],
) -> Dict[Tuple[str, str], list]:
    """
    合并普通命中和跨语言命中，去重。

    Args:
        regular_hits: 普通同语言检测结果。
        crosslingual_hits: 跨语言检测结果。

    Returns:
        合并后的命中字典。
    """
    merged: Dict[Tuple[str, str], list] = defaultdict(list)
    seen: set = set()

    for pair, hits in regular_hits.items():
        for hit in hits:
            key = (hit[0], hit[1])
            if key not in seen:
                merged[pair].append(hit)
                seen.add(key)

    for pair, hits in crosslingual_hits.items():
        for hit in hits:
            key = (hit[0], hit[1])
            if key not in seen:
                merged[pair].append(hit)
                seen.add(key)

    return dict(merged)
