"""
引用检测与惩罚：基于引用标记与引号，区分合理引用与抄袭。
方向性策略：仅当嫌疑侧（左侧文本）存在引用/引号时降低分数，避免参考侧标注稀释结论。
"""

from __future__ import annotations

import re
from typing import Set


# 常见的引用标记模式
CITATION_PATTERNS = [
    r'\[[0-9]+\]',                      # [1] [2]
    r'\([^)]*[0-9]{4}[^)]*\)',         # (Smith, 2020)
    r'[Aa]ccording to\s+[\w\s]+',          # according to Smith
    r'[Aa]s\s+[\w\s]+\s+stated',           # as Smith stated
    r'根据.{1,10}',                     # 根据某某某
    r'引用.{1,10}',                     # 引用某某某
    r'参考.{1,10}',                     # 参考某某某
    r'如.{1,10}所说',                   # 如某某所说
    r'正如.{1,10}指出',                 # 正如某某指出
]

# 引号模式
QUOTE_PATTERNS = [
    r'"[^"]+"',
    r'「[^」]+」',
    r'『[^』]+』',
    r"'[^']+'",
]


def has_citation_marker(text: str) -> bool:
    """
    检查文本中是否存在引用标记（如数字引用、括号年份、中文“参考/引用/根据”等）。

    Args:
        text: 待检测文本。

    Returns:
        True/False，是否包含引用标记。
    """
    for pattern in CITATION_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def has_quotation_mark(text: str) -> bool:
    """
    检查文本是否使用了引号（中英文引号均支持）。

    Args:
        text: 待检测文本。

    Returns:
        True/False，是否包含引号。
    """
    for pattern in QUOTE_PATTERNS:
        if re.search(pattern, text):
            return True
    return False


def is_likely_citation(text: str) -> bool:
    """
    判断文本是否可能为引用：存在引用标记或引号即认为是引用。

    Args:
        text: 待检测文本。

    Returns:
        True/False，是否可能为引用。
    """
    return has_citation_marker(text) or has_quotation_mark(text)




def compute_citation_penalty(
    text_a: str,
    text_b: str,
    similarity: float,
    left_did: str | None = None,
    right_did: str | None = None,
) -> float:
    """
    计算引用惩罚系数（0-1）。

    策略：
    - 若两侧文本均含引用/引号，返回 0.3（共同引用同源可能性高）。
    - 若仅左侧文本（嫌疑侧）含引用/引号，返回 0.6（可能为合理引用）。
    - 其他情况返回 1.0（不降低）。

    Args:
        text_a: 左侧文本（嫌疑/待检测）。
        text_b: 右侧文本（参考）。
        similarity: 原始相似度，用于后续可扩展（当前未使用）。

    Returns:
        惩罚系数（float）。
    """
    left_generic = has_citation_marker(text_a) or has_quotation_mark(text_a)
    right_generic = has_citation_marker(text_b) or has_quotation_mark(text_b)

    # 构建来源候选词集合（基于右侧内容与文档名）
    source_terms = build_source_candidates(text_b, right_did)
    explicit_to_right = contains_source_specific_citation(text_a, source_terms)

    # 明确指向右侧来源
    if explicit_to_right and has_quotation_mark(text_a):
        return 0.40
    if explicit_to_right:
        return 0.60
    if left_generic and right_generic:
        return 0.60
    if has_quotation_mark(text_a):
        return 0.75
    if has_citation_marker(text_a):
        return 0.85
    return 1.00
def _tokenize_candidates(text: str) -> Set[str]:
    """
    简易候选词提取：
    - 英文：首字母大写的单词（可能是人名/标题关键词）
    - 中文：连续的中文词组（长度≥2）
    - 年份：常见年份模式（用于辅助匹配）
    """
    words_en = set(re.findall(r"\b[A-Z][a-z]+\b", text))
    words_zh = set(re.findall(r"[\u4e00-\u9fa5]{2,}", text))
    years = set(re.findall(r"\b(?:19|20)\d{2}\b", text))
    return words_en.union(words_zh).union(years)


def _stem_to_terms(doc_name: str) -> Set[str]:
    """
    根据文档名（不含扩展名）生成候选词集合。
    例如："Smith_2020_Transformers.txt" → {"Smith","2020","Transformers"}。
    """
    stem = doc_name
    if "." in doc_name:
        stem = doc_name.rsplit(".", 1)[0]
    parts = re.split(r"[_\-\s]+|[\W]", stem)
    terms = {p for p in parts if p and (len(p) >= 2)}
    return terms


def build_source_candidates(reference_text: str, reference_doc_name: str | None) -> Set[str]:
    """
    构建“参考文献源”的候选词集合，用于在左侧文本中判断是否明确指向右侧来源。

    Args:
        reference_text: 参考侧（右侧）文本内容。
        reference_doc_name: 参考侧文档名（用于抽取标题/作者线索）。

    Returns:
        候选词集合（英文人名/关键词、中文词组、年份等）。
    """
    candidates = _tokenize_candidates(reference_text)
    if reference_doc_name:
        candidates.update(_stem_to_terms(reference_doc_name))
    # 过滤过短/纯数字词
    filtered = {c for c in candidates if (len(c) >= 2 and not re.fullmatch(r"\d+", c))}
    return set(list(filtered)[:50])


def contains_source_specific_citation(text_left: str, source_terms: Set[str]) -> bool:
    """
    检测左侧文本是否存在“明确指向右侧来源”的引用：
    - 规则：左侧文本中出现来源候选词，且附近（±50字符）存在引用标记或括号年份等模式。
    """
    if not source_terms:
        return False
    # 先快速判断是否存在引用/引号以降低计算
    has_generic = has_citation_marker(text_left) or has_quotation_mark(text_left)
    if not has_generic:
        return False
    for term in source_terms:
        for m in re.finditer(re.escape(term), text_left):
            start = max(0, m.start() - 50)
            end = min(len(text_left), m.end() + 50)
            window = text_left[start:end]
            if has_citation_marker(window) or has_quotation_mark(window) or re.search(r"\b(?:19|20)\d{2}\b", window):
                return True
    return False
