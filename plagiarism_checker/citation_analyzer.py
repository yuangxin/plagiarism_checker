"""
LLM 智能引用评定：使用大语言模型对相似文本对进行引用质量评估。
替代规则硬编码的惩罚计算，提供更精细的引用分析。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import List

logger = logging.getLogger(__name__)


@dataclass
class CitationAssessment:
    """单条命中对的引用质量评估结果。"""
    is_properly_cited: bool = False
    citation_quality: float = 0.0
    paraphrase_level: str = "verbatim"   # "verbatim" | "paraphrase" | "digest"
    is_common_knowledge: bool = False
    adjusted_penalty: float = 1.0
    explanation: str = ""


_CITATION_PROMPT_TEMPLATE = """\
You are an academic citation quality assessor. Evaluate whether the following
similar text pair represents proper academic citation or plagiarism.

**Target sentence (under review):** "{text_a}"
**Reference sentence:** "{text_b}"
**Context (target):** {context_a}
**Context (reference):** {context_b}
**Detected citation markers:** {markers}
**Raw similarity:** {similarity:.2f}

Assess:
1. Is the target properly citing the reference? (with correct attribution)
2. Is it verbatim copying, paraphrasing, or original synthesis?
3. Could this be common knowledge?

Return ONLY a JSON object (no markdown fences, no extra text):
{{
  "is_properly_cited": true or false,
  "citation_quality": 0.0 to 1.0,
  "paraphrase_level": "verbatim" or "paraphrase" or "digest",
  "is_common_knowledge": true or false,
  "adjusted_penalty": 0.0 to 1.0,
  "explanation": "brief explanation"
}}
"""


class CitationAnalyzer:
    """使用 LLM 对相似文本对进行引用质量评估。"""

    def __init__(self, api_config_path: str = "api_config.json") -> None:
        self._agent = None
        self._api_config_path = api_config_path
        self._cache: dict = {}

    def _ensure_agent(self):
        """懒加载 Agent 实例。"""
        if self._agent is not None:
            return
        from .agent import SmartPlagiarismAgent
        self._agent = SmartPlagiarismAgent(self._api_config_path)

    def assess_single(
        self,
        text_a: str,
        text_b: str,
        similarity: float,
        context_a: str = "",
        context_b: str = "",
        markers: str = "",
    ) -> CitationAssessment:
        """
        评估单对文本的引用质量。

        Args:
            text_a: 待检测（嫌疑）句子。
            text_b: 参考句子。
            similarity: 原始相似度。
            context_a: 嫌疑侧上下文（前后句）。
            context_b: 参考侧上下文。
            markers: 规则检测到的引用标记。

        Returns:
            CitationAssessment 结构化评估结果。
        """
        # 缓存键
        cache_key = f"{text_a[:80]}|{text_b[:80]}|{similarity:.2f}"
        if cache_key in self._cache:
            return self._cache[cache_key]

        # 如果无引用标记且无引号，快速返回无引用
        from .citation import has_citation_marker, has_quotation_mark
        if not has_citation_marker(text_a) and not has_quotation_mark(text_a):
            result = CitationAssessment(
                is_properly_cited=False,
                citation_quality=0.0,
                paraphrase_level="verbatim",
                is_common_knowledge=False,
                adjusted_penalty=1.0,
                explanation="No citation markers detected.",
            )
            self._cache[cache_key] = result
            return result

        prompt = _CITATION_PROMPT_TEMPLATE.format(
            text_a=text_a[:300],
            text_b=text_b[:300],
            context_a=context_a[:200] or "(none)",
            context_b=context_b[:200] or "(none)",
            markers=markers or "(none detected)",
            similarity=similarity,
        )

        try:
            self._ensure_agent()
            raw = self._agent._call_llm(prompt)
            if isinstance(raw, dict):
                result = CitationAssessment(
                    is_properly_cited=bool(raw.get("is_properly_cited", False)),
                    citation_quality=float(raw.get("citation_quality", 0.0)),
                    paraphrase_level=str(raw.get("paraphrase_level", "verbatim")),
                    is_common_knowledge=bool(raw.get("is_common_knowledge", False)),
                    adjusted_penalty=float(raw.get("adjusted_penalty", 1.0)),
                    explanation=str(raw.get("explanation", "")),
                )
            else:
                result = self._parse_raw_response(raw)
        except Exception as e:
            logger.warning("CitationAnalyzer LLM call failed: %s", e)
            result = CitationAssessment(
                is_properly_cited=False,
                adjusted_penalty=1.0,
                explanation=f"LLM error: {e}",
            )

        self._cache[cache_key] = result
        return result

    def assess_batch(
        self,
        hits: List[dict],
        rows: list,
        *,
        max_items: int = 20,
    ) -> dict:
        """
        批量评估命中列表中的引用质量。

        Args:
            hits: 命中列表，每项含 i, j, sim, text_i, text_j 等。
            rows: SentenceRecord 列表，用于构建上下文。
            max_items: 最多评估的命中数。

        Returns:
            字典 {(i, j): CitationAssessment}
        """
        results: dict = {}
        # 按相似度降序排列，优先评估高相似度的
        sorted_hits = sorted(hits, key=lambda h: h.get("sim", 0), reverse=True)
        for hit in sorted_hits[:max_items]:
            i = hit.get("i", hit.get("idx_i", 0))
            j = hit.get("j", hit.get("idx_j", 0))
            text_a = hit.get("text_i", "")
            text_b = hit.get("text_j", "")
            sim = hit.get("sim", 0.0)

            # 构建上下文（前后各1句）
            context_a = self._build_context(rows, i)
            context_b = self._build_context(rows, j)

            # 收集引用标记
            from .citation import has_citation_marker
            markers = "citation markers found" if has_citation_marker(text_a) else ""

            assessment = self.assess_single(
                text_a, text_b, sim,
                context_a=context_a,
                context_b=context_b,
                markers=markers,
            )
            results[(i, j)] = assessment

        logger.info("CitationAnalyzer: assessed %d/%d hits", len(results), len(hits))
        return results

    def _build_context(self, rows: list, idx: int) -> str:
        """构建第 idx 行的上下文（前后各1句）。"""
        parts = []
        if idx > 0 and idx - 1 < len(rows):
            parts.append(rows[idx - 1].text)
        if idx < len(rows):
            parts.append(rows[idx].text)
        if idx + 1 < len(rows):
            parts.append(rows[idx + 1].text)
        return " ... ".join(parts)

    def _parse_raw_response(self, raw) -> CitationAssessment:
        """尝试从原始响应中解析 JSON。"""
        if hasattr(raw, "get"):
            text = raw.get("raw_response", str(raw))
        else:
            text = str(raw)

        # 尝试提取 JSON
        try:
            # 去除可能的 markdown 代码块标记
            text = text.strip()
            if text.startswith("```"):
                text = text.split("\n", 1)[-1]
            if text.endswith("```"):
                text = text.rsplit("```", 1)[0]
            data = json.loads(text)
            return CitationAssessment(
                is_properly_cited=bool(data.get("is_properly_cited", False)),
                citation_quality=float(data.get("citation_quality", 0.0)),
                paraphrase_level=str(data.get("paraphrase_level", "verbatim")),
                is_common_knowledge=bool(data.get("is_common_knowledge", False)),
                adjusted_penalty=float(data.get("adjusted_penalty", 1.0)),
                explanation=str(data.get("explanation", "")),
            )
        except (json.JSONDecodeError, ValueError):
            return CitationAssessment(
                adjusted_penalty=1.0,
                explanation="Failed to parse LLM response.",
            )
