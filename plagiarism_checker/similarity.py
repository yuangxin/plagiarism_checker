"""
相似度检测逻辑，包含句子级与段落级的相似配对与聚合。
提供方向性（有序）配对：(嫌疑/待检测, 参考) 的区分，以支持多文档六方向对比。
"""

from __future__ import annotations

import numpy as np
from collections import defaultdict
from typing import Dict, List, Tuple

from .corpus import SentenceRecord, ParagraphRecord
from .citation import compute_citation_penalty


def detect_pairs(
    rows: List[SentenceRecord],
    embeddings: np.ndarray,
    index,
    *,
    k: int = 5,
    threshold: float = 0.82,
    allowed_left: set[str] | None = None,
    allowed_right: set[str] | None = None,
) -> Dict[Tuple[str, str], List[Tuple[int, int, float]]]:
    """
    句子级方向性配对检测。

    Args:
        rows: 句子记录列表，每项包含学生ID/文档名/句子编号/文本等。
        embeddings: 句子向量矩阵，按 `rows` 顺序排列。
        index: FAISS 索引，用于相似度搜索（内积/余弦）。
        k: 每个句子保留的命中数量上限（不含自身）。
        threshold: 相似度阈值，低于此值的命中将被过滤。

    Returns:
        一个字典，键为方向性 `(sid_i, sid_j)`，值为命中列表 `[(i, j, sim)]`，
        其中 i/j 为在 `rows` 中的索引，表示 i 属于嫌疑/待检测侧，j 属于参考侧。
    """
    pair_hits: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = defaultdict(list)
    
    for i, row in enumerate(rows):
        # 搜索最相似的 k+5 个句子（多搜一些，避免都命中自己）
        distances, indices = index.search(embeddings[i : i + 1], k + 5)
        sid_i = row.sid
        taken = 0
        if allowed_left and sid_i not in allowed_left:
            continue
        
        for sim, j in zip(distances[0], indices[0]):
            # 跳过自身命中
            if j == i:
                continue
            sid_j = rows[j].sid
            # 同一学生之间不计入交叉抄袭
            if sid_i == sid_j:
                continue
            # 过滤低相似度命中
            if sim < threshold:
                continue
            if allowed_right and sid_j not in allowed_right:
                continue
            
            pair_key = (sid_i, sid_j)
            pair_hits[pair_key].append((i, j, float(sim)))
            taken += 1
            if taken >= k:
                break
    
    return pair_hits


def detect_paragraph_pairs(
    paras: List[ParagraphRecord],
    embeddings: np.ndarray,
    index,
    *,
    k: int = 3,
    threshold: float = 0.75,
    allowed_left: set[str] | None = None,
    allowed_right: set[str] | None = None,
) -> Dict[Tuple[str, str], List[Tuple[int, int, float]]]:
    """
    段落级方向性配对检测。段落更长，因此默认阈值略低。

    Args:
        paras: 段落记录列表，包含段落文本及所属学生/文档。
        embeddings: 段落向量矩阵。
        index: FAISS 索引。
        k: 每个段落保留的命中数量上限。
        threshold: 段落相似度阈值。

    Returns:
        字典，键为方向性 `(sid_i, sid_j)`，值为命中列表 `[(i, j, sim)]`。
    """
    pair_hits: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = defaultdict(list)
    
    for i, para in enumerate(paras):
        distances, indices = index.search(embeddings[i : i + 1], k + 5)
        sid_i = para.sid
        taken = 0
        if allowed_left and sid_i not in allowed_left:
            continue
        
        for sim, j in zip(distances[0], indices[0]):
            if j == i:
                continue
            sid_j = paras[j].sid
            if sid_i == sid_j:
                continue
            if sim < threshold:
                continue
            if allowed_right and sid_j not in allowed_right:
                continue
            
            pair_key = (sid_i, sid_j)
            pair_hits[pair_key].append((i, j, float(sim)))
            taken += 1
            if taken >= k:
                break
    
    return pair_hits


def detect_pairs_crossset(
    rows_a: List[SentenceRecord],
    rows_b: List[SentenceRecord],
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    index_b,
    *,
    k: int | None = None,
    threshold: float = 0.82,
    index_map_a: List[int],
    index_map_b: List[int],
) -> Dict[Tuple[str, str], List[Tuple[int, int, float]]]:
    # 中文注释：仅以目标集合 rows_a 在参考集合索引 index_b 上进行检索；消除全排列与“非参考占满TopK”的问题。
    pair_hits: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = defaultdict(list)
    search_k = (len(rows_b) if k is None else min(len(rows_b), k))
    for i_local, row in enumerate(rows_a):
        distances, indices = index_b.search(embeddings_a[i_local : i_local + 1], search_k)
        sid_i = row.sid
        for sim, j_local in zip(distances[0], indices[0]):
            if sim < threshold:
                continue
            sid_j = rows_b[j_local].sid
            if sid_i == sid_j:
                continue
            i_global = index_map_a[i_local]
            j_global = index_map_b[j_local]
            pair_hits[(sid_i, sid_j)].append((i_global, j_global, float(sim)))
    return pair_hits


def detect_paragraph_pairs_crossset(
    paras_a: List[ParagraphRecord],
    paras_b: List[ParagraphRecord],
    embeddings_a: np.ndarray,
    embeddings_b: np.ndarray,
    index_b,
    *,
    k: int | None = None,
    threshold: float = 0.75,
    index_map_a: List[int],
    index_map_b: List[int],
) -> Dict[Tuple[str, str], List[Tuple[int, int, float]]]:
    # 中文注释：段落级跨集合检索；仅在参考集合上搜索，命中以阈值过滤，不受TopK上限影响。
    pair_hits: Dict[Tuple[str, str], List[Tuple[int, int, float]]] = defaultdict(list)
    search_k = (len(paras_b) if k is None else min(len(paras_b), k))
    for i_local, para in enumerate(paras_a):
        distances, indices = index_b.search(embeddings_a[i_local : i_local + 1], search_k)
        sid_i = para.sid
        for sim, j_local in zip(distances[0], indices[0]):
            if sim < threshold:
                continue
            sid_j = paras_b[j_local].sid
            if sid_i == sid_j:
                continue
            i_global = index_map_a[i_local]
            j_global = index_map_b[j_local]
            pair_hits[(sid_i, sid_j)].append((i_global, j_global, float(sim)))
    return pair_hits


def aggregate_pairs(
    rows: List[SentenceRecord],
    pair_hits: Dict[Tuple[str, str], List[Tuple[int, int, float]]],
    use_citation_penalty: bool = True,
) -> List[dict]:
    """
    聚合方向性学生对的句子级统计，并按综合分数排序。

    Args:
        rows: 句子记录列表。
        pair_hits: 有序对命中映射 `(sid_i, sid_j) -> [(i, j, sim)]`。
        use_citation_penalty: 是否应用引用惩罚（左侧为嫌疑侧）。

    Returns:
        每对的统计字典列表，包含命中数、均值/最大相似度、覆盖率、总句子数与综合分数。
    """
    sent_count = defaultdict(int)
    for row in rows:
        sent_count[row.sid] += 1

    stats: List[dict] = []
    for pair, hits in pair_hits.items():
        sid_a, sid_b = pair
        
        # 应用引用惩罚（仅当嫌疑侧/左侧存在引用标记时降低）
        adjusted_hits = []
        penalties = []
        for i, j, sim in hits:
            if use_citation_penalty:
                penalty = compute_citation_penalty(
                    rows[i].text,
                    rows[j].text,
                    sim,
                    left_did=rows[i].did,
                    right_did=rows[j].did,
                )
                adjusted_sim = sim * penalty
                penalties.append(penalty)
            else:
                adjusted_sim = sim
            adjusted_hits.append((i, j, adjusted_sim))
        
        sims = [h[2] for h in adjusted_hits]
        if not sims:
            continue
        
        # 统计覆盖的句子数（按方向分别统计，后续取较小值作为保守估计）
        sentences_a = {rows[i].sent_id for i, _, _ in hits if rows[i].sid == sid_a}
        sentences_b = {rows[j].sent_id for _, j, _ in hits if rows[j].sid == sid_b}
        coverage_a = len(sentences_a) / sent_count[sid_a]
        coverage_b = len(sentences_b) / sent_count[sid_b]
        
        # 综合评分公式：平均相似度 40%，覆盖率（较小值） 35%，最大相似度 15%，命中数量 10%
        mean_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))
        coverage_min = min(coverage_a, coverage_b)
        hit_ratio = min(len(hits) / 50.0, 1.0)  # 命中数量归一化到 0-1
        
        # 来源特异性与引用惩罚的统计
        if penalties:
            avg_citation_penalty = float(np.mean(penalties))
            # 简化映射：明确引用(<=0.60)->1.0；一般引用(<1.0)->0.5；无引用(==1.0)->0.0
            specificity_vals = [1.0 if p <= 0.60 else (0.5 if p < 1.0 else 0.0) for p in penalties]
            avg_source_specificity = float(np.mean(specificity_vals))
        else:
            avg_citation_penalty = 1.0
            avg_source_specificity = 0.0

        score = (
            0.40 * mean_sim +
            0.35 * coverage_min +
            0.15 * max_sim +
            0.10 * hit_ratio
        )
        
        stats.append(
            {
                "pair": pair,
                "count": len(hits),
                "mean_sim": mean_sim,
                "max_sim": max_sim,
                "coverage_min": float(coverage_min),
                "coverage_a": float(coverage_a),
                "coverage_b": float(coverage_b),
                "student_a_sent_total": int(sent_count[sid_a]),
                "student_b_sent_total": int(sent_count[sid_b]),
                "avg_citation_penalty": avg_citation_penalty,
                "avg_source_specificity": avg_source_specificity,
                "score": float(score),
            }
        )

    stats.sort(key=lambda item: item["score"], reverse=True)
    return stats


def aggregate_paragraph_pairs(
    paras: List[ParagraphRecord],
    pair_hits: Dict[Tuple[str, str], List[Tuple[int, int, float]]],
) -> List[dict]:
    """
    聚合方向性学生对的段落级统计。

    Args:
        paras: 段落记录列表。
        pair_hits: 有序对命中映射。

    Returns:
        每对的段落级统计列表，包含命中数、均值/最大相似度、覆盖率与分数。
    """
    para_count = defaultdict(int)
    for para in paras:
        para_count[para.sid] += 1

    stats: List[dict] = []
    for pair, hits in pair_hits.items():
        sid_a, sid_b = pair
        sims = [h[2] for h in hits]
        if not sims:
            continue
        
        paras_a = {paras[i].para_id for i, _, _ in hits if paras[i].sid == sid_a}
        paras_b = {paras[j].para_id for _, j, _ in hits if paras[j].sid == sid_b}
        coverage_a = len(paras_a) / para_count[sid_a] if para_count[sid_a] > 0 else 0
        coverage_b = len(paras_b) / para_count[sid_b] if para_count[sid_b] > 0 else 0
        
        mean_sim = float(np.mean(sims))
        max_sim = float(np.max(sims))
        coverage_min = min(coverage_a, coverage_b)
        
        score = (
            0.45 * mean_sim +
            0.35 * coverage_min +
            0.20 * max_sim
        )
        
        stats.append(
            {
                "pair": pair,
                "count": len(hits),
                "mean_sim": mean_sim,
                "max_sim": max_sim,
                "coverage_min": float(coverage_min),
                "coverage_a": float(coverage_a),
                "coverage_b": float(coverage_b),
                "student_a_para_total": int(para_count[sid_a]),
                "student_b_para_total": int(para_count[sid_b]),
                "score": float(score),
            }
        )

    stats.sort(key=lambda item: item["score"], reverse=True)
    return stats


def build_pair_details(
    rows: List[SentenceRecord],
    stats: List[dict],
    pair_hits: Dict[Tuple[str, str], List[Tuple[int, int, float]]],
    *,
    max_hits: int = 50,
) -> List[dict]:
    """
    构建方向性对的详细命中记录，用于界面与报告展示。

    Args:
        rows: 句子记录列表。
        stats: 对应有序对的汇总统计。
        pair_hits: 命中映射。
        max_hits: 每对最多保留的命中条数。

    Returns:
        详细记录列表，包含每对的命中细节与按学生/句子组织的证据。
    """
    details: List[dict] = []
    
    for summary in stats:
        pair = tuple(summary["pair"])
        hits_raw = pair_hits.get(pair, [])[:max_hits]

        sentences = {}

        def ensure_entry(record: SentenceRecord) -> dict:
            sid = record.sid
            sent_id = int(record.sent_id)
            if sid not in sentences:
                sentences[sid] = {}
            if sent_id not in sentences[sid]:
                sentences[sid][sent_id] = {
                    "text": record.text,
                    "did": record.did,
                    "hits": [],
                }
            return sentences[sid][sent_id]

        normalized_hits = []
        for idx_i, idx_j, sim in hits_raw:
            rec_i = rows[idx_i]
            rec_j = rows[idx_j]
            
            # 引用惩罚（方向性）：仅当左侧存在引用/引号时降低
            citation_penalty = compute_citation_penalty(
                rec_i.text,
                rec_j.text,
                sim,
                left_did=rec_i.did,
                right_did=rec_j.did,
            )
            # 解释标签
            if citation_penalty <= 0.60:
                citation_label = "明确引用"
                source_specificity = 1.0
            elif citation_penalty < 1.0:
                citation_label = "一般引用"
                source_specificity = 0.5
            else:
                citation_label = "无引用"
                source_specificity = 0.0

            normalized = {
                "i": int(idx_i),
                "j": int(idx_j),
                "sim": float(sim),
                "adjusted_sim": float(sim * citation_penalty),
                "citation_penalty": float(citation_penalty),
                "citation_label": citation_label,
                "source_specificity": float(source_specificity),
                "sid_i": rec_i.sid,
                "sid_j": rec_j.sid,
                "did_i": rec_i.did,
                "did_j": rec_j.did,
                "sent_id_i": int(rec_i.sent_id),
                "sent_id_j": int(rec_j.sent_id),
                "text_i": rec_i.text,
                "text_j": rec_j.text,
            }
            normalized_hits.append(normalized)

            left_entry = ensure_entry(rec_i)
            left_entry["hits"].append(
                {
                    "other_sid": rec_j.sid,
                    "other_sent_id": int(rec_j.sent_id),
                    "other_text": rec_j.text,
                    "sim": float(sim),
                }
            )

            right_entry = ensure_entry(rec_j)
            right_entry["hits"].append(
                {
                    "other_sid": rec_i.sid,
                    "other_sent_id": int(rec_i.sent_id),
                    "other_text": rec_i.text,
                    "sim": float(sim),
                }
            )

        details.append(
            {
                "pair": list(pair),
                "count": summary["count"],
                "mean_sim": summary["mean_sim"],
                "max_sim": summary["max_sim"],
                "coverage_min": summary["coverage_min"],
                "coverage_a": summary["coverage_a"],
                "coverage_b": summary["coverage_b"],
                "student_a_sent_total": summary["student_a_sent_total"],
                "student_b_sent_total": summary["student_b_sent_total"],
                "score": summary["score"],
                "hits": normalized_hits,
                "sentences": sentences,
            }
        )
    return details
