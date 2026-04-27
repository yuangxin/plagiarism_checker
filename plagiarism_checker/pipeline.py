"""
抄袭检测的主要流程控制：加载语料、向量化、索引构建、相似配对、聚合与报告输出。
支持句子级/段落级检测、引用惩罚、并行加速、跨语言与智能Agent分析。
"""

from __future__ import annotations

from dataclasses import dataclass
import json
import logging
from pathlib import Path
from typing import List, Tuple

logger = logging.getLogger(__name__)

import numpy as np

from .corpus import SentenceRecord, load_corpus, load_paragraphs
from .embedder import (
    build_embeddings,
    build_embeddings_parallel,
    build_multilingual_embeddings,
    build_index,
)
from .similarity import (
    detect_pairs,
    detect_paragraph_pairs,
    detect_pairs_crossset,
    detect_paragraph_pairs_crossset,
    aggregate_pairs,
    aggregate_paragraph_pairs,
    build_pair_details,
)
from .reporting import (
    write_summary_csv,
    write_pair_results,
    write_evidence_top,
    write_paragraph_summary,
    write_word_report,
    write_word_summary_report,
)
from .crosslingual import (
    detect_crosslingual_pairs,
    merge_crosslingual_hits,
    get_pair_languages,
)


@dataclass
class PipelineConfig:
    submissions_dir: Path = Path("./paraphrase_outputs")
    model_name: str = "all-MiniLM-L6-v2"
    device: str | None = None              # 设备选择：None自动, 'cpu', 'cuda'
    use_parallel: bool = False             # CPU多线程并行加速
    num_workers: int = 2                   # 并行worker数量
    index_top_k: int = 5
    similarity_threshold: float = 0.82
    max_hits_per_pair: int = 50
    output_dir: Path = Path(".")
    
    # 新增功能开关
    enable_paragraph_check: bool = True    # 段落级检测开关
    enable_citation_check: bool = True     # 引用惩罚开关（方向性）
    enable_multilingual: bool = False      # 跨语言检测开关
    
    # 段落检测参数
    para_top_k: int = 3
    para_threshold: float = 0.75

    # 新增Agent配置
    enable_agent: bool = False             # 智能Agent分析开关
    agent_threshold: float = 0.70          # 触发分析的风险分数阈值
    api_config_path: str = "api_config.json"  # Agent API配置路径
    agent_max_reports: int = 3
    agent_dual_phase: bool = False

    # LLM 引用评定配置
    enable_citation_llm: bool = False      # 使用 LLM 替代规则引用评定
    citation_llm_max: int = 20             # 每对最多评估命中数

    # 目标模式过滤：仅保留 (左∈targets, 右∈references) 的方向性对
    target_stems: List[str] | None = None
    reference_stems: List[str] | None = None


class PlagiarismPipeline:
    """端到端的抄袭检测流程。将配置与各处理阶段组织为可复用管道。"""

    def __init__(self, config: PipelineConfig) -> None:
        self.config = config

    def _build_embeddings(self, texts: list[str]) -> np.ndarray:
        """Select and build embeddings based on pipeline configuration."""
        cfg = self.config
        if cfg.enable_multilingual:
            return build_multilingual_embeddings(
                texts, model_name="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2", device=cfg.device,
            )
        elif cfg.use_parallel and (cfg.device is None or cfg.device == 'cpu'):
            return build_embeddings_parallel(
                texts, model_name=cfg.model_name, device='cpu', num_workers=cfg.num_workers,
            )
        else:
            return build_embeddings(
                texts, model_name=cfg.model_name, device=cfg.device,
            )

    def run(self) -> Tuple[List[dict], List[dict]]:
        """
        执行句子级检测流程。

        Returns:
            (sent_stats, sent_details): 句子级统计与详细命中。
        """
        cfg = self.config
        
        # 1) 加载语料
        rows = load_corpus(cfg.submissions_dir)
        if not rows:
            raise RuntimeError(f"{cfg.submissions_dir} 里没找到有效文本")

        # 2) 选择模型并向量化（支持多语言/并行/单机）
        embeddings = self._build_embeddings([row.text for row in rows])

        # 3) 建立相似度索引
        index = build_index(embeddings)
        
        # 4) 句子级方向性配对检测
        if cfg.target_stems:
            # 中文注释：Target模式下按“目标集合 vs 参考集合”分组向量化与索引，仅在参考索引上搜索，避免全排列。
            idx_a = [i for i, r in enumerate(rows) if r.sid in set(cfg.target_stems)]
            idx_b = [i for i, r in enumerate(rows) if r.sid in set(cfg.reference_stems or [])]
            rows_a = [rows[i] for i in idx_a]
            rows_b = [rows[i] for i in idx_b]
            emb_a = self._build_embeddings([r.text for r in rows_a])
            emb_b = self._build_embeddings([r.text for r in rows_b])
            index_b = build_index(emb_b)
            pair_hits = detect_pairs_crossset(
                rows_a,
                rows_b,
                emb_a,
                emb_b,
                index_b,
                k=None,
                threshold=cfg.similarity_threshold,
                index_map_a=idx_a,
                index_map_b=idx_b,
            )
        else:
            pair_hits = detect_pairs(
                rows,
                embeddings,
                index,
                k=cfg.index_top_k,
                threshold=cfg.similarity_threshold,
            )
        
        # 5) 跨语言检测（当 enable_multilingual 时额外搜索跨语言命中）
        cross_lingual_pairs: dict = {}
        if cfg.enable_multilingual:
            cross_hits = detect_crosslingual_pairs(
                rows, embeddings, index,
                k=cfg.index_top_k, threshold=0.65,
            )
            if cross_hits:
                pair_hits = merge_crosslingual_hits(pair_hits, cross_hits)
                cross_lingual_pairs = get_pair_languages(rows, pair_hits)
                logger.info(
                    "Cross-lingual merge: %d pairs, %d cross-lingual",
                    len(pair_hits),
                    sum(1 for v in cross_lingual_pairs.values() if v.get("cross_lingual")),
                )

        # 6) 聚合统计（可选引用惩罚）
        stats = aggregate_pairs(
            rows,
            pair_hits,
            use_citation_penalty=cfg.enable_citation_check,
        )

        # 标注跨语言信息到统计
        for s in stats:
            pair = tuple(s["pair"])
            cl_info = cross_lingual_pairs.get(pair, {})
            s["cross_lingual"] = cl_info.get("cross_lingual", False)
            s["lang_a"] = cl_info.get("lang_a", "")
            s["lang_b"] = cl_info.get("lang_b", "")

        # 7) 构建详细命中记录
        details = build_pair_details(
            rows,
            stats,
            pair_hits,
            max_hits=cfg.max_hits_per_pair,
        )

        # 传递跨语言标记到 details
        for d in details:
            pair = tuple(d["pair"])
            cl_info = cross_lingual_pairs.get(pair, {})
            d["cross_lingual"] = cl_info.get("cross_lingual", False)
            d["lang_a"] = cl_info.get("lang_a", "")
            d["lang_b"] = cl_info.get("lang_b", "")

        return stats, details

    def run_with_paragraphs(self) -> Tuple[List[dict], List[dict], List[dict], List[dict]]:
        """
        同时执行句子级与段落级检测。

        Returns:
            (sent_stats, sent_details, para_stats, para_details)
        """
        cfg = self.config
        
        # 句子级检测（重用 run()）
        sent_stats, sent_details = self.run()
        
        if not cfg.enable_paragraph_check:
            return sent_stats, sent_details, [], []
        
        # 段落级检测
        paras = load_paragraphs(cfg.submissions_dir)
        if not paras:
            return sent_stats, sent_details, [], []

        para_embeddings = self._build_embeddings([p.text for p in paras])
        
        # 段落索引
        para_index = build_index(para_embeddings)
        
        if cfg.target_stems:
            # 中文注释：段落级同样采用跨集合检索逻辑，以阈值过滤命中。
            idx_pa = [i for i, p in enumerate(paras) if p.sid in set(cfg.target_stems)]
            idx_pb = [i for i, p in enumerate(paras) if p.sid in set(cfg.reference_stems or [])]
            paras_a = [paras[i] for i in idx_pa]
            paras_b = [paras[i] for i in idx_pb]
            emb_pa = self._build_embeddings([p.text for p in paras_a])
            emb_pb = self._build_embeddings([p.text for p in paras_b])
            para_index_b = build_index(emb_pb)
            para_pair_hits = detect_paragraph_pairs_crossset(
                paras_a,
                paras_b,
                emb_pa,
                emb_pb,
                para_index_b,
                k=None,
                threshold=cfg.para_threshold,
                index_map_a=idx_pa,
                index_map_b=idx_pb,
            )
        else:
            para_pair_hits = detect_paragraph_pairs(
                paras,
                para_embeddings,
                para_index,
                k=cfg.para_top_k,
                threshold=cfg.para_threshold,
            )
        
        para_stats = aggregate_paragraph_pairs(paras, para_pair_hits)
        
        # 段落详情（简版）：保留核心字段，便于界面/报告展示
        para_details = []
        for summary in para_stats:
            pair = tuple(summary["pair"])
            hits_raw = para_pair_hits.get(pair, [])[:cfg.max_hits_per_pair]
            
            para_matches = []
            for idx_i, idx_j, sim in hits_raw:
                para_i = paras[idx_i]
                para_j = paras[idx_j]
                para_matches.append({
                    "sid_i": para_i.sid,
                    "sid_j": para_j.sid,
                    "para_id_i": para_i.para_id,
                    "para_id_j": para_j.para_id,
                    "sim": float(sim),
                    "text_i": para_i.text[:200] + "..." if len(para_i.text) > 200 else para_i.text,
                    "text_j": para_j.text[:200] + "..." if len(para_j.text) > 200 else para_j.text,
                })
            
            para_details.append({
                "pair": list(pair),
                "score": summary["score"],
                "count": summary["count"],
                "mean_sim": summary["mean_sim"],
                "max_sim": summary["max_sim"],
                "coverage_min": summary["coverage_min"],
                "coverage_a": summary["coverage_a"],
                "coverage_b": summary["coverage_b"],
                "matches": para_matches,
            })
        
        return sent_stats, sent_details, para_stats, para_details

    def write_reports(
        self,
        stats: List[dict],
        details: List[dict],
        para_stats: List[dict] = None,
        para_details: List[dict] = None,
    ) -> None:
        """
        写入各种报告文件（CSV/JSON/Word）。

        Args:
            stats: 句子级统计列表。
            details: 句子级详细命中列表。
            para_stats: 段落级统计列表。
            para_details: 段落级详细命中列表。
        """
        output_dir = self.config.output_dir
        output_dir.mkdir(parents=True, exist_ok=True)

        # 句子级报告
        write_summary_csv(output_dir / "pair_summary.csv", stats)
        write_pair_results(output_dir / "pair_results.json", details)
        write_evidence_top(output_dir / "evidence_top.json", details)
        
        # 生成Word报告
        try:
            # 详细Word报告（包含具体匹配内容）
            write_word_report(output_dir / "plagiarism_report.docx", stats, details)
            
            # 汇总Word报告（仅统计信息）
            write_word_summary_report(output_dir / "plagiarism_summary_report.docx", stats)
        except Exception as e:
            logger.warning("Error generating Word report: %s", e)

        # 段落级报告
        if para_stats and para_details:
            write_paragraph_summary(
                output_dir / "paragraph_summary.csv",
                para_stats
            )
            write_pair_results(
                output_dir / "paragraph_results.json",
                para_details
            )
            
            # 段落级Word报告
            try:
                # 为段落数据添加hits字段以兼容Word报告格式
                para_details_with_hits = []
                for detail in para_details:
                    detail_copy = dict(detail)
                    detail_copy['hits'] = detail.get('matches', [])
                    para_details_with_hits.append(detail_copy)
                write_word_report(output_dir / "plagiarism_paragraph_report.docx", para_stats, para_details_with_hits)
            except Exception as e:
                logger.warning("Error generating paragraph Word report: %s", e)

    def run_with_agent(self) -> Tuple[List, List, List]:
        """
        带 Agent 的句子级检测与深度分析流程。

        Returns:
            (sent_stats, sent_details, agent_reports)
        """
        logger.info("Step 1: Running standard plagiarism detection...")
        # 执行常规检测
        sent_stats, sent_details = self.run()
        logger.info("Found %d document pairs with matches", len(sent_stats))
        
        # 如果未启用Agent
        if not self.config.enable_agent:
            logger.info("Agent is disabled in config")
            return sent_stats, sent_details, []
        
        logger.info("Step 2: Initializing AI Agent...")
        # 尝试导入agent模块
        try:
            from .agent import SmartPlagiarismAgent, generate_agent_report, generate_agent_report_batch
            logger.info("Agent module imported successfully")
        except ImportError as e:
            logger.error("Failed to import agent module: %s", e)
            return sent_stats, sent_details, []
        
        # 初始化 Agent
        try:
            agent = SmartPlagiarismAgent(self.config.api_config_path, dual_phase=self.config.agent_dual_phase)
            logger.info("Agent initialized with config: %s", self.config.api_config_path)
        except Exception as e:
            import traceback
            logger.error("Agent initialization failed:")
            logger.error("Error: %s", e)
            logger.debug("Traceback:", exc_info=True)
            return sent_stats, sent_details, []
        
        logger.info("Step 3: Filtering candidates (threshold >= %s)...", self.config.agent_threshold)
        agent_reports = []
        candidates = [d for d in sent_details if d.get('score', 0) >= self.config.agent_threshold]
        candidates.sort(key=lambda x: x.get('score', 0), reverse=True)

        logger.debug("Total pairs: %d", len(sent_details))
        logger.debug("Candidates above threshold: %d", len(candidates))
        if candidates:
            logger.debug("Top candidate score: %.3f", candidates[0].get('score', 0))
        else:
            logger.info("No pairs meet the threshold of %s", self.config.agent_threshold)
            return sent_stats, sent_details, []
        
        # 处理缓存
        output_dir = self.config.output_dir
        cache_path = output_dir / "agent_cache.json"
        cache = {}
        if cache_path.exists():
            try:
                cache = json.loads(cache_path.read_text(encoding='utf-8'))
                logger.info("Loaded %d cached reports", len(cache))
            except Exception as e:
                logger.warning("Failed to load cache: %s", e)
                cache = {}
        
        # 分离已缓存和新的候选
        new_candidates = []
        for d in candidates:
            key = f"{d['pair'][0]}__{d['pair'][1]}"
            if key in cache:
                agent_reports.append({'pair': d['pair'], 'report': cache[key]})
                logger.debug("Using cached report for %s vs %s", d['pair'][0], d['pair'][1])
            else:
                new_candidates.append(d)

        logger.debug("Cached reports: %d", len(agent_reports))
        logger.debug("New candidates to analyze: %d", len(new_candidates))
        
        # 确定需要生成的报告数量
        max_reports = self.config.agent_max_reports if self.config.agent_max_reports > 0 else len(new_candidates)
        limit = min(max_reports - len(agent_reports), len(new_candidates))
        batch = new_candidates[:limit]
        
        logger.info("Will generate %d new reports (max: %d)", len(batch), max_reports)
        
        if batch:
            logger.info("Step 4: Generating AI analysis reports...")
            # 读取文本
            texts = {}
            for d in batch:
                a, b = d['pair'][0], d['pair'][1]
                if a not in texts:
                    texts[a] = self._read_full_text(a)
                    logger.debug("Loaded text for %s (%d chars)", a, len(texts[a]))
                if b not in texts:
                    texts[b] = self._read_full_text(b)
                    logger.debug("Loaded text for %s (%d chars)", b, len(texts[b]))
            
            # 批量生成报告
            logger.info("Calling AI API for batch analysis...")
            success_count = 0
            try:
                batched = generate_agent_report_batch(agent, batch, texts, dual_phase=self.config.agent_dual_phase)
                for item in batched:
                    agent_reports.append(item)
                    key = f"{item['pair'][0]}__{item['pair'][1]}"
                    cache[key] = item['report']
                    success_count += 1
                    logger.info("Report %d/%d: %s vs %s", success_count, len(batch), item['pair'][0], item['pair'][1])
            except Exception as batch_error:
                logger.warning("Batch processing failed: %s", batch_error)
                logger.info("Falling back to individual report generation...")

                # 逐个生成报告
                for idx, detail in enumerate(batch, 1):
                    try:
                        logger.debug("Generating report %d/%d...", idx, len(batch))
                        a = self._read_full_text(detail['pair'][0])
                        b = self._read_full_text(detail['pair'][1])
                        report = generate_agent_report(agent, detail, a, b, dual_phase=self.config.agent_dual_phase)
                        agent_reports.append({'pair': detail['pair'], 'report': report})
                        key = f"{detail['pair'][0]}__{detail['pair'][1]}"
                        cache[key] = report
                        success_count += 1
                        logger.info("Success: %s vs %s", detail['pair'][0], detail['pair'][1])
                    except Exception as e:
                        logger.error("Failed: %s vs %s", detail['pair'][0], detail['pair'][1])
                        logger.error("Error: %s", e)
                        continue
            
            logger.info("Successfully generated %d/%d reports", success_count, len(batch))

            # 保存缓存
            try:
                cache_path.write_text(json.dumps(cache, ensure_ascii=False, indent=2), encoding='utf-8')
                logger.info("Cache saved to %s", cache_path)
            except Exception as e:
                logger.warning("Failed to save cache: %s", e)

        logger.info("Agent analysis complete: %d total reports available", len(agent_reports))
        return sent_stats, sent_details, agent_reports

    def run_with_citation_analysis(self):
        """
        运行检测 + LLM 引用评定。

        Returns:
            (sent_stats, sent_details, citation_assessments)
            citation_assessments: {(i, j): CitationAssessment}
        """
        sent_stats, sent_details = self.run()

        if not self.config.enable_citation_llm:
            return sent_stats, sent_details, {}

        try:
            from .citation_analyzer import CitationAnalyzer
        except ImportError:
            logger.warning("CitationAnalyzer not available, skipping LLM citation analysis")
            return sent_stats, sent_details, {}

        logger.info("Running LLM citation analysis...")
        analyzer = CitationAnalyzer(self.config.api_config_path)

        # 加载 rows 以构建上下文
        rows = load_corpus(self.config.submissions_dir)

        all_assessments: dict = {}
        for detail in sent_details:
            hits = detail.get("hits", [])
            if not hits:
                continue
            batch_result = analyzer.assess_batch(
                hits, rows,
                max_items=self.config.citation_llm_max,
            )
            all_assessments.update(batch_result)

            # 用 LLM 返回的 adjusted_penalty 替换规则 penalty
            for hit in hits:
                key = (hit.get("i", 0), hit.get("j", 0))
                if key in batch_result:
                    assess = batch_result[key]
                    old_penalty = hit.get("citation_penalty", 1.0)
                    hit["citation_penalty_llm"] = old_penalty
                    hit["citation_penalty"] = assess.adjusted_penalty
                    hit["adjusted_sim"] = hit["sim"] * assess.adjusted_penalty
                    hit["citation_label"] = (
                        "规范引用" if assess.is_properly_cited
                        else "公共知识" if assess.is_common_knowledge
                        else f"LLM:{assess.paraphrase_level}"
                    )
                    hit["citation_explanation"] = assess.explanation

        logger.info("Citation analysis complete: %d hits assessed", len(all_assessments))
        return sent_stats, sent_details, all_assessments

    def _read_full_text(self, student_id: str) -> str:
        """读取指定学生的完整文本内容。"""
        from .corpus import iter_documents
        for sid, doc_path in iter_documents(self.config.submissions_dir):
            if sid == student_id:
                return doc_path.read_text(encoding='utf-8', errors='ignore')
        return ""
