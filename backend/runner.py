"""Unified detection runner: 3-stage cascade (base → LLM citation → Agent)."""

from __future__ import annotations

import logging
import sys
from pathlib import Path

# Ensure project root is on sys.path so plagiarism_checker is importable
_project_root = str(Path(__file__).resolve().parent.parent)
_API_CONFIG = str(Path(__file__).resolve().parent.parent / "api_config.json")
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from plagiarism_checker.pipeline import PlagiarismPipeline, PipelineConfig

logger = logging.getLogger(__name__)

SENSITIVITY_MAP = {"low": 0.70, "medium": 0.82, "high": 0.90}


def run_detection(
    job_dir: str | Path,
    *,
    sensitivity: str = "medium",
    enable_ai: bool = False,
    agent_depth: str = "quick",
    enable_crosslingual: bool = False,
    detection_mode: str = "all",
    target_names: list[str] | None = None,
    progress_callback=None,
) -> dict:
    """
    Execute the unified 3-stage detection pipeline.

    Args:
        job_dir: Directory containing uploaded text files.
        sensitivity: "low" (0.70) / "medium" (0.82) / "high" (0.90).
        enable_ai: If True, run LLM citation refinement + Agent analysis.
        agent_depth: "quick" or "thorough" (dual-phase).
        enable_crosslingual: Enable cross-language detection.
        detection_mode: "all" for mutual comparison, "target" for target-vs-reference.
        target_names: List of target file stems (only used in "target" mode).
        progress_callback: Optional callable(stage: str, progress: int) for SSE.

    Returns:
        Dict with keys: sent_stats, sent_details, para_stats, para_details,
        agent_reports, auto_adjusted, original_threshold, adjusted_threshold.
    """
    job_dir = Path(job_dir)
    threshold = SENSITIVITY_MAP.get(sensitivity, 0.82)

    # ── Auto-detect if cross-lingual is needed ───────────────
    auto_crosslingual = False
    if not enable_crosslingual:
        try:
            from plagiarism_checker.corpus import load_corpus
            rows = load_corpus(job_dir)
            langs = set()
            for r in rows:
                if r.lang:
                    langs.add(r.lang)
            if len(langs) > 1:
                enable_crosslingual = True
                auto_crosslingual = True
                logger.info("Auto-enabled cross-lingual: detected languages %s", langs)
        except Exception:
            pass

    def _notify(stage: str, progress: int):
        if progress_callback:
            progress_callback(stage, progress)

    # ── Build PipelineConfig ────────────────────────────────
    cfg_kwargs = dict(
        submissions_dir=job_dir,
        similarity_threshold=threshold,
        para_threshold=max(0.60, threshold - 0.07),
        enable_paragraph_check=True,
        enable_citation_check=True,
        enable_multilingual=enable_crosslingual,
        enable_agent=False,
        enable_citation_llm=False,
        output_dir=job_dir,
    )

    if detection_mode == "target" and target_names:
        all_files = [p.stem for p in job_dir.glob("*.txt")]
        cfg_kwargs["target_stems"] = target_names
        cfg_kwargs["reference_stems"] = [f for f in all_files if f not in target_names]

    cfg = PipelineConfig(**cfg_kwargs)

    # ── Stage 1: Base detection (always) ────────────────────
    _notify("Loading model and building embeddings...", 10)
    pipeline = PlagiarismPipeline(cfg)

    _notify("Running sentence & paragraph detection...", 30)
    sent_stats, sent_details, para_stats, para_details = pipeline.run_with_paragraphs()

    auto_adjusted = False
    original_threshold = None
    adjusted_threshold = None

    # ── Auto-retry on empty results ─────────────────────────
    if not sent_stats and threshold > 0.60:
        auto_adjusted = True
        original_threshold = threshold
        adjusted_threshold = max(0.60, threshold - 0.10)
        logger.info("No results at %.2f, retrying at %.2f", threshold, adjusted_threshold)

        _notify(f"No matches at {threshold:.2f}, auto-adjusting to {adjusted_threshold:.2f}...", 35)
        cfg.similarity_threshold = adjusted_threshold
        cfg.para_threshold = max(0.55, adjusted_threshold - 0.07)
        pipeline = PlagiarismPipeline(cfg)
        sent_stats, sent_details, para_stats, para_details = pipeline.run_with_paragraphs()

    _notify("Base detection complete.", 50)

    # ── Stage 2: LLM citation refinement (optional) ─────────
    if enable_ai and sent_details:
        _notify("Running AI citation analysis...", 60)
        try:
            from plagiarism_checker.citation_analyzer import CitationAnalyzer
            from plagiarism_checker.corpus import load_corpus

            analyzer = CitationAnalyzer(_API_CONFIG)
            rows = load_corpus(job_dir)

            for detail in sent_details:
                hits = detail.get("hits", [])
                if not hits:
                    continue
                batch_result = analyzer.assess_batch(hits, rows, max_items=20)
                for hit in hits:
                    key = (hit.get("i", 0), hit.get("j", 0))
                    if key in batch_result:
                        assess = batch_result[key]
                        hit["citation_penalty_llm"] = hit.get("citation_penalty", 1.0)
                        hit["citation_penalty"] = assess.adjusted_penalty
                        hit["adjusted_sim"] = hit["sim"] * assess.adjusted_penalty
                        hit["citation_label"] = (
                            "规范引用" if assess.is_properly_cited
                            else "公共知识" if assess.is_common_knowledge
                            else f"LLM:{assess.paraphrase_level}"
                        )
                        hit["citation_explanation"] = assess.explanation
            _notify("Citation analysis complete.", 75)
        except Exception as e:
            logger.warning("LLM citation analysis failed: %s", e)
            _notify("Citation analysis skipped (API error).", 75)

    # ── Stage 3: Agent deep analysis (optional) ─────────────
    agent_reports = []
    if enable_ai and sent_details:
        _notify("Running AI deep analysis...", 80)
        try:
            from plagiarism_checker.agent import SmartPlagiarismAgent, generate_agent_report, generate_agent_report_batch

            agent_threshold = 0.40
            candidates = [d for d in sent_details if d.get("score", 0) >= agent_threshold]
            candidates.sort(key=lambda x: x.get("score", 0), reverse=True)
            max_reports = 5 if agent_depth == "quick" else 10
            candidates = candidates[:max_reports]

            if candidates:
                dual_phase = agent_depth == "thorough"
                agent = SmartPlagiarismAgent(
                    _API_CONFIG,
                    dual_phase=dual_phase,
                )

                # Load full texts
                texts = {}
                for d in candidates:
                    for name in d["pair"]:
                        if name not in texts:
                            texts[name] = pipeline._read_full_text(name)

                try:
                    batched = generate_agent_report_batch(agent, candidates, texts, dual_phase=dual_phase)
                    agent_reports = batched
                except Exception:
                    for detail in candidates:
                        a = texts.get(detail["pair"][0], "")
                        b = texts.get(detail["pair"][1], "")
                        try:
                            report = generate_agent_report(agent, detail, a, b, dual_phase=dual_phase)
                            agent_reports.append({"pair": detail["pair"], "report": report})
                        except Exception as e2:
                            logger.warning("Agent report failed for %s: %s", detail["pair"], e2)

            _notify("AI analysis complete.", 95)
        except Exception as e:
            logger.warning("Agent analysis failed: %s", e)
            _notify("AI analysis skipped (API error).", 95)

    _notify("Done.", 100)

    return {
        "sent_stats": sent_stats,
        "sent_details": sent_details,
        "para_stats": para_stats,
        "para_details": para_details,
        "agent_reports": agent_reports,
        "auto_adjusted": auto_adjusted,
        "original_threshold": original_threshold,
        "adjusted_threshold": adjusted_threshold,
        "auto_crosslingual": auto_crosslingual,
    }
