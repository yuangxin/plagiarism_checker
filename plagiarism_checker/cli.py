"""
命令行入口：提供参数配置并运行检测管道，支持段落检测与引用惩罚。
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .pipeline import PipelineConfig, PlagiarismPipeline


def build_parser() -> argparse.ArgumentParser:
    """
    构建命令行参数解析器。

    Returns:
        argparse.ArgumentParser 对象。
    """
    parser = argparse.ArgumentParser(
        description="学生作业抄袭检测工具"
    )
    parser.add_argument(
        "--submissions-dir",
        type=Path,
        default=Path("./paraphrase_outputs"),
        help="学生提交文件夹路径",
    )
    parser.add_argument(
        "--model-name",
        default="all-MiniLM-L6-v2",
        help="使用的Sentence Transformer模型",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="计算设备 (None=自动, 'cuda', 'cpu')",
    )
    parser.add_argument(
        "--parallel",
        action="store_true",
        help="启用CPU多线程并行",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=2,
        help="并行worker数量",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        default=5,
        help="每个句子保留的最相似句子数",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.82,
        help="相似度阈值",
    )
    parser.add_argument(
        "--max-hits",
        type=int,
        default=50,
        help="每对学生报告中保存的最大匹配数",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="报告输出目录",
    )
    parser.add_argument(
        "--enable-paragraph",
        action="store_true",
        help="启用段落级检测",
    )
    parser.add_argument(
        "--enable-citation",
        action="store_true",
        help="启用引用检测",
    )
    parser.add_argument(
        "--enable-multilingual",
        action="store_true",
        help="启用跨语言检测（中英混合）",
    )
    parser.add_argument(
        "--para-threshold",
        type=float,
        default=0.75,
        help="段落级相似度阈值",
    )
    
    # Agent 参数
    parser.add_argument(
        "--enable-agent",
        action="store_true",
        help="启用AI智能分析Agent",
    )
    parser.add_argument(
        "--agent-threshold",
        type=float,
        default=0.70,
        help="触发Agent分析的风险分数阈值",
    )
    parser.add_argument(
        "--agent-config",
        default="api_config.json",
        help="Agent API配置文件路径",
    )
    parser.add_argument(
        "--agent-dual-phase",
        action="store_true",
        help="启用双阶段分析（检察官+辩护）",
    )
    parser.add_argument(
        "--agent-max-reports",
        type=int,
        default=3,
        help="最大Agent分析报告数量（0=无限制）",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    """
    命令行主入口：解析参数、构建管道并输出报告。

    Args:
        argv: 可选的参数列表（用于测试），默认从系统参数读取。
    """
    parser = build_parser()
    args = parser.parse_args(argv)

    config = PipelineConfig(
        submissions_dir=args.submissions_dir,
        model_name=args.model_name,
        device=args.device,
        use_parallel=args.parallel,
        num_workers=args.workers,
        index_top_k=args.top_k,
        similarity_threshold=args.threshold,
        max_hits_per_pair=args.max_hits,
        output_dir=args.output_dir,
        enable_paragraph_check=args.enable_paragraph,
        enable_citation_check=args.enable_citation,
        enable_multilingual=args.enable_multilingual,
        para_threshold=args.para_threshold,
        # Agent 配置
        enable_agent=args.enable_agent,
        agent_threshold=args.agent_threshold,
        api_config_path=args.agent_config,
        agent_dual_phase=args.agent_dual_phase,
        agent_max_reports=args.agent_max_reports,
    )
    
    pipeline = PlagiarismPipeline(config)
    
    # 根据配置选择运行模式
    if config.enable_agent:
        print("运行模式：带AI智能分析")
        sent_stats, sent_details, agent_reports = pipeline.run_with_agent()
        pipeline.write_reports(sent_stats, sent_details)
        print(f"生成报告：句子级 + Agent分析（{len(agent_reports)}个报告）")
    elif config.enable_paragraph_check:
        print("运行模式：句子级 + 段落级")
        sent_stats, sent_details, para_stats, para_details = pipeline.run_with_paragraphs()
        pipeline.write_reports(sent_stats, sent_details, para_stats, para_details)
        if para_stats:
            print(f"生成报告：句子级 + 段落级")
        else:
            print(f"生成报告：仅句子级")
    else:
        print("运行模式：仅句子级")
        stats, details = pipeline.run()
        pipeline.write_reports(stats, details)
        print(f"生成报告：仅句子级")


    if __name__ == "__main__":
        main()
