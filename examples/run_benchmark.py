#!/usr/bin/env python3
"""Run TREC RAGTIME Tier B benchmark: generate, evaluate, report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from rag_framework.config.ragtime import clear_ragtime_api_config_cache, get_ragtime_api_config
from rag_framework.llm.config import clear_llm_config_cache, get_llm_config
from research.benchmark.config import BenchmarkConfig
from research.benchmark.pipeline import BenchmarkPipeline

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="TREC RAGTIME Tier B benchmark pipeline")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "benchmark_ragtime_poster.yml",
        help="Benchmark YAML config",
    )
    parser.add_argument(
        "--live",
        action="store_true",
        help="Disable dry-run and allow live RAGTIME/OpenRouter API calls",
    )
    parser.add_argument(
        "--phase",
        choices=["generate", "evaluate", "report", "poster", "all"],
        default="all",
        help="Run a single phase or all configured phases",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    clear_ragtime_api_config_cache()
    clear_llm_config_cache()

    config = BenchmarkConfig.from_yaml(args.config)
    if args.live:
        import os

        config.dry_run = False
        os.environ["BENCHMARK_DRY_RUN"] = "false"
        os.environ["RAGTIME_DRY_RUN"] = "false"
        clear_ragtime_api_config_cache()
    else:
        import os

        os.environ.setdefault("BENCHMARK_DRY_RUN", "true")
        os.environ.setdefault("RAGTIME_DRY_RUN", "true")
        clear_ragtime_api_config_cache()
    if args.phase != "all":
        config.phases = [args.phase]

    print("TREC RAGTIME Benchmark")
    print("=" * 60)
    print(f"Benchmark: {config.benchmark_id}")
    print(f"Data root: {config.data_root}")
    print(f"Dry run: {config.dry_run}")
    print(f"Phases: {', '.join(config.phases)}")
    print(f"Systems: {', '.join(config.systems)}")
    print(f"Topic filter: {config.topic_filter}")
    print(f"Max topics: {config.max_topics}")

    ragtime = get_ragtime_api_config()
    llm = get_llm_config()
    print(f"\nRAGTIME API configured: {ragtime.is_configured} (dry_run={ragtime.dry_run})")
    print(f"OpenRouter model: {llm.model}")

    if args.live and not ragtime.is_configured:
        raise SystemExit(
            "Live mode requires RAGTIME_API_URL and RAGTIME_BEARER_TOKEN in .env. "
            "Register at https://trec.nist.gov/ and see README.md"
        )

    pipeline = BenchmarkPipeline(config)
    result = pipeline.run()

    print("\n" + "=" * 60)
    if result.run_plan_written:
        print(f"Dry-run plan written: {config.benchmark_output_dir() / 'run_plan.json'}")
    if result.report_path:
        print(f"Report: {result.report_path}")
    if result.poster_path:
        print(f"Poster summary: {result.poster_path}")
    if result.evaluation is not None:
        print(json.dumps(result.evaluation.metadata, indent=2))


if __name__ == "__main__":
    main()
