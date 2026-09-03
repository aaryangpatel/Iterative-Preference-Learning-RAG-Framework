#!/usr/bin/env python3
"""Generate a single benchmark system/topic pair and save to runs/."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run one benchmark system on one topic")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--system", required=True)
    parser.add_argument("--topic", required=True)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["RAGTIME_DRY_RUN"] = "false"
    os.environ["BENCHMARK_DRY_RUN"] = "false"

    from rag_framework.config.ragtime import clear_ragtime_api_config_cache
    from rag_framework.llm.config import clear_llm_config_cache
    from crucible.loaders import save_report_bundle
    from research.benchmark.baselines import build_system
    from research.benchmark.config import BenchmarkConfig
    from research.benchmark.datasets import load_benchmark_data
    from research.document_source import DocumentSource

    clear_ragtime_api_config_cache()
    clear_llm_config_cache()

    config = BenchmarkConfig.from_yaml(args.config)
    data = load_benchmark_data(config)
    topic = next(item for item in data.topics if item.query.query_id == args.topic)
    source = DocumentSource(config.experiment.document_source)
    documents = source.fetch(topic.query)
    system = build_system(args.system, config.experiment)
    print(f"Generating {args.system}/{args.topic} ({len(documents)} documents)...", flush=True)
    run_result = system.generate(topic, documents)
    run_result.system_id = args.system
    output_root = config.benchmark_output_dir() / "runs" / args.system
    output_root.mkdir(parents=True, exist_ok=True)
    path = output_root / f"{args.topic}.json"
    save_report_bundle(run_result.bundle, path)
    if run_result.experiment is not None:
        exp_path = output_root / f"{args.topic}.experiment.json"
        exp_path.write_text(run_result.experiment.model_dump_json(indent=2), encoding="utf-8")
    sentences = len(run_result.bundle.report.sentences)
    print(f"Saved {path} ({sentences} sentences)", flush=True)
    if sentences == 0:
        raise SystemExit("Report is empty — aborting")


if __name__ == "__main__":
    main()
