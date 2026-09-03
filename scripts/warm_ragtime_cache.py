#!/usr/bin/env python3
"""Pre-fetch RAGTIME documents into per-topic cache (real Search API only)."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Warm RAGTIME document cache for benchmark topics")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "benchmark_ragtime_poster.yml",
        help="Benchmark YAML (topics and cache_dir come from experiment config)",
    )
    parser.add_argument(
        "--topics",
        nargs="+",
        default=None,
        help="Optional topic ids to refresh (default: all topics in config filter)",
    )
    parser.add_argument(
        "--max-topics",
        type=int,
        default=None,
        help="Override max topics to cache (default: use config value)",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-fetch even when cache file already exists",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    os.environ["RAGTIME_DRY_RUN"] = "false"
    os.environ["BENCHMARK_DRY_RUN"] = "false"

    from rag_framework.config.ragtime import clear_ragtime_api_config_cache
    from research.benchmark.config import BenchmarkConfig
    from research.benchmark.datasets import load_benchmark_data
    from research.document_source import DocumentSource

    clear_ragtime_api_config_cache()
    config = BenchmarkConfig.from_yaml(args.config)
    if args.max_topics is not None:
        config.max_topics = args.max_topics

    data = load_benchmark_data(config)
    topics = data.topics
    if args.topics:
        allowed = set(args.topics)
        topics = [topic for topic in topics if topic.query.query_id in allowed]
    source = DocumentSource(config.experiment.document_source)
    cache_dir = config.experiment.document_source.cache_dir
    print(f"Caching {len(topics)} topics → {cache_dir}")

    for topic in topics:
        topic_id = topic.query.query_id
        cache_path = Path(cache_dir) / f"{topic_id}.json" if cache_dir else None
        if cache_path and cache_path.exists() and not args.force:
            print(f"  skip {topic_id} (cache exists)")
            continue
        print(f"  fetch {topic_id}...")
        documents = source.fetch(topic.query, force_refresh=args.force)
        print(f"    {len(documents)} documents cached")


if __name__ == "__main__":
    main()
