#!/usr/bin/env python3
"""Regenerate poster figures and summary from existing benchmark evaluation scores."""

from __future__ import annotations

import argparse
from pathlib import Path

from dotenv import load_dotenv

from research.benchmark.config import BenchmarkConfig
from research.benchmark.evaluator import BenchmarkEvaluator
from research.benchmark.poster_analysis import PosterAnalysis
from research.benchmark.reporting import BenchmarkReporter

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate poster assets from saved benchmark scores")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "benchmark_ragtime_poster.yml",
        help="Benchmark YAML config (must match the run that produced scores.json)",
    )
    parser.add_argument(
        "--report",
        action="store_true",
        help="Also regenerate eval/summary_table.md and .csv",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = BenchmarkConfig.from_yaml(args.config)
    evaluator = BenchmarkEvaluator(config)
    evaluation = evaluator.run_evaluate()

    if args.report:
        report_path = BenchmarkReporter(config).run_report(evaluation)
        print(f"Report: {report_path}")

    poster_path = PosterAnalysis(config).run(evaluation)
    print(f"Poster summary: {poster_path}")
    print(f"Figures: {config.benchmark_output_dir() / 'poster' / 'figures'}")


if __name__ == "__main__":
    main()
