#!/usr/bin/env python3
"""Run the iterative preference-learning research loop for one query."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from dotenv import load_dotenv

from crucible.loaders import load_ragtime_requests
from research.config import ExperimentConfig
from research.loop import PreferenceLearningLoop
from rag_framework.models.query import Query

load_dotenv()

ROOT = Path(__file__).resolve().parent.parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Iterative CRUCIBLE + PrefNugget preference-learning loop")
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "experiments" / "preference_loop.yml",
        help="Experiment YAML config path",
    )
    parser.add_argument(
        "--query-file",
        type=Path,
        default=None,
        help="JSONL file with one RAGTIME-style request (uses first line)",
    )
    parser.add_argument("--query-id", type=str, default=None, help="Query id when using --query-file")
    parser.add_argument("--title", type=str, default=None, help="Query title (inline query mode)")
    parser.add_argument("--problem", type=str, default=None, help="Problem statement (inline query mode)")
    parser.add_argument("--background", type=str, default=None, help="Query background (inline query mode)")
    return parser.parse_args()


def load_query(args: argparse.Namespace) -> tuple[Query, dict]:
    if args.query_file is not None:
        records = list(load_ragtime_requests(args.query_file))
        if args.query_id:
            record = next((record for record in records if str(record.get("request_id")) == args.query_id), records[0])
        else:
            record = records[0]
        return Query.from_report_request(record), record

    if args.title and args.problem:
        query_id = args.query_id or "custom-query"
        record = {
            "request_id": query_id,
            "title": args.title,
            "background": args.background or "",
            "problem_statement": args.problem,
        }
        return Query.from_report_request(record), record

    default_request = next(
        load_ragtime_requests(ROOT / "data" / "crucible" / "requests" / "ragtime_request_1059.jsonl")
    )
    return Query.from_report_request(default_request), default_request


def main() -> None:
    args = parse_args()
    config = ExperimentConfig.from_yaml(args.config)
    query, request = load_query(args)

    print("Preference-Learning Research Loop")
    print("=" * 60)
    print(f"Experiment: {config.experiment_id}")
    print(f"Query: {query.title} ({query.query_id})")
    print(f"Document source: {config.document_source.provider}")
    print(f"Max rounds: {config.iteration.max_rounds}")

    loop = PreferenceLearningLoop(config)
    result = loop.run(query, request=request)

    print("\n" + "=" * 60)
    print(f"Completed in {result.total_rounds} rounds (converged={result.converged})")
    print(f"Final champion: {result.final_champion.report.run_id}")
    print(f"Sentences: {len(result.final_champion.report.sentences)}")
    print(f"Report preview:\n{result.final_champion.report_text()[:500]}...")
    print(f"\nArtifacts: {config.experiment_output_dir()}")
    print(json.dumps(result.metadata, indent=2))


if __name__ == "__main__":
    main()
