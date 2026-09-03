#!/usr/bin/env python3
"""Extract gold nugget questions from RAGTIME score release human judgments."""

from __future__ import annotations

import argparse
import ast
import csv
import json
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_JUDGMENTS_DIR = ROOT / "data" / "benchmark" / "ragtime25" / "almost-human.short-topics"
DEFAULT_OUTPUT = ROOT / "data" / "benchmark" / "ragtime25" / "gold_nuggets" / "gold_nuggets.jsonl"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Convert RAGTIME human nugget judgments to gold_nuggets.jsonl")
    parser.add_argument(
        "--judgments-dir",
        type=Path,
        default=DEFAULT_JUDGMENTS_DIR,
        help="Directory containing *.judgments.tsv files",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output gold_nuggets.jsonl path",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    questions_by_topic = extract_gold_questions(args.judgments_dir)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as handle:
        for topic_id in sorted(questions_by_topic, key=lambda value: int(value)):
            questions = sorted(questions_by_topic[topic_id])
            payload = {
                "topic_id": topic_id,
                "questions": [
                    {"question_id": f"{topic_id}-gold-{index}", "text": text}
                    for index, text in enumerate(questions)
                ],
            }
            handle.write(json.dumps(payload, ensure_ascii=False) + "\n")
    print(f"Wrote {len(questions_by_topic)} topics → {args.output}")
    total = sum(len(questions) for questions in questions_by_topic.values())
    print(f"Total gold questions: {total}")


def extract_gold_questions(judgments_dir: Path) -> dict[str, set[str]]:
    """Parse human nugget_mentioned rows and collect unique questions per topic."""
    questions_by_topic: dict[str, set[str]] = defaultdict(set)
    for path in sorted(judgments_dir.glob("*.judgments.tsv")):
        with path.open(encoding="utf-8") as handle:
            reader = csv.DictReader(handle, delimiter="\t")
            for row in reader:
                if row.get("judge") != "human":
                    continue
                if row.get("annotation_job") != "nugget_mentioned":
                    continue
                topic_id = str(row.get("topic_id", "")).strip()
                annotation = row.get("annotation", "")
                if not topic_id or not annotation:
                    continue
                for question in _parse_annotation_questions(annotation):
                    cleaned = question.strip()
                    if cleaned and cleaned.lower() != "other":
                        questions_by_topic[topic_id].add(cleaned)
    return questions_by_topic


def _parse_annotation_questions(annotation: str) -> list[str]:
    try:
        payload = ast.literal_eval(annotation)
    except (SyntaxError, ValueError):
        return []
    questions: list[str] = []
    if isinstance(payload, list):
        for item in payload:
            if isinstance(item, (list, tuple)) and item:
                questions.append(str(item[0]))
            elif isinstance(item, str):
                questions.append(item)
    return questions


if __name__ == "__main__":
    main()
