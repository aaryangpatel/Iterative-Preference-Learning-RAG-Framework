"""Write PrefNugget workflow output artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import yaml

from prefnugget.models.judgment import RunScore
from prefnugget.models.nugget import NuggetQuestionBank
from prefnugget.pipeline.config import PrefNuggetWorkflowConfig


def write_nugget_banks_jsonl(
    banks: dict[str, NuggetQuestionBank],
    path: Path,
) -> None:
    """Write nugget banks to JSONL (one bank per line, v3-style).

    Parameters
    ----------
    banks : dict[str, NuggetQuestionBank]
        topic_id -> question bank.
    path : Path
        Output ``*.nuggets.jsonl`` path.
    """
    with path.open("w", encoding="utf-8") as handle:
        for topic_id, bank in sorted(banks.items()):
            payload = {
                "query_id": topic_id,
                "nugget_bank": {
                    question.text: {
                        "question": question.text,
                        "answers": {},
                        "question_id": question.question_id,
                        "aggregator_type": "OR",
                        "query_id": topic_id,
                    }
                    for question in bank.questions
                },
            }
            handle.write(json.dumps(payload) + "\n")


def write_eval_leaderboard(
    scores_by_topic: dict[str, list[RunScore]],
    path: Path,
    measure: str = "NUGGET_COVERAGE",
) -> None:
    """Write TREC-style eval leaderboard (``*.eval.txt``).

    Parameters
    ----------
    scores_by_topic : dict[str, list[RunScore]]
        topic_id -> graded runs.
    path : Path
        Output eval file path.
    measure : str
        Measure name (NUGGET_COVERAGE, AVG_GRADE, etc.).
    """
    rows: list[tuple[str, str, str, str]] = []
    for topic_id, scores in sorted(scores_by_topic.items()):
        for score in scores:
            if measure == "NUGGET_COVERAGE":
                value = score.nugget_coverage
            elif measure == "AVG_GRADE":
                value = score.mean_grade
            elif measure == "MAX_GRADE":
                value = float(score.max_grade)
            elif measure == "COVERED_COUNT":
                value = float(score.covered_count)
            else:
                value = score.mean_grade
            rows.append((score.run_id, topic_id, measure, f"{value:.4f}"))

    with path.open("w", encoding="utf-8") as handle:
        handle.write("run_id query_id measure value\n")
        for row in rows:
            handle.write(" ".join(row) + "\n")


def write_config_snapshot(config: PrefNuggetWorkflowConfig, path: Path) -> None:
    """Write resolved workflow config YAML.

    Parameters
    ----------
    config : PrefNuggetWorkflowConfig
        Workflow configuration.
    path : Path
        Output ``*.config.yml`` path.
    """
    payload = {
        "variant": config.variant,
        "grading_mode": config.grading_mode,
        "max_questions": config.max_questions,
        "num_pivot": config.num_pivot,
        "num_others": config.num_others,
        "allow_ties": config.allow_ties,
        "bidirectional": config.bidirectional,
    }
    path.write_text(yaml.safe_dump(payload, sort_keys=False), encoding="utf-8")
