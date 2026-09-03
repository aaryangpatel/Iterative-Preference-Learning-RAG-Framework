"""Load PrefNugget kiddie-format data."""

from __future__ import annotations

import json
from pathlib import Path

from prefnugget.models.response import RagRunRecord
from prefnugget.models.topic import PrefNuggetTopic


def load_topics(path: Path) -> list[PrefNuggetTopic]:
    """Load kiddie topics JSONL.

    Parameters
    ----------
    path : Path
        ``kiddie-topics.jsonl`` path.

    Returns
    -------
    list[PrefNuggetTopic]
        Parsed topics.
    """
    topics: list[PrefNuggetTopic] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                topics.append(PrefNuggetTopic.model_validate_json(stripped))
    return topics


def load_rag_runs(path: Path) -> list[RagRunRecord]:
    """Load one kiddie TREC RAG run JSONL file.

    Parameters
    ----------
    path : Path
        ``run1.jsonl`` style file.

    Returns
    -------
    list[RagRunRecord]
        One record per topic in the run file.
    """
    records: list[RagRunRecord] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                records.append(RagRunRecord.model_validate_json(stripped))
    return records


def load_truth_leaderboard(path: Path) -> dict[tuple[str, str], float]:
    """Load kiddie fake eval leaderboard (trec_eval format).

    Parameters
    ----------
    path : Path
        ``kiddie_fake.eval.ir_measures.txt`` path.

    Returns
    -------
    dict[tuple[str, str], float]
        Mapping (run_id, query_id) -> RELEVANCE value.
    """
    scores: dict[tuple[str, str], float] = {}
    with path.open(encoding="utf-8") as handle:
        header_skipped = False
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            if not header_skipped:
                header_skipped = True
                continue
            run_id, query_id, measure, value = stripped.split()
            if measure == "RELEVANCE":
                scores[(run_id, query_id)] = float(value)
    return scores
