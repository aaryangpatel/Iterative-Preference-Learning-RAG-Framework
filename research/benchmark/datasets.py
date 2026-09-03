"""Load TREC RAGTIME benchmark metadata (topics, qrels, gold nuggets)."""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from crucible.loaders import load_ragtime_requests
from prefnugget.models.nugget import NuggetQuestion, NuggetQuestionBank
from rag_framework.models.query import Query

from research.adapters import query_to_topic
from research.benchmark.config import BenchmarkConfig


@dataclass
class RagtimeTopicRecord:
    """One RAGTIME report request with query and raw request dict.

    Parameters
    ----------
    query : Query
        Parsed query for pipelines.
    request : dict
        Original RAGTIME JSON record.
    """

    query: Query
    request: dict


@dataclass
class RagtimeBenchmarkData:
    """All metadata needed for Tier B benchmark (no full 22 GB corpus).

    Parameters
    ----------
    topics : list[RagtimeTopicRecord]
        Filtered evaluation topics.
    gold_nuggets_by_topic : dict[str, NuggetQuestionBank]
        Manual gold nugget banks per topic id.
    qrels_by_topic : dict[str, dict[str, int]]
        topic_id -> {doc_id: relevance_label}.
    topic_doc_ids : dict[str, list[str]]
        Top doc ids per topic from qrels (for optional local cache builds).
    """

    topics: list[RagtimeTopicRecord]
    gold_nuggets_by_topic: dict[str, NuggetQuestionBank]
    qrels_by_topic: dict[str, dict[str, int]]
    topic_doc_ids: dict[str, list[str]]


def load_benchmark_data(config: BenchmarkConfig) -> RagtimeBenchmarkData:
    """Load topics, gold nuggets, and qrels for a benchmark configuration.

    Parameters
    ----------
    config : BenchmarkConfig
        Benchmark settings with ``data_root`` paths.

    Returns
    -------
    RagtimeBenchmarkData
        Parsed metadata bundle.

    Raises
    ------
    FileNotFoundError
        When required metadata files are missing (run setup script first).
    """
    topics_path = config.resolve_data_path(config.topics_file)
    if not topics_path.exists():
        raise FileNotFoundError(
            f"Topics file not found: {topics_path}. Run: python scripts/setup_ragtime_benchmark.py"
        )

    all_topics = [_topic_record(record) for record in load_ragtime_requests(topics_path)]
    filtered = _filter_topics(all_topics, config)
    if config.max_topics is not None:
        filtered = filtered[: config.max_topics]

    gold_path = config.resolve_data_path(config.gold_nuggets_file)
    gold_nuggets = load_gold_nuggets(gold_path) if gold_path.exists() else {}

    qrels_path = config.resolve_data_path(Path("qrels/2025.mlir.qrels"))
    qrels_by_topic, topic_doc_ids = load_qrels(qrels_path) if qrels_path.exists() else ({}, {})

    return RagtimeBenchmarkData(
        topics=filtered,
        gold_nuggets_by_topic=gold_nuggets,
        qrels_by_topic=qrels_by_topic,
        topic_doc_ids=topic_doc_ids,
    )


def _topic_record(record: dict) -> RagtimeTopicRecord:
    query = Query.from_report_request(record)
    return RagtimeTopicRecord(query=query, request=record)


def _filter_topics(topics: list[RagtimeTopicRecord], config: BenchmarkConfig) -> list[RagtimeTopicRecord]:
    if config.topic_filter == "all":
        return topics
    if config.topic_filter == "short":
        return [topic for topic in topics if int(topic.request.get("limit", 2000)) == 2000]
    if config.topic_filter == "assessed":
        assessed_ids = load_assessed_topic_ids(config)
        assessed_set = set(assessed_ids)
        return [topic for topic in topics if topic.query.query_id in assessed_set]
    filter_path = Path(config.topic_filter)
    if not filter_path.is_absolute():
        candidate = config.resolve_data_path(filter_path)
        if candidate.exists():
            filter_path = candidate
    if filter_path.exists():
        ids = {line.strip() for line in filter_path.read_text(encoding="utf-8").splitlines() if line.strip()}
        return [topic for topic in topics if topic.query.query_id in ids]
    return topics


def load_assessed_topic_ids(config: BenchmarkConfig) -> list[str]:
    """Load assessed topic ids from manifest or fall back to short-topic ids."""
    if config.assessed_topics_file is not None:
        path = config.resolve_data_path(config.assessed_topics_file)
        if path.exists():
            payload = json.loads(path.read_text(encoding="utf-8"))
            if isinstance(payload, list):
                return [str(topic_id) for topic_id in payload]
            if isinstance(payload, dict) and "topic_ids" in payload:
                return [str(topic_id) for topic_id in payload["topic_ids"]]
    topics_path = config.resolve_data_path(config.topics_file)
    short_ids = []
    for record in load_ragtime_requests(topics_path):
        if int(record.get("limit", 2000)) == 2000:
            short_ids.append(str(record.get("topic_id") or record.get("request_id")))
    return short_ids


def load_gold_nuggets(path: Path) -> dict[str, NuggetQuestionBank]:
    """Load normalized gold nugget banks from JSONL.

    Each line:
    ``{"topic_id": "1001", "questions": [{"question_id": "...", "text": "..."}]}``

    Parameters
    ----------
    path : Path
        Gold nuggets JSONL path.

    Returns
    -------
    dict[str, NuggetQuestionBank]
        Banks keyed by topic id.
    """
    banks: dict[str, NuggetQuestionBank] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            topic_id = str(record["topic_id"])
            questions = [
                NuggetQuestion(
                    question_id=str(question.get("question_id") or f"{topic_id}-g{index}"),
                    text=str(question["text"]),
                    extraction_method="queryonly",
                    confidence=float(question.get("confidence", 1.0)),
                    metadata={"source": "gold", **(question.get("metadata") or {})},
                )
                for index, question in enumerate(record.get("questions", []))
            ]
            banks[topic_id] = NuggetQuestionBank(topic_id=topic_id, questions=questions)
    return banks


def load_qrels(path: Path, max_docs_per_topic: int = 100) -> tuple[dict[str, dict[str, int]], dict[str, list[str]]]:
    """Load TREC qrels and build per-topic doc id lists.

    Parameters
    ----------
    path : Path
        TREC qrels file (``query_id doc_id rel`` per line).
    max_docs_per_topic : int
        Maximum doc ids to retain per topic (sorted by relevance).

    Returns
    -------
    tuple[dict[str, dict[str, int]], dict[str, list[str]]]
        Qrel maps and top doc id lists.
    """
    qrels_by_topic: dict[str, dict[str, int]] = defaultdict(dict)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic_id, doc_id, rel = parts[0], parts[2], int(parts[3])
            qrels_by_topic[topic_id][doc_id] = rel

    topic_doc_ids: dict[str, list[str]] = {}
    for topic_id, doc_rels in qrels_by_topic.items():
        ranked = sorted(doc_rels.items(), key=lambda item: (-item[1], item[0]))
        topic_doc_ids[topic_id] = [doc_id for doc_id, _rel in ranked[:max_docs_per_topic]]
    return dict(qrels_by_topic), topic_doc_ids


def topic_to_prefnugget_topic(topic: RagtimeTopicRecord):
    """Convert a benchmark topic record to PrefNugget topic."""
    return query_to_topic(topic.query)
