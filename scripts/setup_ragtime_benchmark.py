#!/usr/bin/env python3
"""Download small RAGTIME metadata files (no 22 GB corpus).

Downloads topics and qrels from NIST public URLs. Gold nuggets require
manual extraction of the nugget score release tarball — see README.md.
"""

from __future__ import annotations

import argparse
import json
import tarfile
import urllib.request
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DATA_ROOT = ROOT / "data" / "benchmark" / "ragtime25"

TOPICS_URL = "https://trec.nist.gov/data/ragtime/ragtime25_main_eng.jsonl"
QRELS_URL = "https://trec.nist.gov/data/ragtime/2025.mlir.qrels"
NUGGET_RELEASE_URL = "https://trec.nist.gov/data/ragtime/ragtime2025-repgen-score-release.0303.tgz"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Set up TREC RAGTIME Tier B benchmark metadata")
    parser.add_argument(
        "--data-root",
        type=Path,
        default=DEFAULT_DATA_ROOT,
        help="Benchmark metadata directory",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip HTTP downloads (only rebuild manifests from existing files)",
    )
    parser.add_argument(
        "--nugget-release-dir",
        type=Path,
        default=None,
        help="Path to extracted ragtime2025-repgen-score-release directory for gold nugget conversion",
    )
    parser.add_argument(
        "--max-docs-per-topic",
        type=int,
        default=100,
        help="Max qrel doc ids to store per topic in manifest",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_root = args.data_root
    topics_dir = data_root / "topics"
    qrels_dir = data_root / "qrels"
    manifests_dir = data_root / "manifests"
    gold_dir = data_root / "gold_nuggets"
    cache_dir = data_root / "cache"

    for directory in (topics_dir, qrels_dir, manifests_dir, gold_dir, cache_dir):
        directory.mkdir(parents=True, exist_ok=True)

    topics_path = topics_dir / "ragtime25_main_eng.jsonl"
    qrels_path = qrels_dir / "2025.mlir.qrels"
    gold_path = gold_dir / "gold_nuggets.jsonl"

    if not args.skip_download:
        print(f"Downloading topics → {topics_path}")
        _download(TOPICS_URL, topics_path)
        print(f"Downloading qrels → {qrels_path}")
        _download(QRELS_URL, qrels_path)
        print(f"\nGold nugget tarball (manual step): {NUGGET_RELEASE_URL}")
        print(f"  wget {NUGGET_RELEASE_URL}")
        print("  tar -xzf ragtime2025-repgen-score-release.0303.tgz -C data/benchmark/ragtime25/")
        print("  python scripts/setup_ragtime_benchmark.py --nugget-release-dir <extracted_dir>")

    if not topics_path.exists():
        raise FileNotFoundError(f"Missing topics file: {topics_path}")

    assessed_ids = _build_assessed_topics(topics_path, manifests_dir / "assessed_topics.json")
    print(f"Assessed/short topics: {len(assessed_ids)} ids → {manifests_dir / 'assessed_topics.json'}")

    if qrels_path.exists():
        _build_topic_doc_manifest(qrels_path, manifests_dir / "topic_doc_ids.json", args.max_docs_per_topic)
        print(f"Built topic doc manifest → {manifests_dir / 'topic_doc_ids.json'}")

    if args.nugget_release_dir is not None:
        count = _convert_nugget_release(args.nugget_release_dir, gold_path)
        print(f"Converted {count} topic gold nugget banks → {gold_path}")
    elif not gold_path.exists():
        _write_gold_nuggets_readme(gold_dir / "README.txt", NUGGET_RELEASE_URL)

    print("\nSetup complete.")
    print(f"Data root: {data_root}")
    print("Next: configure .env (RAGTIME_API_URL, RAGTIME_BEARER_TOKEN, OPENROUTER_API_KEY)")
    print("Then: python examples/run_benchmark.py --config experiments/benchmark_ragtime_poster.yml --live")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    urllib.request.urlretrieve(url, dest)


def _build_assessed_topics(topics_path: Path, out_path: Path) -> list[str]:
    short_ids: list[str] = []
    with topics_path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            if int(record.get("limit", 2000)) == 2000:
                short_ids.append(str(record.get("topic_id") or record.get("request_id")))
    out_path.write_text(json.dumps({"topic_ids": short_ids}, indent=2), encoding="utf-8")
    return short_ids


def _build_topic_doc_manifest(qrels_path: Path, out_path: Path, max_docs: int) -> None:
    by_topic: dict[str, list[tuple[str, int]]] = defaultdict(list)
    with qrels_path.open(encoding="utf-8") as handle:
        for line in handle:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            topic_id, doc_id, rel = parts[0], parts[2], int(parts[3])
            by_topic[topic_id].append((doc_id, rel))

    manifest: dict[str, list[str]] = {}
    for topic_id, pairs in by_topic.items():
        ranked = sorted(pairs, key=lambda item: (-item[1], item[0]))
        manifest[topic_id] = [doc_id for doc_id, _rel in ranked[:max_docs]]

    out_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def _convert_nugget_release(release_dir: Path, out_path: Path) -> int:
    """Best-effort conversion of RAGTIME nugget release files to normalized JSONL."""
    release_dir = release_dir.resolve()
    records: list[dict] = []

    for json_path in sorted(release_dir.rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        topic_id = str(payload.get("topic_id") or payload.get("request_id") or payload.get("query_id") or "")
        if not topic_id:
            continue
        questions_raw = payload.get("questions") or payload.get("nuggets") or payload.get("qa_pairs") or []
        questions = []
        for index, item in enumerate(questions_raw):
            if isinstance(item, str):
                questions.append({"question_id": f"{topic_id}-g{index}", "text": item})
            elif isinstance(item, dict):
                text = item.get("text") or item.get("question") or item.get("nugget")
                if text:
                    questions.append(
                        {
                            "question_id": str(item.get("question_id") or item.get("nugget_id") or f"{topic_id}-g{index}"),
                            "text": str(text),
                        }
                    )
        if questions:
            records.append({"topic_id": topic_id, "questions": questions})

    for jsonl_path in sorted(release_dir.rglob("*.jsonl")):
        with jsonl_path.open(encoding="utf-8") as handle:
            for line in handle:
                stripped = line.strip()
                if not stripped:
                    continue
                payload = json.loads(stripped)
                topic_id = str(payload.get("topic_id") or payload.get("request_id") or "")
                questions_raw = payload.get("questions") or payload.get("nuggets") or []
                questions = []
                for index, item in enumerate(questions_raw):
                    text = item if isinstance(item, str) else item.get("text") or item.get("question")
                    if text:
                        questions.append({"question_id": f"{topic_id}-g{index}", "text": str(text)})
                if topic_id and questions:
                    records.append({"topic_id": topic_id, "questions": questions})

    if not records and (release_dir / "ragtime2025-repgen-score-release.0303.tgz").exists():
        with tarfile.open(release_dir / "ragtime2025-repgen-score-release.0303.tgz") as archive:
            archive.extractall(path=release_dir / "_extracted")
        return _convert_nugget_release(release_dir / "_extracted", out_path)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    return len(records)


def _write_gold_nuggets_readme(path: Path, tarball_url: str) -> None:
    path.write_text(
        "\n".join(
            [
                "Gold nuggets not yet converted.",
                f"Download and extract: {tarball_url}",
                "Then run:",
                "  python scripts/setup_ragtime_benchmark.py \\",
                "    --nugget-release-dir data/benchmark/ragtime25/<extracted_folder>",
            ]
        ),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
