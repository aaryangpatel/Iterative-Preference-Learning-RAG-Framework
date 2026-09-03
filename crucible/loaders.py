"""Load CRUCIBLE-format data files."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

from rag_framework.loaders.jsonl import load_documents_from_jsonl
from rag_framework.models.document import Document

from crucible.models.citation import CitedReport, CitedSentence, Citation, SourceSpan
from crucible.models.nugget import CrucibleNugget, CrucibleNuggetBank, NuggetAnswer


def load_collection(path: Path) -> list[Document]:
    """Load a CRUCIBLE/RAGTIME collection JSONL file.

    Parameters
    ----------
    path : Path
        Collection JSONL path.

    Returns
    -------
    list[Document]
        Parsed documents.
    """
    return load_documents_from_jsonl(path)


def save_cited_report_jsonl(report: CitedReport, path: Path) -> None:
    """Write one cited report as a JSONL line.

    Parameters
    ----------
    report : CitedReport
        Report to serialize.
    path : Path
        Output JSONL path (appended if exists).
    """
    with path.open("a", encoding="utf-8") as handle:
        handle.write(report.model_dump_json() + "\n")


def save_report_bundle(bundle: "CrucibleReportBundle", path: Path) -> None:
    """Write a full report bundle (query + citations + text) as JSON.

    Parameters
    ----------
    bundle : CrucibleReportBundle
        Complete report artifact.
    path : Path
        Output JSON path.
    """
    from crucible.models.report_output import CrucibleReportBundle

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(bundle.to_export_dict(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def load_cited_reports(path: Path) -> list[CitedReport]:
    """Load CRUCIBLE cited reports from JSONL.

    Parameters
    ----------
    path : Path
        JSONL file with one ``CitedReport`` per line.

    Returns
    -------
    list[CitedReport]
        Parsed reports.
    """
    reports: list[CitedReport] = []
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                reports.append(CitedReport.model_validate_json(stripped))
    return reports


def load_nugget_bank_v3(path: Path) -> CrucibleNuggetBank:
    """Load a CRUCIBLE v3 nugget bank JSON file.

    Parameters
    ----------
    path : Path
        Path to ``sample_nuggets_v3.json`` style file.

    Returns
    -------
    CrucibleNuggetBank
        Parsed nugget bank.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    nuggets: list[CrucibleNugget] = []
    for question, entry in payload.get("nugget_bank", {}).items():
        answers = {
            key: NuggetAnswer(
                answer=value["answer"],
                references=value.get("references", []),
            )
            for key, value in entry.get("answers", {}).items()
        }
        nuggets.append(
            CrucibleNugget(
                nugget_id=entry.get("question_id", CrucibleNugget.make_question_id(question)),
                text=question,
                question=entry.get("question", question),
                answers=answers,
                aggregator_type=entry.get("aggregator_type", "OR"),
                metadata={"creator": entry.get("creator", [])},
            )
        )
    return CrucibleNuggetBank(
        query_id=str(payload["query_id"]),
        title_query=payload.get("title_query"),
        format_version=payload.get("format_version", "v3"),
        nuggets=nuggets,
    )


def load_report_bundle(path: Path) -> "CrucibleReportBundle":
    """Load a report bundle saved by ``save_report_bundle``.

    Parameters
    ----------
    path : Path
        JSON bundle path.

    Returns
    -------
    CrucibleReportBundle
        Parsed report bundle.
    """
    from crucible.models.nugget import CrucibleNuggetBank
    from crucible.models.report_output import CrucibleReportBundle
    from rag_framework.models.document import Document
    from rag_framework.models.query import Query

    from crucible.models.citation import CitedReport

    payload = json.loads(path.read_text(encoding="utf-8"))
    nugget_raw = payload.get("nugget_bank")
    nugget_bank = CrucibleNuggetBank.model_validate(nugget_raw) if nugget_raw else None
    return CrucibleReportBundle(
        query=Query.model_validate(payload["query"]),
        report=CitedReport.model_validate(payload["report"]),
        source_documents=[Document.model_validate(document) for document in payload.get("source_documents", [])],
        nugget_bank=nugget_bank,
        metadata=payload.get("metadata", {}),
    )


def load_ragtime_requests(path: Path) -> Iterator[dict]:
    """Stream RAGTIME report requests from JSONL.

    Parameters
    ----------
    path : Path
        ``ragtime25_main_all_2k.jsonl`` style file.

    Yields
    ------
    dict
        Parsed request records.
    """
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if stripped:
                yield json.loads(stripped)


def build_cited_report_from_ragtime(
    topic_id: str,
    team_id: str,
    run_id: str,
    sentences: list[dict],
    document_lookup: dict[str, dict],
    collection_id: str | None = None,
) -> CitedReport:
    """Build a ``CitedReport`` from RAGTIME-style sentence dicts.

    Parameters
    ----------
    topic_id : str
        Topic identifier.
    team_id : str
        Team id.
    run_id : str
        Run id.
    sentences : list[dict]
        Each item has ``text`` and ``citations`` mapping doc_id -> confidence.
    document_lookup : dict[str, dict]
        doc_id -> document record with ``text``.
    collection_id : str | None
        Collection handle.

    Returns
    -------
    CitedReport
        Structured cited report.
    """
    cited_sentences: list[CitedSentence] = []
    for index, sentence in enumerate(sentences):
        citations: list[Citation] = []
        for doc_id, confidence in (sentence.get("citations") or {}).items():
            document = document_lookup.get(doc_id, {"text": ""})
            citations.append(
                Citation(
                    citation_id=f"{topic_id}-s{index}-c{doc_id}",
                    span=SourceSpan(
                        doc_id=doc_id,
                        text=document.get("text", ""),
                        metadata={"url": document.get("url")},
                    ),
                    confidence=float(confidence),
                )
            )
        cited_sentences.append(
            CitedSentence(
                sentence_id=f"{topic_id}-s{index}",
                text=sentence["text"],
                citations=citations,
                metadata=dict(sentence.get("metadata") or {}),
            )
        )
    report = CitedReport(
        report_id=f"{run_id}-{topic_id}",
        topic_id=topic_id,
        team_id=team_id,
        run_id=run_id,
        sentences=cited_sentences,
        collection_id=collection_id,
    )
    report.autofill_references()
    return report
