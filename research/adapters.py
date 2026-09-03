"""Adapters bridging CRUCIBLE reports and PrefNugget judging formats."""

from __future__ import annotations

import hashlib

from crucible.models.nugget import CrucibleNugget, CrucibleNuggetBank, NuggetAnswer
from crucible.models.report_output import CrucibleReportBundle
from prefnugget.models.nugget import NuggetQuestionBank
from prefnugget.models.response import RagRunRecord, ResponseSentence, RunDocument, RunMetadata
from prefnugget.models.topic import PrefNuggetTopic
from rag_framework.models.query import Query
from rag_framework.similarity import tokenize


def query_to_topic(query: Query) -> PrefNuggetTopic:
    """Convert a ``Query`` to a PrefNugget topic for judging.

    Parameters
    ----------
    query : Query
        Research query with title/background/problem fields.

    Returns
    -------
    PrefNuggetTopic
        Topic record for PrefNugget judges.
    """
    return PrefNuggetTopic(
        request_id=query.query_id,
        title=query.title or query.text,
        background=query.background or "",
        problem_statement=query.problem_statement or query.text,
    )


def bundle_to_rag_run(bundle: CrucibleReportBundle, run_id: str | None = None) -> RagRunRecord:
    """Convert a CRUCIBLE report bundle into a PrefNugget ``RagRunRecord``.

    Parameters
    ----------
    bundle : CrucibleReportBundle
        Generated report with source documents.
    run_id : str | None
        Override run id (defaults to ``bundle.report.run_id``).

    Returns
    -------
    RagRunRecord
        Format suitable for pairwise preference judging.
    """
    documents: dict[str, RunDocument] = {}
    for document in bundle.source_documents:
        citation_key = document.doc_id
        documents[citation_key] = RunDocument(
            id=citation_key,
            text=document.text,
            title=document.title,
            url=document.metadata.get("url"),
            metadata=dict(document.metadata),
        )

    responses: list[ResponseSentence] = []
    for sentence in bundle.report.sentences:
        citation_keys = [citation.span.doc_id for citation in sentence.citations]
        responses.append(
            ResponseSentence(
                text=sentence.text,
                citations=citation_keys,
                metadata=dict(sentence.metadata),
            )
        )

    resolved_run_id = run_id or bundle.report.run_id
    return RagRunRecord(
        metadata=RunMetadata(
            team_id=bundle.report.team_id,
            run_id=resolved_run_id,
            topic_id=bundle.report.topic_id,
        ),
        responses=responses,
        documents=documents,
    )


def question_bank_to_crucible_bank(
    bank: NuggetQuestionBank,
    title_query: str | None = None,
) -> CrucibleNuggetBank:
    """Convert PrefNugget question bank to CRUCIBLE nugget bank.

    Parameters
    ----------
    bank : NuggetQuestionBank
        PrefNugget atomic questions.
    title_query : str | None
        Optional query title for the bank.

    Returns
    -------
    CrucibleNuggetBank
        CRUCIBLE-format nuggets for report generation.
    """
    nuggets: list[CrucibleNugget] = []
    for question in bank.questions:
        nuggets.append(
            CrucibleNugget(
                nugget_id=question.question_id,
                text=question.text,
                question=question.text,
                answers={question.text: NuggetAnswer(answer=question.text, references=[])},
                confidence=question.confidence,
                metadata={
                    "extraction_method": question.extraction_method,
                    "source": "prefnugget",
                    **question.metadata,
                },
            )
        )
    return CrucibleNuggetBank(
        query_id=bank.topic_id,
        title_query=title_query,
        nuggets=nuggets,
        metadata={"source": "prefnugget_question_bank", **bank.metadata},
    )


def _question_fingerprint(text: str) -> set[str]:
    return set(tokenize(text))


def merge_crucible_banks(
    base: CrucibleNuggetBank,
    *others: CrucibleNuggetBank,
    similarity_threshold: float = 0.85,
) -> CrucibleNuggetBank:
    """Merge multiple CRUCIBLE nugget banks with deduplication.

    Parameters
    ----------
    base : CrucibleNuggetBank
        Primary bank (preserved first).
    *others : CrucibleNuggetBank
        Additional banks to merge in order.
    similarity_threshold : float
        Jaccard threshold for treating questions as duplicates.

    Returns
    -------
    CrucibleNuggetBank
        Deduplicated merged bank.
    """
    kept: list[CrucibleNugget] = list(base.nuggets)
    kept_fingerprints: list[set[str]] = [_question_fingerprint(n.alignment_text()) for n in kept]

    for bank in others:
        for nugget in bank.nuggets:
            tokens = _question_fingerprint(nugget.alignment_text())
            duplicate = False
            for existing in kept_fingerprints:
                jaccard = len(tokens & existing) / len(tokens | existing) if tokens | existing else 0.0
                if jaccard >= similarity_threshold:
                    duplicate = True
                    break
            if not duplicate:
                kept.append(nugget)
                kept_fingerprints.append(tokens)

    return CrucibleNuggetBank(
        query_id=base.query_id,
        title_query=base.title_query,
        nuggets=kept,
        metadata={
            "merged": True,
            "sources": list(
                dict.fromkeys(
                    [base.metadata.get("source", "base")]
                    + [bank.metadata.get("source", "other") for bank in others]
                )
            ),
        },
    )


def make_run_id(experiment_id: str, round_index: int, suffix: str) -> str:
    """Build a deterministic run id for an experiment round."""
    raw = f"{experiment_id}-r{round_index}-{suffix}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:8]
    return f"{experiment_id}-r{round_index}-{suffix}-{digest}"
