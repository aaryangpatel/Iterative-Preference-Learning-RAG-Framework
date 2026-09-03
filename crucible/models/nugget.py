"""CRUCIBLE-style nugget banks: atomic factual statements with doc references."""

from __future__ import annotations

import hashlib
from typing import Any

from pydantic import BaseModel, Field

from crucible.models.citation import Citation


class NuggetAnswer(BaseModel):
    """One acceptable answer variant for a CRUCIBLE nugget question.

    Parameters
    ----------
    answer : str
        Answer text.
    references : list[dict[str, str]]
        Document references as ``{"doc_id": "..."}`` entries.
    """

    answer: str
    references: list[dict[str, str]] = Field(default_factory=list)


class CrucibleNugget(BaseModel):
    """Atomic factual statement extracted from a cited report.

    CRUCIBLE stores nuggets as questions in v3 banks; here we preserve the
    declarative fact in ``text`` and optionally mirror the question form in
    ``question``.

    Parameters
    ----------
    nugget_id : str
        Stable nugget identifier.
    text : str
        Atomic factual statement.
    question : str | None
        Question form used in v3 nugget banks.
    citations : list[Citation]
        Supporting citations inherited from the parent sentence.
    source_sentence_id : str | None
        Parent ``CitedSentence.sentence_id``.
    source_sentence_text : str | None
        Parent sentence before atomization.
    answers : dict[str, NuggetAnswer]
        Optional v3-style answer variants keyed by answer text.
    aggregator_type : str
        v3 bank aggregator (typically ``OR``).
    confidence : float
        Extractor confidence.
    metadata : dict[str, Any]
        Extractor name, format version, etc.
    """

    nugget_id: str
    text: str
    question: str | None = None
    citations: list[Citation] = Field(default_factory=list)
    source_sentence_id: str | None = None
    source_sentence_text: str | None = None
    answers: dict[str, NuggetAnswer] = Field(default_factory=dict)
    aggregator_type: str = "OR"
    confidence: float = 1.0
    metadata: dict[str, Any] = Field(default_factory=dict)

    def doc_ids(self) -> set[str]:
        """Return document ids from citations and answer references."""
        ids = {citation.span.doc_id for citation in self.citations}
        for answer in self.answers.values():
            for reference in answer.references:
                doc_id = reference.get("doc_id")
                if doc_id:
                    ids.add(doc_id)
        return ids

    def alignment_text(self) -> str:
        """Text used for semantic alignment (question preferred if set)."""
        return self.question or self.text

    @staticmethod
    def make_question_id(question: str) -> str:
        """Hash a question string into a stable id (v3 bank style)."""
        return hashlib.md5(question.encode("utf-8")).hexdigest()


class CrucibleNuggetBank(BaseModel):
    """v3-format nugget bank for one query/topic.

    Parameters
    ----------
    query_id : str
        Topic/query identifier.
    title_query : str | None
        Short query title.
    format_version : str
        Bank format version (``v3``).
    nuggets : list[CrucibleNugget]
        Atomic nuggets for this query.
    metadata : dict[str, Any]
        Extraction configuration snapshot.
    """

    query_id: str
    title_query: str | None = None
    format_version: str = "v3"
    nuggets: list[CrucibleNugget] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def to_v3_dict(self) -> dict[str, Any]:
        """Serialize to CRUCIBLE ``sample_nuggets_v3.json`` layout."""
        bank: dict[str, Any] = {}
        for nugget in self.nuggets:
            question = nugget.question or nugget.text
            answers = {
                key: {
                    "answer": value.answer,
                    "references": value.references,
                }
                for key, value in nugget.answers.items()
            }
            if not answers:
                answers = {nugget.text: {"answer": nugget.text, "references": []}}
            bank[question] = {
                "question": question,
                "answers": answers,
                "question_id": nugget.nugget_id,
                "aggregator_type": nugget.aggregator_type,
                "query_id": self.query_id,
            }
        return {
            "query_id": self.query_id,
            "title_query": self.title_query,
            "format_version": self.format_version,
            "nugget_bank": bank,
        }
