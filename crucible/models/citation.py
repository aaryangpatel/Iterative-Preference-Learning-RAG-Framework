"""CRUCIBLE-style citation tracking for generated report sentences."""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


class SourceSpan(BaseModel):
    """Passage slice from a retrieved corpus document.

    Parameters
    ----------
    doc_id : str
        Corpus document identifier (RAGTIME UUID_offset format).
    text : str
        Verbatim supporting passage text.
    start_char : int | None
        Start offset in parent document text.
    end_char : int | None
        End offset in parent document text.
    metadata : dict[str, Any]
        Retrieval rank, URL, chunker name, etc.
    """

    doc_id: str
    text: str
    start_char: int | None = None
    end_char: int | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class Citation(BaseModel):
    """Link from a generated sentence to a supporting source span.

    Parameters
    ----------
    citation_id : str
        Unique citation id within a report.
    span : SourceSpan
        Supporting passage.
    confidence : float
        Support strength on a 0-100 scale (RAGTIME convention).
    support_label : Literal["supported", "partial", "unsupported"]
        Discretized support quality.
    method : str
        Assignment method (generation, align_prompt, heuristic).
    metadata : dict[str, Any]
        Prompt or evaluator metadata.
    """

    citation_id: str
    span: SourceSpan
    confidence: float = 100.0
    support_label: Literal["supported", "partial", "unsupported"] = "supported"
    method: str = "generation"
    metadata: dict[str, Any] = Field(default_factory=dict)


class CitedSentence(BaseModel):
    """One generated sentence with linked citations.

    Parameters
    ----------
    sentence_id : str
        Stable id within the report.
    text : str
        Generated sentence text.
    citations : list[Citation]
        Supporting evidence links.
    metadata : dict[str, Any]
        Generation iteration, model, etc.
    """

    sentence_id: str
    text: str
    citations: list[Citation] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)

    def doc_ids(self) -> set[str]:
        """Return all cited document ids."""
        return {citation.span.doc_id for citation in self.citations}

    def primary_citation(self) -> Citation | None:
        """Return the citation with highest confidence."""
        if not self.citations:
            return None
        return max(self.citations, key=lambda citation: citation.confidence)


class CitedReport(BaseModel):
    """CRUCIBLE report with per-sentence citation tracking.

    Parameters
    ----------
    report_id : str
        Run and topic identifier.
    topic_id : str
        Query/topic id (RAGTIME ``topic_id``).
    team_id : str
        Submitter team id.
    run_id : str
        Run identifier.
    sentences : list[CitedSentence]
        Ordered cited sentences.
    references : list[str]
        Union of cited doc ids (RAGTIME ``references`` list).
    collection_id : str | None
        Source collection handle.
    metadata : dict[str, Any]
        Task, limit, description, etc.
    """

    report_id: str
    topic_id: str
    team_id: str
    run_id: str
    sentences: list[CitedSentence] = Field(default_factory=list)
    references: list[str] = Field(default_factory=list)
    collection_id: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def all_citations(self) -> list[Citation]:
        """Flatten all citations across sentences."""
        citations: list[Citation] = []
        for sentence in self.sentences:
            citations.extend(sentence.citations)
        return citations

    def uncited_sentences(self) -> list[CitedSentence]:
        """Sentences with no citations attached."""
        return [sentence for sentence in self.sentences if not sentence.citations]

    def invalid_doc_ids(self, valid_ids: set[str]) -> set[str]:
        """Document ids cited but not present in the retrieval pool."""
        cited = {doc_id for sentence in self.sentences for doc_id in sentence.doc_ids()}
        return cited - valid_ids

    def citation_coverage(self) -> float:
        """Fraction of sentences with at least one citation."""
        if not self.sentences:
            return 0.0
        cited_count = len(self.sentences) - len(self.uncited_sentences())
        return cited_count / len(self.sentences)

    def autofill_references(self):
        """Rebuild ``references`` from sentence citations."""
        reference_set = set()
        for sentence in self.sentences:
            reference_set.update(sentence.doc_ids())
        self.references = sorted(reference_set)
