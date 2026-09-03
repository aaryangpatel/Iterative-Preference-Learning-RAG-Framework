"""Core data models for documents and retrieval results."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field


class Document(BaseModel):
    """A corpus document compatible with CRUCIBLE/RAGTIME JSONL format.

    Parameters
    ----------
    doc_id : str
        Unique document identifier (``id`` in CRUCIBLE collections).
    text : str
        Main document body used for retrieval and generation.
    title : str | None
        Optional title prepended to text during indexing when present.
    metadata : dict[str, Any]
        Arbitrary metadata (URL, provenance, language, etc.).
    """

    model_config = ConfigDict(extra="allow", populate_by_name=True)

    doc_id: str = Field(validation_alias="id")
    text: str
    title: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    @property
    def id(self) -> str:
        """Alias for ``doc_id`` to match CRUCIBLE naming."""
        return self.doc_id

    def searchable_text(self) -> str:
        """Return text used for lexical retrieval (title + body)."""
        if self.title:
            return f"{self.title} {self.text}"
        return self.text

    @classmethod
    def from_jsonl_record(cls, record: dict[str, Any]) -> Document:
        """Build a ``Document`` from a parsed JSONL record.

        Parameters
        ----------
        record : dict[str, Any]
            Parsed JSON object from a collection JSONL line.

        Returns
        -------
        Document
            Normalized document with ``id`` mapped to ``doc_id``.
        """
        doc_id = record.get("id") or record.get("doc_id")
        if doc_id is None:
            raise ValueError("JSONL record must contain 'id' or 'doc_id'")

        metadata = dict(record.get("metadata") or {})
        for key in ("url", "created", "source", "lang"):
            if key in record and key not in metadata:
                metadata[key] = record[key]

        return cls(
            doc_id=str(doc_id),
            text=record["text"],
            title=record.get("title"),
            metadata=metadata,
        )


class RankedDocument(BaseModel):
    """A document with retrieval rank and score.

    Parameters
    ----------
    document : Document
        The retrieved document.
    rank : int
        1-based rank in the result list (1 = best).
    score : float
        Retrieval or reranking score (higher = more relevant).
    retrieval_score : float | None
        Original first-stage retrieval score before reranking.
    metadata : dict[str, Any]
        Stage-specific metadata (e.g. reranker name, matched terms).
    """

    document: Document
    rank: int
    score: float
    retrieval_score: float | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """Structured output of a retrieval pipeline run for one query.

    Parameters
    ----------
    query_id : str
        Identifier for the query (topic id in TREC-style benchmarks).
    query_text : str
        The query string that was searched.
    documents : list[RankedDocument]
        Ranked retrieved documents.
    run_id : str | None
        Optional run identifier for experiment tracking.
    collection : str | None
        Name or path of the indexed collection.
    metadata : dict[str, Any]
        Pipeline metadata (retriever name, reranker used, top_k, etc.).
    """

    query_id: str
    query_text: str
    documents: list[RankedDocument] = Field(default_factory=list)
    run_id: str | None = None
    collection: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)

    def doc_ids(self) -> list[str]:
        """Return document IDs in rank order."""
        return [ranked.document.doc_id for ranked in self.documents]

    def top_document(self) -> RankedDocument | None:
        """Return the highest-ranked document, if any."""
        if not self.documents:
            return None
        return self.documents[0]
