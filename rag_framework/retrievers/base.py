"""Abstract retriever interface."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from rag_framework.models.document import Document
from rag_framework.models.query import Query


@dataclass(frozen=True)
class ScoredHit:
    """A single retrieval hit before rank assignment.

    Parameters
    ----------
    document : Document
        Retrieved document.
    score : float
        Relevance score from the retriever (higher = more relevant).
    """

    document: Document
    score: float


class Retriever(ABC):
    """First-stage retriever that scores documents for a query."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable retriever identifier."""

    @abstractmethod
    def index(self, documents: list[Document]) -> None:
        """Build or refresh the retrieval index from a document collection.

        Parameters
        ----------
        documents : list[Document]
            Full corpus to index.
        """

    @abstractmethod
    def retrieve(self, query: Query, top_k: int) -> list[ScoredHit]:
        """Return top-k documents for a query.

        Parameters
        ----------
        query : Query
            Query object; ``query.retrieval_text()`` is used for search.
        top_k : int
            Maximum number of hits to return.

        Returns
        -------
        list[ScoredHit]
            Hits sorted by descending score.
        """

    @property
    def is_indexed(self) -> bool:
        """Whether ``index`` has been called successfully."""
        return False
