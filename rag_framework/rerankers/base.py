"""Abstract reranker interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from rag_framework.models.document import RankedDocument
from rag_framework.models.query import Query


class Reranker(ABC):
    """Second-stage reranker that reorders first-stage retrieval hits."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable reranker identifier."""

    @abstractmethod
    def rerank(
        self,
        query: Query,
        candidates: list[RankedDocument],
        top_k: int,
    ) -> list[RankedDocument]:
        """Rerank candidate documents and return top-k.

        Parameters
        ----------
        query : Query
            Original query.
        candidates : list[RankedDocument]
            First-stage ranked documents. ``retrieval_score`` should be set.
        top_k : int
            Number of documents to return after reranking.

        Returns
        -------
        list[RankedDocument]
            Reranked documents with updated ``score`` and 1-based ``rank``.
        """
