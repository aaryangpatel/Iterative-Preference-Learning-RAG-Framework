"""Lexical overlap reranker (lightweight, no neural model)."""

from __future__ import annotations

import re

from rag_framework.models.document import RankedDocument
from rag_framework.models.query import Query
from rag_framework.rerankers.base import Reranker

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def _token_set(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall(text.lower()))


class LexicalOverlapReranker(Reranker):
    """Rerank by query–document token overlap with retrieval-score tie-break.

    The rerank score is ``overlap_count + 0.001 * retrieval_score`` so that
    BM25 ordering is preserved when overlap is equal. This provides a cheap
    second stage suitable for research prototyping without GPU dependencies.

    Parameters
    ----------
    boost_title : bool
        If True, title tokens are counted twice in overlap scoring.
    """

    def __init__(self, boost_title: bool = True) -> None:
        self._boost_title = boost_title

    @property
    def name(self) -> str:
        return "lexical_overlap"

    def rerank(
        self,
        query: Query,
        candidates: list[RankedDocument],
        top_k: int,
    ) -> list[RankedDocument]:
        """Score candidates by token overlap and return reranked top-k.

        Parameters
        ----------
        query : Query
            Query used for overlap computation.
        candidates : list[RankedDocument]
            First-stage results.
        top_k : int
            Output list size.

        Returns
        -------
        list[RankedDocument]
            Reranked documents with updated scores and ranks.
        """
        query_tokens = _token_set(query.retrieval_text())
        scored: list[tuple[float, RankedDocument]] = []

        for candidate in candidates:
            doc_tokens = _token_set(candidate.document.searchable_text())
            if self._boost_title and candidate.document.title:
                doc_tokens |= _token_set(candidate.document.title)

            overlap = len(query_tokens & doc_tokens)
            retrieval_score = candidate.retrieval_score or candidate.score
            rerank_score = float(overlap) + 0.001 * retrieval_score

            scored.append(
                (
                    rerank_score,
                    RankedDocument(
                        document=candidate.document,
                        rank=candidate.rank,
                        score=rerank_score,
                        retrieval_score=retrieval_score,
                        metadata={
                            **candidate.metadata,
                            "overlap_count": overlap,
                            "reranker": self.name,
                        },
                    ),
                )
            )

        scored.sort(key=lambda item: item[0], reverse=True)
        reranked: list[RankedDocument] = []
        for rank, (_, candidate) in enumerate(scored[:top_k], start=1):
            reranked.append(candidate.model_copy(update={"rank": rank}))
        return reranked
