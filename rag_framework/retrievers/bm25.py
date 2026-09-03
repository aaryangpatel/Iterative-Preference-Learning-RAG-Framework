"""BM25 lexical retriever."""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from rag_framework.models.document import Document
from rag_framework.models.query import Query
from rag_framework.retrievers.base import Retriever, ScoredHit

_TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization for BM25.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Token sequence.
    """
    return _TOKEN_PATTERN.findall(text.lower())


class BM25Retriever(Retriever):
    """In-memory BM25 retriever over a document collection.

    Parameters
    ----------
    k1 : float
        BM25 term-frequency saturation parameter.
    b : float
        BM25 length-normalization parameter.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self._k1 = k1
        self._b = b
        self._documents: list[Document] = []
        self._corpus_tokens: list[list[str]] = []
        self._bm25: BM25Okapi | None = None

    @property
    def name(self) -> str:
        return "bm25"

    @property
    def is_indexed(self) -> bool:
        return self._bm25 is not None

    def index(self, documents: list[Document]) -> None:
        """Tokenize documents and build a BM25 index.

        Parameters
        ----------
        documents : list[Document]
            Corpus documents. Empty list clears the index.
        """
        self._documents = list(documents)
        self._corpus_tokens = [tokenize(doc.searchable_text()) for doc in self._documents]
        if self._documents:
            self._bm25 = BM25Okapi(self._corpus_tokens, k1=self._k1, b=self._b)
        else:
            self._bm25 = None

    def retrieve(self, query: Query, top_k: int) -> list[ScoredHit]:
        """Score all indexed documents and return the top-k hits.

        Parameters
        ----------
        query : Query
            Query to search.
        top_k : int
            Number of results to return.

        Returns
        -------
        list[ScoredHit]
            Top hits by BM25 score.

        Raises
        ------
        RuntimeError
            If ``index`` has not been called.
        """
        if self._bm25 is None:
            raise RuntimeError("BM25Retriever.index() must be called before retrieve()")

        query_tokens = tokenize(query.retrieval_text())
        if not query_tokens:
            return []

        scores = self._bm25.get_scores(query_tokens)
        ranked_indices = sorted(
            range(len(scores)),
            key=lambda index: scores[index],
            reverse=True,
        )[:top_k]

        hits: list[ScoredHit] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score <= 0.0:
                continue
            hits.append(
                ScoredHit(
                    document=self._documents[index],
                    score=score,
                )
            )
        return hits
