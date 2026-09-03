"""Retrieval pipeline that composes retriever and optional reranker."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from rag_framework.loaders.jsonl import load_documents_from_jsonl
from rag_framework.models.document import RankedDocument, RetrievalResult
from rag_framework.models.query import Query
from rag_framework.rerankers.base import Reranker
from rag_framework.retrievers.base import Retriever, ScoredHit


@dataclass
class RetrievalPipelineConfig:
    """Configuration for a retrieval pipeline run.

    Parameters
    ----------
    top_k : int
        Final number of documents returned per query.
    retrieve_k : int | None
        First-stage pool size before reranking. Defaults to ``top_k`` when
        no reranker is used, or ``max(top_k * 4, 20)`` when a reranker is set.
    run_id : str | None
        Experiment run identifier stored in results.
    collection : str | None
        Collection name or path stored in results.
    """

    top_k: int = 10
    retrieve_k: int | None = None
    run_id: str | None = None
    collection: str | None = None


class RetrievalPipeline:
    """End-to-end retrieval: index corpus, retrieve, optionally rerank.

    This is the first building block of an iterative RAG framework. Later
    stages (nugget extraction, alignment, report generation) can consume
    ``RetrievalResult`` objects produced here.

    Parameters
    ----------
    retriever : Retriever
        First-stage retriever (e.g. ``BM25Retriever``).
    reranker : Reranker | None
        Optional second-stage reranker.
    config : RetrievalPipelineConfig | None
        Pipeline hyperparameters.
    """

    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None = None,
        config: RetrievalPipelineConfig | None = None,
    ) -> None:
        self.retriever = retriever
        self.reranker = reranker
        self.config = config or RetrievalPipelineConfig()

    def _retrieve_k(self) -> int:
        if self.config.retrieve_k is not None:
            return self.config.retrieve_k
        if self.reranker is None:
            return self.config.top_k
        return max(self.config.top_k * 4, 20)

    def index_collection(self, documents: list) -> None:
        """Build the retriever index from an in-memory document list.

        Parameters
        ----------
        documents : list[Document]
            Corpus to index.
        """
        self.retriever.index(documents)

    def index_jsonl(self, collection_path: Path) -> list:
        """Load a JSONL collection and index it.

        Parameters
        ----------
        collection_path : Path
            Path to collection JSONL.

        Returns
        -------
        list[Document]
            Loaded documents (useful for tests and inspection).
        """
        documents = load_documents_from_jsonl(collection_path)
        self.index_collection(documents)
        if self.config.collection is None:
            self.config.collection = str(collection_path)
        return documents

    def _hits_to_ranked(self, hits: list[ScoredHit]) -> list[RankedDocument]:
        ranked: list[RankedDocument] = []
        for rank, hit in enumerate(hits, start=1):
            ranked.append(
                RankedDocument(
                    document=hit.document,
                    rank=rank,
                    score=hit.score,
                    retrieval_score=hit.score,
                    metadata={"retriever": self.retriever.name},
                )
            )
        return ranked

    def retrieve(self, query: Query) -> RetrievalResult:
        """Run retrieval (and optional reranking) for a single query.

        Parameters
        ----------
        query : Query
            Query to process.

        Returns
        -------
        RetrievalResult
            Structured ranked documents with scores and metadata.
        """
        retrieve_k = self._retrieve_k()
        hits = self.retriever.retrieve(query, top_k=retrieve_k)
        candidates = self._hits_to_ranked(hits)

        if self.reranker is not None:
            documents = self.reranker.rerank(
                query=query,
                candidates=candidates,
                top_k=self.config.top_k,
            )
        else:
            documents = candidates[: self.config.top_k]

        return RetrievalResult(
            query_id=query.query_id,
            query_text=query.retrieval_text(),
            documents=documents,
            run_id=self.config.run_id,
            collection=self.config.collection,
            metadata={
                "retriever": self.retriever.name,
                "reranker": self.reranker.name if self.reranker else None,
                "top_k": self.config.top_k,
                "retrieve_k": retrieve_k,
            },
        )

    def retrieve_batch(self, queries: list[Query]) -> list[RetrievalResult]:
        """Run retrieval for multiple queries.

        Parameters
        ----------
        queries : list[Query]
            Queries to process.

        Returns
        -------
        list[RetrievalResult]
            One result per query, in input order.
        """
        return [self.retrieve(query) for query in queries]
