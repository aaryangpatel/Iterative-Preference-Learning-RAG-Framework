"""Document retrieval for research experiments."""

from __future__ import annotations

import json
import time
from pathlib import Path

from crucible.loaders import load_collection
from rag_framework.config.ragtime import RagtimeApiConfig, get_ragtime_api_config
from rag_framework.models.document import Document
from rag_framework.models.query import Query
from rag_framework.retrievers.ragtime_api import RagtimeSearchClient

from research.config import DocumentSourceConfig
from research.ragtime_retrieval import (
    build_search_queries,
    filter_low_quality_documents,
    score_document_quality,
    topic_keywords,
)


class DocumentSource:
    """Fetch and cache documents for a query from configured provider.

    Parameters
    ----------
    config : DocumentSourceConfig
        Provider settings (local JSONL or RAGTIME Search API).
    """

    def __init__(self, config: DocumentSourceConfig) -> None:
        self.config = config
        self._cache: dict[str, list[Document]] = {}
        self._ragtime: RagtimeSearchClient | None = None
        if config.provider == "ragtime_api":
            api_config = get_ragtime_api_config()
            merged = RagtimeApiConfig(
                api_url=api_config.api_url,
                bearer_token=api_config.bearer_token,
                pipeline=config.ragtime_pipeline,
                collection=config.ragtime_collection,
                dry_run=api_config.dry_run,
                timeout_seconds=api_config.timeout_seconds,
                user_agent=api_config.user_agent,
            )
            self._ragtime = RagtimeSearchClient(config=merged)
        elif config.provider != "jsonl":
            raise ValueError(
                f"Unsupported document provider: {config.provider!r}. "
                "Use 'ragtime_api' for TREC RAGTIME benchmarks or 'jsonl' for a local corpus."
            )

    def fetch(self, query: Query, force_refresh: bool = False) -> list[Document]:
        """Retrieve documents for a query, using cache when available.

        Parameters
        ----------
        query : Query
            User query.
        force_refresh : bool
            When True, bypass cache and re-fetch.

        Returns
        -------
        list[Document]
            Retrieved documents.
        """
        cache_key = query.query_id
        if not force_refresh and cache_key in self._cache:
            return self._cache[cache_key]

        if self.config.provider == "jsonl":
            if self.config.collection_path is None:
                raise ValueError("document_source.collection_path required for jsonl provider")
            documents = load_collection(Path(self.config.collection_path))
        elif self.config.provider == "ragtime_api":
            documents = self._fetch_ragtime(query, force_refresh=force_refresh)
        else:
            raise ValueError(f"Unknown document provider: {self.config.provider}")

        self._cache[cache_key] = documents
        return documents

    def _fetch_ragtime(self, query: Query, force_refresh: bool = False) -> list[Document]:
        cache_path = self._ragtime_cache_path(query.query_id)
        if cache_path is not None and cache_path.exists() and not force_refresh:
            return self._load_cached_documents(cache_path)

        if self._ragtime is None:
            raise RuntimeError("RAGTIME client not initialized")
        if self._ragtime.config.dry_run:
            if cache_path is not None and cache_path.exists():
                return self._load_cached_documents(cache_path)
            raise FileNotFoundError(
                f"RAGTIME dry-run enabled and no cached documents at {cache_path}. "
                "Run warm_ragtime_cache.py with live API once or provide cache JSON."
            )

        documents = self._fetch_ragtime_multi(query)
        if not documents:
            raise RuntimeError(
                f"RAGTIME Search API returned no documents for topic {query.query_id}. "
                "Retry later or check API credentials."
            )
        self._validate_ragtime_documents(documents, context=f"topic {query.query_id}")

        if cache_path is not None:
            self._save_cached_documents(cache_path, documents)
        return documents

    def _fetch_ragtime_multi(self, query: Query) -> list[Document]:
        """Search with multiple query variants, merge, filter, and rerank."""
        if self._ragtime is None:
            raise RuntimeError("RAGTIME client not initialized")

        search_queries = build_search_queries(query)
        merged: dict[str, Document] = {}
        for query_index, search_text in enumerate(search_queries):
            if query_index > 0:
                time.sleep(1.0)
            batch = self._ragtime.fetch_documents(
                search_text,
                max_results=self.config.max_documents,
                fetch_full_text=True,
            )
            for document in batch:
                existing = merged.get(document.doc_id)
                if existing is None:
                    document.metadata["search_query_index"] = query_index
                    document.metadata["search_query"] = search_text[:120]
                    merged[document.doc_id] = document
                    continue
                if float(document.metadata.get("score", 0.0)) > float(
                    existing.metadata.get("score", 0.0)
                ):
                    document.metadata["search_query_index"] = query_index
                    document.metadata["search_query"] = search_text[:120]
                    merged[document.doc_id] = document

        documents = filter_low_quality_documents(
            list(merged.values()),
            query,
            min_keep=min(5, self.config.max_documents),
        )
        return documents[: self.config.max_documents]

    def _ragtime_cache_path(self, query_id: str) -> Path | None:
        if self.config.cache_dir is None:
            return None
        return Path(self.config.cache_dir) / f"{query_id}.json"

    @staticmethod
    def _validate_ragtime_documents(documents: list[Document], context: str) -> None:
        for document in documents:
            source = document.metadata.get("source")
            if source != "ragtime_api":
                raise ValueError(
                    f"Expected RAGTIME documents for {context}, got source={source!r} "
                    f"for doc_id={document.doc_id!r}."
                )
            if document.doc_id.startswith("ddg-"):
                raise ValueError(
                    f"Stale DuckDuckGo document id {document.doc_id!r} in {context}. "
                    "Delete cache and re-fetch from RAGTIME Search API."
                )

    @staticmethod
    def _load_cached_documents(path: Path) -> list[Document]:
        payload = json.loads(path.read_text(encoding="utf-8"))
        documents = [Document.model_validate(item) for item in payload]
        DocumentSource._validate_ragtime_documents(documents, context=str(path))
        return documents

    @staticmethod
    def _save_cached_documents(path: Path, documents: list[Document]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = [document.model_dump() for document in documents]
        path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
