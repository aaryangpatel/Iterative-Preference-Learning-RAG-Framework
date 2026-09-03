"""TREC RAGTIME search service client (remote retrieval, no local corpus)."""

from __future__ import annotations

import json
import ssl
import time
import urllib.error
import urllib.request
from dataclasses import dataclass

import certifi

from rag_framework.config.ragtime import RagtimeApiConfig, get_ragtime_api_config
from rag_framework.models.document import Document


class RagtimeApiError(RuntimeError):
    """Raised when the RAGTIME API is misconfigured or returns an error."""


@dataclass
class RagtimeSearchHit:
    """One document hit from the RAGTIME search API.

    Parameters
    ----------
    doc_id : str
        Collection document identifier.
    rank : int
        One-based rank in the result list.
    score : float
        Retrieval score when provided by the API.
    title : str | None
        Document title if available in the payload.
    snippet : str | None
        Short text snippet if returned without full content fetch.
    """

    doc_id: str
    rank: int
    score: float = 0.0
    title: str | None = None
    snippet: str | None = None


class RagtimeSearchClient:
    """Search and fetch documents via the TREC RAGTIME hosted search service.

    See https://trec-ragtime.github.io/search_api.html for registration and usage.
    When ``config.dry_run`` is True, no HTTP requests are made.

    Parameters
    ----------
    config : RagtimeApiConfig | None
        API settings. Loads from environment when omitted.
    """

    def __init__(self, config: RagtimeApiConfig | None = None) -> None:
        self._config = config or get_ragtime_api_config()

    @property
    def config(self) -> RagtimeApiConfig:
        return self._config

    def _require_live_config(self) -> None:
        if self._config.dry_run:
            raise RagtimeApiError(
                "RAGTIME API dry-run is enabled (RAGTIME_DRY_RUN or BENCHMARK_DRY_RUN). "
                "Disable dry-run before calling the live search service."
            )
        if not self._config.is_configured:
            raise RagtimeApiError(
                "RAGTIME API is not configured. Set RAGTIME_API_URL and RAGTIME_BEARER_TOKEN "
                "in .env after registering at https://trec.nist.gov/"
            )

    def search(self, query_text: str, max_results: int = 25) -> list[RagtimeSearchHit]:
        """Retrieve ranked document ids for a query string.

        Parameters
        ----------
        query_text : str
            Search query (typically ``Query.retrieval_text()``).
        max_results : int
            Maximum documents to return.

        Returns
        -------
        list[RagtimeSearchHit]
            Ranked hits with document ids.

        Raises
        ------
        RagtimeApiError
            On misconfiguration, dry-run, or HTTP failure.
        """
        self._require_live_config()
        payload = {
            "pipeline": self._config.pipeline,
            "query": query_text,
        }
        response = self._post_json(self._config.search_endpoint, payload)
        return self._parse_search_response(response, max_results)

    def fetch_document(self, doc_id: str) -> dict:
        """Fetch full document content for one collection id.

        Parameters
        ----------
        doc_id : str
            RAGTIME document uuid.

        Returns
        -------
        dict
            Parsed JSON document payload from the ``/content`` endpoint.
        """
        self._require_live_config()
        payload = {
            "collection": self._config.collection,
            "id": doc_id,
        }
        return self._post_json(self._config.content_endpoint, payload)

    def fetch_documents(
        self,
        query_text: str,
        max_results: int = 25,
        fetch_full_text: bool = True,
    ) -> list[Document]:
        """Search then optionally hydrate hits into ``Document`` records.

        Parameters
        ----------
        query_text : str
            Search query text.
        max_results : int
            Maximum documents to return.
        fetch_full_text : bool
            When True, call ``/content`` for each hit to populate document text.

        Returns
        -------
        list[Document]
            RAG-ready documents with ``doc_id``, ``text``, and metadata.
        """
        hits = self.search(query_text, max_results=max_results)
        documents: list[Document] = []
        for hit in hits:
            text = hit.snippet or ""
            title = hit.title
            metadata = {
                "rank": hit.rank,
                "score": hit.score,
                "source": "ragtime_api",
                "collection": self._config.collection,
            }
            if fetch_full_text:
                content_payload = self.fetch_document(hit.doc_id)
                text, title = self._extract_document_fields(content_payload, fallback_text=text)
                metadata["raw_content_keys"] = sorted(content_payload.keys())
            documents.append(
                Document(
                    doc_id=hit.doc_id,
                    text=text,
                    title=title,
                    metadata=metadata,
                )
            )
        return documents

    def _post_json(self, url: str, payload: dict, max_attempts: int = 6) -> dict | list:
        body = json.dumps(payload).encode("utf-8")
        token = self._config.bearer_token.strip()
        if not token.lower().startswith("bearer "):
            token = f"Bearer {token}"
        headers = {
            "Authorization": token,
            "Content-Type": "application/json",
            "Accept": "application/json",
            "User-Agent": self._config.user_agent,
        }
        last_error: Exception | None = None
        for attempt in range(max_attempts):
            request = urllib.request.Request(url, data=body, method="POST", headers=headers)
            try:
                context = ssl.create_default_context(cafile=certifi.where())
                with urllib.request.urlopen(
                    request, timeout=self._config.timeout_seconds, context=context
                ) as handle:
                    raw = handle.read().decode("utf-8")
                return json.loads(raw)
            except urllib.error.HTTPError as error:
                detail = error.read().decode("utf-8", errors="replace")
                if error.code in {500, 502, 503, 504} and attempt + 1 < max_attempts:
                    last_error = RagtimeApiError(f"RAGTIME API HTTP {error.code}: {detail}")
                    wait_seconds = 2**attempt
                    try:
                        err_payload = json.loads(detail)
                        retry_after = err_payload.get("retry_after")
                        if retry_after is not None:
                            wait_seconds = max(wait_seconds, int(retry_after))
                    except (json.JSONDecodeError, TypeError, ValueError):
                        pass
                    if error.code == 504:
                        wait_seconds = max(wait_seconds, 120)
                    time.sleep(wait_seconds)
                    continue
                raise RagtimeApiError(f"RAGTIME API HTTP {error.code}: {detail}") from error
            except (urllib.error.URLError, TimeoutError) as error:
                if attempt + 1 < max_attempts:
                    last_error = RagtimeApiError(f"RAGTIME API connection failed: {error}")
                    time.sleep(2**attempt)
                    continue
                raise RagtimeApiError(f"RAGTIME API connection failed: {error}") from error
        if last_error is not None:
            raise last_error
        raise RagtimeApiError("RAGTIME API request failed without a captured error")

    @staticmethod
    def _parse_search_response(response: dict | list, max_results: int) -> list[RagtimeSearchHit]:
        if isinstance(response, list):
            items = response
        elif isinstance(response, dict):
            scores = response.get("scores")
            if isinstance(scores, dict) and scores:
                ranked = sorted(scores.items(), key=lambda item: item[1], reverse=True)
                return [
                    RagtimeSearchHit(doc_id=str(doc_id), rank=index, score=float(score))
                    for index, (doc_id, score) in enumerate(ranked[:max_results], start=1)
                ]
            items = response.get("results") or response.get("documents") or response.get("hits") or []
        else:
            items = []

        hits: list[RagtimeSearchHit] = []
        for index, item in enumerate(items[:max_results], start=1):
            if isinstance(item, str):
                hits.append(RagtimeSearchHit(doc_id=item, rank=index))
                continue
            doc_id = str(
                item.get("id")
                or item.get("docid")
                or item.get("doc_id")
                or item.get("document_id")
                or ""
            )
            if not doc_id:
                continue
            hits.append(
                RagtimeSearchHit(
                    doc_id=doc_id,
                    rank=int(item.get("rank", index)),
                    score=float(item.get("score", 0.0)),
                    title=item.get("title"),
                    snippet=item.get("text") or item.get("snippet") or item.get("doc"),
                )
            )
        return hits

    @staticmethod
    def _extract_document_fields(payload: dict, fallback_text: str = "") -> tuple[str, str | None]:
        if "doc" in payload and isinstance(payload["doc"], dict):
            nested = payload["doc"]
            text = nested.get("text") or nested.get("body") or nested.get("contents") or fallback_text
            title = nested.get("title")
            return str(text), title
        text = payload.get("text") or payload.get("body") or payload.get("contents") or fallback_text
        title = payload.get("title")
        return str(text), title
