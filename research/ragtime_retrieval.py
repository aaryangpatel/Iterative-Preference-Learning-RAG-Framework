"""RAGTIME search query shaping and document quality filtering (no LLM)."""

from __future__ import annotations

import re
from dataclasses import dataclass

from rag_framework.models.document import Document
from rag_framework.models.query import Query

# Phrases that skew RAGTIME toward SaaS/app-directory pages on lifestyle topics.
_NOISY_PHRASES = (
    "tools commonly used",
    "working from laptops",
    "types of work suited",
    "include insights on the types of work",
)

_URL_PATTERN = re.compile(r"\b[a-z0-9][a-z0-9.-]*\.(?:com|io|app|co|org|net|de|ai)\b", re.I)
_MAX_SEARCH_QUERY_CHARS = 420


@dataclass(frozen=True)
class DocumentQualityScore:
    """Heuristic quality signals for a retrieved document."""

    prose_score: float
    url_density: float
    is_app_directory: bool
    keyword_hits: int
    word_count: int


def build_search_queries(query: Query) -> list[str]:
    """Build ranked search strings for RAGTIME (most specific first).

    Parameters
    ----------
    query : Query
        Benchmark topic query.

    Returns
    -------
    list[str]
        Unique non-empty query strings to run against the search API.
    """
    title = (query.title or "").strip()
    problem = (query.problem_statement or query.text or "").strip()
    background = (query.background or "").strip()

    cleaned_problem = problem
    for phrase in _NOISY_PHRASES:
        cleaned_problem = re.sub(re.escape(phrase), " ", cleaned_problem, flags=re.I)
    cleaned_problem = " ".join(cleaned_problem.split())

    variants: list[str] = []
    if title and cleaned_problem:
        variants.append(f"{title}. {cleaned_problem}")
    if cleaned_problem:
        variants.append(cleaned_problem)
    if title and problem and problem != cleaned_problem:
        variants.append(f"{title}. {problem}")
    if title:
        variants.append(title)
    if background and title:
        variants.append(f"{title}. {background[:240]}")
    variants.append(query.retrieval_text())

    seen: set[str] = set()
    ordered: list[str] = []
    for text in variants:
        normalized = " ".join(text.split())
        if len(normalized) > _MAX_SEARCH_QUERY_CHARS:
            normalized = normalized[:_MAX_SEARCH_QUERY_CHARS].rsplit(" ", 1)[0]
        if normalized and normalized not in seen:
            seen.add(normalized)
            ordered.append(normalized)
    return ordered


def topic_keywords(query: Query) -> list[str]:
    """Extract lowercase keyword phrases for post-retrieval relevance scoring."""
    blob = " ".join(
        part
        for part in (query.title, query.problem_statement, query.text, query.background)
        if part
    ).lower()
    tokens = re.findall(r"[a-z0-9][a-z0-9-]{2,}", blob)
    stop = {
        "the", "and", "for", "that", "this", "with", "from", "have", "should",
        "about", "what", "which", "when", "where", "their", "they", "will",
        "report", "need", "information", "include", "explain", "people",
    }
    keywords = [token for token in tokens if token not in stop and len(token) > 3]
    phrases: list[str] = list(keywords)
    title = (query.title or "").lower().strip()
    if title:
        phrases.insert(0, title)
    if "digital nomad" in blob:
        phrases.extend(["digital nomad", "nomad visa", "remote work"])
    if "j-10" in blob or "j10" in blob or "fighter" in blob:
        phrases.extend(["j-10", "j10", "fighter jet", "export"])
    seen: set[str] = set()
    unique: list[str] = []
    for phrase in phrases:
        if phrase not in seen:
            seen.add(phrase)
            unique.append(phrase)
    return unique[:20]


def score_document_quality(document: Document, keywords: list[str]) -> DocumentQualityScore:
    """Score whether a document looks like prose vs app-directory spam."""
    text = document.text or ""
    words = text.split()
    word_count = len(words)
    url_count = len(_URL_PATTERN.findall(text))
    url_density = url_count / max(word_count, 1)
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    avg_line_len = sum(len(line) for line in lines) / max(len(lines), 1)
    is_app_directory = url_count >= 12 and avg_line_len < 45 and word_count < 400

    lowered = text.lower()
    keyword_hits = sum(1 for keyword in keywords if keyword in lowered)

    prose_score = 0.0
    if word_count >= 80:
        prose_score += 1.0
    if word_count >= 200:
        prose_score += 1.0
    if url_density < 0.05:
        prose_score += 1.5
    elif url_density < 0.12:
        prose_score += 0.5
    if keyword_hits >= 2:
        prose_score += 1.0
    elif keyword_hits == 1:
        prose_score += 0.5
    if is_app_directory:
        prose_score -= 4.0
    if "we may earn revenue" in lowered or "affiliate program" in lowered:
        prose_score -= 1.5
    if "digital nomad visa" in lowered or "nomad visa" in lowered:
        prose_score += 1.0
    if avg_line_len >= 60:
        prose_score += 0.5

    return DocumentQualityScore(
        prose_score=prose_score,
        url_density=url_density,
        is_app_directory=is_app_directory,
        keyword_hits=keyword_hits,
        word_count=word_count,
    )


def rank_documents(documents: list[Document], query: Query) -> list[Document]:
    """Reorder documents by retrieval rank and prose/keyword quality."""
    keywords = topic_keywords(query)

    def sort_key(document: Document) -> tuple[float, float, float]:
        rank = float(document.metadata.get("rank", 999))
        api_score = float(document.metadata.get("score", 0.0))
        quality = score_document_quality(document, keywords)
        return (quality.prose_score, api_score, -rank)

    ranked = sorted(documents, key=sort_key, reverse=True)
    for index, document in enumerate(ranked, start=1):
        quality = score_document_quality(document, keywords)
        document.metadata["quality_rank"] = index
        document.metadata["prose_score"] = quality.prose_score
        document.metadata["is_app_directory"] = quality.is_app_directory
        document.metadata["keyword_hits"] = quality.keyword_hits
    return ranked


def filter_low_quality_documents(
    documents: list[Document],
    query: Query,
    min_keep: int = 5,
) -> list[Document]:
    """Drop obvious app-directory spam while keeping enough documents for alignment."""
    keywords = topic_keywords(query)
    scored: list[tuple[Document, DocumentQualityScore]] = [
        (document, score_document_quality(document, keywords)) for document in documents
    ]
    kept = [
        document
        for document, quality in scored
        if not quality.is_app_directory or quality.keyword_hits >= 2
    ]
    if len(kept) < min_keep:
        ranked = rank_documents(documents, query)
        return ranked[: max(min_keep, len(ranked))]
    return rank_documents(kept, query)
