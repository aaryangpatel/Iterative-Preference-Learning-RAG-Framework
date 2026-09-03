"""Pure math text-similarity utilities (no track-specific types)."""

from __future__ import annotations

import math
import re
from collections import Counter

_TOKEN = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Lowercase alphanumeric tokenization.

    Parameters
    ----------
    text : str
        Input text.

    Returns
    -------
    list[str]
        Token list.
    """
    return _TOKEN.findall(text.lower())


def tfidf_vectors(texts: list[str]) -> list[Counter[str]]:
    """Build TF-IDF weighted token counters for a text list.

    Parameters
    ----------
    texts : list[str]
        Corpus of strings to vectorize.

    Returns
    -------
    list[Counter[str]]
        One sparse vector per input text.
    """
    tokenized = [tokenize(text) for text in texts]
    document_count = len(tokenized)
    document_frequency: Counter[str] = Counter()
    for tokens in tokenized:
        document_frequency.update(set(tokens))

    vectors: list[Counter[str]] = []
    for tokens in tokenized:
        term_frequency = Counter(tokens)
        vector: Counter[str] = Counter()
        for term, count in term_frequency.items():
            idf = math.log((1 + document_count) / (1 + document_frequency[term])) + 1.0
            vector[term] = count * idf
        vectors.append(vector)
    return vectors


def cosine_similarity(vector_a: Counter[str], vector_b: Counter[str]) -> float:
    """Cosine similarity between two sparse TF-IDF vectors.

    Parameters
    ----------
    vector_a : Counter[str]
        First vector.
    vector_b : Counter[str]
        Second vector.

    Returns
    -------
    float
        Similarity in [0, 1].
    """
    if not vector_a or not vector_b:
        return 0.0
    shared = set(vector_a) & set(vector_b)
    dot = sum(vector_a[t] * vector_b[t] for t in shared)
    norm_a = math.sqrt(sum(value * value for value in vector_a.values()))
    norm_b = math.sqrt(sum(value * value for value in vector_b.values()))
    if norm_a == 0.0 or norm_b == 0.0:
        return 0.0
    return dot / (norm_a * norm_b)


def pairwise_cosine_matrix(texts_a: list[str], texts_b: list[str]) -> list[list[float]]:
    """Compute cosine similarity matrix between two text lists.

    Parameters
    ----------
    texts_a : list[str]
        First side texts.
    texts_b : list[str]
        Second side texts.

    Returns
    -------
    list[list[float]]
        Matrix with shape (len(texts_a), len(texts_b)).
    """
    combined = texts_a + texts_b
    vectors = tfidf_vectors(combined)
    vectors_a = vectors[: len(texts_a)]
    vectors_b = vectors[len(texts_a) :]
    return [
        [cosine_similarity(left, right) for right in vectors_b]
        for left in vectors_a
    ]
