"""
search.py
---------
Search pipeline for the TIL agent.
Builds text and vector indexes and exposes a hybrid search interface
that combines lexical and semantic retrieval.
"""

from __future__ import annotations

import numpy as np
from minsearch import Index, VectorSearch
from sentence_transformers import SentenceTransformer

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

EMBEDDING_MODEL_NAME = "multi-qa-distilbert-cos-v1"
DEFAULT_NUM_RESULTS = 5


# ---------------------------------------------------------------------------
# Index construction
# ---------------------------------------------------------------------------

def build_text_index(chunks: list[dict]) -> Index:
    """
    Build a lexical search index from document chunks.

    Args:
        chunks: List of chunk dicts containing 'chunk' and 'filename' fields.

    Returns:
        Fitted minsearch Index instance.
    """
    index = Index(
        text_fields=["chunk", "filename"],
        keyword_fields=[],
    )
    index.fit(chunks)
    return index


def build_vector_index(
    chunks: list[dict],
    model: SentenceTransformer | None = None,
) -> tuple[VectorSearch, SentenceTransformer]:
    """
    Build a semantic search index from document chunks.

    Encodes each chunk's text with a sentence-transformer model and fits
    a VectorSearch index over the resulting embeddings.

    Args:
        chunks: List of chunk dicts containing a 'chunk' field.
        model:  Pre-loaded SentenceTransformer. Loaded from the default
                model name if not provided.

    Returns:
        Tuple of (VectorSearch index, SentenceTransformer model).
    """
    if model is None:
        model = SentenceTransformer(EMBEDDING_MODEL_NAME)

    embeddings = np.array([model.encode(c["chunk"]) for c in chunks])
    vector_index = VectorSearch()
    vector_index.fit(embeddings, chunks)
    return vector_index, model


# ---------------------------------------------------------------------------
# Search functions
# ---------------------------------------------------------------------------

def text_search(
    query: str,
    index: Index,
    n: int = DEFAULT_NUM_RESULTS,
) -> list[dict]:
    """
    Perform lexical search over the text index.

    Args:
        query: Search string.
        index: Fitted minsearch Index.
        n:     Maximum number of results to return.

    Returns:
        List of matching chunk dicts.
    """
    return index.search(query, num_results=n)


def vector_search(
    query: str,
    vector_index: VectorSearch,
    model: SentenceTransformer,
    n: int = DEFAULT_NUM_RESULTS,
) -> list[dict]:
    """
    Perform semantic search over the vector index.

    Args:
        query:        Search string.
        vector_index: Fitted VectorSearch index.
        model:        SentenceTransformer used to encode the query.
        n:            Maximum number of results to return.

    Returns:
        List of matching chunk dicts ranked by cosine similarity.
    """
    q_vec = model.encode(query)
    return vector_index.search(q_vec, num_results=n)


def hybrid_search(
    query: str,
    text_index: Index,
    vector_index: VectorSearch,
    model: SentenceTransformer,
    n: int = DEFAULT_NUM_RESULTS,
) -> list[dict]:
    """
    Combine lexical and semantic search results with deduplication.

    Text results are listed first (higher precision), followed by
    semantic results that were not already returned by text search.

    Args:
        query:        Search string.
        text_index:   Fitted minsearch Index.
        vector_index: Fitted VectorSearch index.
        model:        SentenceTransformer for query encoding.
        n:            Number of results requested from each sub-search.

    Returns:
        Deduplicated list of chunk dicts.
    """
    text_results = text_search(query, text_index, n)
    vector_results = vector_search(query, vector_index, model, n)

    seen: set[str] = set()
    combined: list[dict] = []
    for result in text_results + vector_results:
        key = result.get("filename", "") + str(result.get("start", result.get("chunk", "")[:40]))
        if key not in seen:
            seen.add(key)
            combined.append(result)

    return combined


# ---------------------------------------------------------------------------
# SearchPipeline — convenience wrapper
# ---------------------------------------------------------------------------

class SearchPipeline:
    """
    Encapsulates the full hybrid search pipeline.

    Holds pre-built indexes and exposes a single `search` method
    compatible with the Pydantic AI tool interface.

    Usage:
        pipeline = SearchPipeline.build(chunks)
        results  = pipeline.search("how to squash commits in git")
    """

    def __init__(
        self,
        text_index: Index,
        vector_index: VectorSearch,
        model: SentenceTransformer,
    ) -> None:
        self._text_index = text_index
        self._vector_index = vector_index
        self._model = model

    @classmethod
    def build(cls, chunks: list[dict]) -> "SearchPipeline":
        """Build all indexes from a list of document chunks."""
        text_index = build_text_index(chunks)
        vector_index, model = build_vector_index(chunks)
        return cls(text_index, vector_index, model)

    def search(self, query: str, n: int = DEFAULT_NUM_RESULTS) -> list[dict]:
        """Run hybrid search and return the top-n results."""
        return hybrid_search(
            query,
            self._text_index,
            self._vector_index,
            self._model,
            n,
        )
