"""
vector_search.py - Dense semantic search via FAISS

Why: Captures meaning-based similarity. "Profitability concerns" matches
"earnings uncertainty" even without shared keywords.
"""

import numpy as np
import faiss
from openai import OpenAI

EMBEDDING_MODEL = "text-embedding-3-small"


def embed_query(query: str, client: OpenAI) -> np.ndarray:
    """Embed a single query string. Returns L2-normalised (1, 1536) array."""
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=[query])
    vec = np.array(response.data[0].embedding, dtype=np.float32).reshape(1, -1)
    faiss.normalize_L2(vec)
    return vec


def vector_search(
    query: str,
    faiss_index: faiss.Index,
    metadata: list[dict],
    client: OpenAI,
    top_k: int = 20,
) -> list[dict]:
    """
    Return top_k chunks by cosine similarity.
    Each result includes the original chunk metadata + 'vector_score'.
    """
    query_vec = embed_query(query, client)
    scores, indices = faiss_index.search(query_vec, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx == -1:
            continue
        chunk = dict(metadata[idx])
        chunk["vector_score"] = float(score)
        chunk["retrieval_source"] = "vector"
        results.append(chunk)
    return results
