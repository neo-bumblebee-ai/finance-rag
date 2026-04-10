"""
reranker.py - Cross-encoder reranking via Cohere Rerank API

Takes the top-40 fused candidates and scores each against the query
using a cross-encoder (query + document seen together, not independently).
Much more precise than bi-encoder similarity but too slow to run on the
full corpus - which is why we funnel through BM25 + vector first.

Result: top-5 chunks with the highest relevance to the actual query.
"""

import cohere


def rerank(
    query: str,
    candidates: list[dict],
    cohere_api_key: str,
    top_n: int = 5,
    model: str = "rerank-english-v3.0",
) -> list[dict]:
    """
    Rerank candidate chunks using Cohere's rerank API.

    Args:
        query:          The user's original question.
        candidates:     Chunk dicts from RRF fusion (top-40).
        cohere_api_key: Cohere API key.
        top_n:          Final chunks to return after reranking.
        model:          Cohere rerank model name.

    Returns:
        Top-N chunks sorted by rerank relevance score (descending),
        with 'rerank_score' added.
    """
    if not candidates:
        return []

    client = cohere.Client(api_key=cohere_api_key)
    documents = [chunk["text"] for chunk in candidates]

    response = client.rerank(
        model=model,
        query=query,
        documents=documents,
        top_n=top_n,
        return_documents=False,
    )

    reranked = []
    for result in response.results:
        chunk = dict(candidates[result.index])
        chunk["rerank_score"] = round(result.relevance_score, 4)
        chunk["retrieval_source"] = "reranked"
        reranked.append(chunk)

    return reranked
