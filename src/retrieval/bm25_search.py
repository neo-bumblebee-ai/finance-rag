"""
bm25_search.py — Keyword / sparse search via BM25

Why: Financial text contains exact terms that vector search consistently
underranks — ticker symbols (NVDA, JPM), metrics (EBITDA, Tier 1 capital),
and regulatory references (Basel III, SEC Rule 10b-5). BM25 catches these
precisely because it rewards exact token matches without embedding averaging.
"""

from rank_bm25 import BM25Okapi


def bm25_search(
    query: str,
    bm25_index: BM25Okapi,
    metadata: list[dict],
    top_k: int = 20,
) -> list[dict]:
    """
    Return top_k chunks by BM25 score.
    Each result includes the original chunk metadata + 'bm25_score'.
    """
    tokenised_query = query.lower().split()
    scores = bm25_index.get_scores(tokenised_query)

    top_indices = sorted(
        range(len(scores)), key=lambda i: scores[i], reverse=True
    )[:top_k]

    results = []
    for idx in top_indices:
        if scores[idx] == 0:
            continue
        chunk = dict(metadata[idx])
        chunk["bm25_score"] = float(scores[idx])
        chunk["retrieval_source"] = "bm25"
        results.append(chunk)
    return results
