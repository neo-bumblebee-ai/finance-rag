"""
fusion.py — Reciprocal Rank Fusion (RRF)

Merges ranked lists from vector search and BM25 without needing to tune
score weights. RRF rewards chunks that rank highly in both lists and
degrades gracefully when one retriever returns noise.

Formula: RRF(d) = Σ_r  1 / (k + rank_r(d))   where k=60 (standard default)
"""


def reciprocal_rank_fusion(
    result_lists: list[list[dict]],
    id_key: str = "chunk_index",
    k: int = 60,
    top_n: int = 40,
) -> list[dict]:
    """
    Fuse multiple ranked result lists using RRF.

    Args:
        result_lists: Each list is sorted by descending relevance.
        id_key:       Metadata key that uniquely identifies a chunk.
        k:            RRF smoothing constant (60 is the standard default).
        top_n:        Number of fused results to return.

    Returns:
        Merged chunk list sorted by descending RRF score,
        with 'rrf_score' added to each chunk.
    """
    scores: dict[str, float] = {}
    chunk_map: dict[str, dict] = {}

    for result_list in result_lists:
        for rank, chunk in enumerate(result_list, start=1):
            chunk_id = str(chunk[id_key])
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (k + rank)
            if chunk_id not in chunk_map:
                chunk_map[chunk_id] = chunk

    fused = []
    for chunk_id, score in sorted(
        scores.items(), key=lambda x: x[1], reverse=True
    )[:top_n]:
        result = dict(chunk_map[chunk_id])
        result["rrf_score"] = round(score, 6)
        result["retrieval_source"] = "rrf_fusion"
        fused.append(result)

    return fused
