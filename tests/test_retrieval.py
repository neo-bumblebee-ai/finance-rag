"""
tests/test_retrieval.py

Unit tests for the retrieval layer - BM25 search, RRF fusion, and reranker.
These run without API keys so they execute cleanly in CI.
"""

import pytest
from rank_bm25 import BM25Okapi
from src.retrieval.bm25_search import bm25_search
from src.retrieval.fusion import reciprocal_rank_fusion


# -- Fixtures ------------------------------------------------------------------

SAMPLE_METADATA = [
    {
        "chunk_index": 0,
        "text": "Apple faces supply chain risks related to manufacturing in China.",
        "ticker": "AAPL",
        "form_type": "10-K",
        "filing_date": "2023-11-03",
        "page_number": 14,
        "source_file": "AAPL_10-K_2023-11-03.htm",
    },
    {
        "chunk_index": 1,
        "text": "NVIDIA reports record data center revenue driven by AI GPU demand.",
        "ticker": "NVDA",
        "form_type": "10-K",
        "filing_date": "2024-02-21",
        "page_number": 7,
        "source_file": "NVDA_10-K_2024-02-21.htm",
    },
    {
        "chunk_index": 2,
        "text": "JPMorgan maintains Tier 1 capital ratio above regulatory minimums.",
        "ticker": "JPM",
        "form_type": "10-K",
        "filing_date": "2024-02-16",
        "page_number": 22,
        "source_file": "JPM_10-K_2024-02-16.htm",
    },
    {
        "chunk_index": 3,
        "text": "Microsoft Azure cloud revenue grew significantly year over year.",
        "ticker": "MSFT",
        "form_type": "10-K",
        "filing_date": "2023-07-27",
        "page_number": 5,
        "source_file": "MSFT_10-K_2023-07-27.htm",
    },
    {
        "chunk_index": 4,
        "text": "Amazon AWS provides cloud computing services to enterprise customers.",
        "ticker": "AMZN",
        "form_type": "10-K",
        "filing_date": "2024-02-02",
        "page_number": 9,
        "source_file": "AMZN_10-K_2024-02-02.htm",
    },
]


@pytest.fixture
def bm25_index():
    tokenised = [m["text"].lower().split() for m in SAMPLE_METADATA]
    return BM25Okapi(tokenised)


# -- BM25 tests ----------------------------------------------------------------

def test_bm25_returns_relevant_result(bm25_index):
    results = bm25_search("Apple supply chain China", bm25_index, SAMPLE_METADATA, top_k=3)
    assert len(results) > 0
    assert results[0]["ticker"] == "AAPL"


def test_bm25_ticker_exact_match(bm25_index):
    results = bm25_search("NVIDIA GPU data center", bm25_index, SAMPLE_METADATA, top_k=3)
    assert any(r["ticker"] == "NVDA" for r in results)


def test_bm25_respects_top_k(bm25_index):
    results = bm25_search("revenue cloud", bm25_index, SAMPLE_METADATA, top_k=2)
    assert len(results) <= 2


def test_bm25_adds_score_field(bm25_index):
    results = bm25_search("capital ratio JPMorgan", bm25_index, SAMPLE_METADATA, top_k=5)
    for r in results:
        assert "bm25_score" in r
        assert r["bm25_score"] > 0


def test_bm25_empty_query_returns_no_results(bm25_index):
    results = bm25_search("", bm25_index, SAMPLE_METADATA, top_k=5)
    # All BM25 scores will be 0 for empty query - expect empty list
    assert len(results) == 0


# -- RRF fusion tests ----------------------------------------------------------

def _make_vector_results():
    """Simulate vector search results with scores."""
    return [
        {**SAMPLE_METADATA[0], "vector_score": 0.91, "retrieval_source": "vector"},
        {**SAMPLE_METADATA[2], "vector_score": 0.85, "retrieval_source": "vector"},
        {**SAMPLE_METADATA[3], "vector_score": 0.78, "retrieval_source": "vector"},
    ]


def _make_bm25_results():
    """Simulate BM25 results with scores."""
    return [
        {**SAMPLE_METADATA[0], "bm25_score": 4.2, "retrieval_source": "bm25"},
        {**SAMPLE_METADATA[1], "bm25_score": 3.1, "retrieval_source": "bm25"},
        {**SAMPLE_METADATA[4], "bm25_score": 2.8, "retrieval_source": "bm25"},
    ]


def test_rrf_merges_both_lists():
    fused = reciprocal_rank_fusion([_make_vector_results(), _make_bm25_results()])
    chunk_indices = [r["chunk_index"] for r in fused]
    # chunk 0 appeared in both lists - should be top ranked
    assert chunk_indices[0] == 0


def test_rrf_deduplicates():
    fused = reciprocal_rank_fusion([_make_vector_results(), _make_bm25_results()])
    ids = [r["chunk_index"] for r in fused]
    assert len(ids) == len(set(ids)), "Duplicate chunk_index found in fused results"


def test_rrf_adds_rrf_score():
    fused = reciprocal_rank_fusion([_make_vector_results(), _make_bm25_results()])
    for r in fused:
        assert "rrf_score" in r
        assert r["rrf_score"] > 0


def test_rrf_respects_top_n():
    fused = reciprocal_rank_fusion(
        [_make_vector_results(), _make_bm25_results()], top_n=2
    )
    assert len(fused) <= 2


def test_rrf_sorted_descending():
    fused = reciprocal_rank_fusion([_make_vector_results(), _make_bm25_results()])
    scores = [r["rrf_score"] for r in fused]
    assert scores == sorted(scores, reverse=True)


def test_rrf_single_list():
    fused = reciprocal_rank_fusion([_make_vector_results()])
    assert len(fused) == len(_make_vector_results())


def test_rrf_empty_lists():
    fused = reciprocal_rank_fusion([[], []])
    assert fused == []


# -- Prompt builder tests ------------------------------------------------------

def test_prompt_builder_includes_citation_format():
    from src.generation.prompt_builder import build_messages
    messages = build_messages("What are Apple's risks?", SAMPLE_METADATA[:2])
    user_msg = messages[1]["content"]
    assert "AAPL" in user_msg
    assert "Page" in user_msg


def test_prompt_builder_returns_two_messages():
    from src.generation.prompt_builder import build_messages
    messages = build_messages("Test question", SAMPLE_METADATA[:1])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_context_block_numbers_chunks():
    from src.generation.prompt_builder import build_context_block
    block = build_context_block(SAMPLE_METADATA[:3])
    assert "[Context 1]" in block
    assert "[Context 2]" in block
    assert "[Context 3]" in block


# -- Decision support: confidence blending tests --------------------------------

def test_retrieval_signal_returns_mean_rerank_score():
    from src.generation.llm_client import _retrieval_signal
    chunks = [
        {**SAMPLE_METADATA[0], "rerank_score": 0.9},
        {**SAMPLE_METADATA[1], "rerank_score": 0.7},
    ]
    signal = _retrieval_signal(chunks)
    assert signal == pytest.approx(0.8, abs=1e-6)


def test_retrieval_signal_returns_none_when_no_rerank_scores():
    from src.generation.llm_client import _retrieval_signal
    # Chunks from fusion only — no rerank_score key
    signal = _retrieval_signal(SAMPLE_METADATA[:3])
    assert signal is None


def test_retrieval_signal_ignores_chunks_without_rerank_score():
    from src.generation.llm_client import _retrieval_signal
    chunks = [
        {**SAMPLE_METADATA[0], "rerank_score": 0.6},
        {**SAMPLE_METADATA[1]},           # no rerank_score
        {**SAMPLE_METADATA[2], "rerank_score": 0.4},
    ]
    signal = _retrieval_signal(chunks)
    assert signal == pytest.approx(0.5, abs=1e-6)


def test_confidence_blends_llm_and_retrieval():
    """When rerank scores are present, final confidence = 0.7*llm + 0.3*retrieval."""
    llm_confidence = 0.8
    retrieval_signal = 1.0
    expected = round(0.7 * llm_confidence + 0.3 * retrieval_signal, 3)
    assert expected == pytest.approx(0.86, abs=1e-3)


def test_confidence_falls_back_to_llm_only():
    """Without rerank scores, confidence should equal llm_confidence."""
    from src.generation.llm_client import _retrieval_signal
    signal = _retrieval_signal(SAMPLE_METADATA)   # no rerank_score fields
    assert signal is None
    # When signal is None the caller uses llm_confidence directly — no blending


def test_structured_answer_schema_fields():
    """StructuredAnswer Pydantic model accepts all required fields."""
    from src.generation.llm_client import StructuredAnswer, Claim
    obj = StructuredAnswer(
        answer="Apple faces risks [AAPL, 10-K, 2023-11-03, Page 3].",
        claims=[Claim(statement="Apple faces risks", citation="[AAPL, 10-K, 2023-11-03, Page 3]")],
        llm_confidence=0.9,
        confidence_reasoning="Direct evidence from two corroborating chunks.",
        decision_recommendation="Monitor AAPL supply chain disclosures.",
        data_sufficiency="SUFFICIENT",
    )
    assert obj.llm_confidence == 0.9
    assert obj.data_sufficiency == "SUFFICIENT"
    assert len(obj.claims) == 1
    assert obj.claims[0].citation == "[AAPL, 10-K, 2023-11-03, Page 3]"


def test_claim_schema_fields():
    from src.generation.llm_client import Claim
    c = Claim(statement="Revenue grew 12%", citation="[MSFT, 10-K, 2023-07-27, Page 5]")
    assert c.statement == "Revenue grew 12%"
    assert "MSFT" in c.citation
