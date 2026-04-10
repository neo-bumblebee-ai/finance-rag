"""
main.py - FastAPI application

POST /ask   - run the full retrieval + generation pipeline
GET  /health - health check with chunk count
"""

import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel
from dotenv import load_dotenv

from src.ingestion.indexer import load_indexes, indexes_exist
from src.retrieval.vector_search import vector_search
from src.retrieval.bm25_search import bm25_search
from src.retrieval.fusion import reciprocal_rank_fusion
from src.retrieval.reranker import rerank
from src.generation.llm_client import generate_answer
from src.observability.langfuse_tracer import build_langfuse_client

load_dotenv()

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not indexes_exist():
        raise RuntimeError(
            "Indexes not found. Run `make ingest` first."
        )
    faiss_index, bm25_index, metadata = load_indexes()
    _state["faiss_index"] = faiss_index
    _state["bm25_index"] = bm25_index
    _state["metadata"] = metadata
    _state["openai_client"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    langfuse_client = build_langfuse_client()
    _state["langfuse"] = langfuse_client
    print(f"[api] Ready - {len(metadata)} chunks indexed.")
    yield
    _state.clear()


app = FastAPI(
    title="Finance RAG - Ask My 10-Ks",
    description=(
        "Production RAG over SEC 10-K / 10-Q filings. "
        "Hybrid retrieval (BM25 + vector) -> RRF fusion -> Cohere rerank -> GPT-4o."
    ),
    version="1.0.0",
    lifespan=lifespan,
)


class AskRequest(BaseModel):
    question: str
    top_k_retrieval: int = 20
    top_n_rerank: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    chunks_used: int
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int


@app.post("/ask", response_model=AskResponse)
async def ask(request: AskRequest):
    """
    Answer a question about SEC filings.

    Pipeline:
        Vector search (top-K) + BM25 (top-K)
        -> RRF fusion (top-40)
        -> Cohere rerank (top-N)
        -> GPT-4o with citation-enforced prompt
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    faiss_index = _state["faiss_index"]
    bm25_index  = _state["bm25_index"]
    metadata    = _state["metadata"]
    openai_client = _state["openai_client"]
    langfuse    = _state["langfuse"]
    cohere_key  = os.environ.get("COHERE_API_KEY", "")

    vector_results = vector_search(
        request.question, faiss_index, metadata, openai_client,
        top_k=request.top_k_retrieval,
    )
    bm25_results = bm25_search(
        request.question, bm25_index, metadata,
        top_k=request.top_k_retrieval,
    )
    fused = reciprocal_rank_fusion(
        [vector_results, bm25_results], top_n=40
    )

    if cohere_key:
        top_chunks = rerank(
            request.question, fused, cohere_key,
            top_n=request.top_n_rerank,
        )
    else:
        print("[api] No COHERE_API_KEY - skipping rerank, using top-N from fusion.")
        top_chunks = fused[: request.top_n_rerank]

    if not top_chunks:
        raise HTTPException(
            status_code=404, detail="No relevant filing content found."
        )

    result = generate_answer(
        query=request.question,
        chunks=top_chunks,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        langfuse_client=langfuse,
    )

    return AskResponse(
        question=request.question,
        answer=result["answer"],
        chunks_used=result["chunks_used"],
        latency_ms=result["latency_ms"],
        cost_usd=result["cost_usd"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
    )


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chunks_indexed": len(_state.get("metadata", [])),
    }
