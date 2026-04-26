"""
main.py - FastAPI application (production edition)

Endpoints:
  POST /auth/token    — issue JWT (username + password)
  POST /ask           — full RAG pipeline (requires Bearer token)
  POST /ingest/run    — trigger ingestion (ADMIN only)
  GET  /health        — health check (public)

Security layers:
  1. JWT RBAC — every /ask call requires a valid Bearer token
  2. Ticker-level access — ANALYST/VIEWER restricted to allowed_tickers
  3. Guardrails — scope + safety checks run before the RAG pipeline
"""

import os
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import OAuth2PasswordRequestForm
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
from src.auth.models import User
from src.auth.rbac import (
    authenticate_user,
    create_access_token,
    require_permission,
    require_ticker_access,
)
from src.guardrails.chain import GuardrailChain

load_dotenv()

_state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    if not indexes_exist():
        raise RuntimeError("Indexes not found. Run `make ingest` first.")

    faiss_index, bm25_index, metadata = load_indexes()
    _state["faiss_index"]   = faiss_index
    _state["bm25_index"]    = bm25_index
    _state["metadata"]      = metadata
    _state["openai_client"] = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    _state["langfuse"]      = build_langfuse_client()

    # Guardrail chain — skip LLM checks if GUARDRAILS_DISABLED=true (testing)
    _disable = os.environ.get("GUARDRAILS_DISABLED", "false").lower() == "true"
    _state["guardrails"] = GuardrailChain(
        openai_api_key=os.environ["OPENAI_API_KEY"],
        skip_scope=_disable,
        skip_safety=_disable,
    )

    print(f"[api] Ready — {len(metadata)} chunks indexed. "
          f"Guardrails: {'disabled' if _disable else 'enabled'}")
    yield
    _state.clear()


app = FastAPI(
    title="Finance RAG — Ask My 10-Ks",
    description=(
        "Production RAG over SEC 10-K / 10-Q filings. "
        "JWT RBAC · LangChain guardrails · Hybrid BM25+vector retrieval · "
        "Cohere rerank · GPT-4o structured output · LangSmith + Langfuse tracing."
    ),
    version="2.0.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Request / response models
# ---------------------------------------------------------------------------

class AskRequest(BaseModel):
    question: str
    top_k_retrieval: int = 20
    top_n_rerank: int = 5
    # Optional: restrict answer to a specific ticker (enforced against RBAC)
    ticker_filter: str | None = None


class Claim(BaseModel):
    statement: str
    citation: str


class AskResponse(BaseModel):
    question: str
    answer: str
    claims: list[Claim]
    confidence_score: float
    confidence_reasoning: str
    decision_recommendation: str
    data_sufficiency: str
    chunks_used: int
    latency_ms: float
    cost_usd: float
    input_tokens: int
    output_tokens: int
    # Auth context echoed back so callers can audit
    queried_by: str
    role: str


class IngestResponse(BaseModel):
    status: str
    message: str


# ---------------------------------------------------------------------------
# Auth endpoint (public — no token required)
# ---------------------------------------------------------------------------

@app.post("/auth/token")
async def login(form_data: OAuth2PasswordRequestForm = Depends()):
    """
    Exchange username + password for a JWT Bearer token.

    Example:
        curl -X POST /auth/token \\
             -d "username=alice&password=alice-secret"
    """
    user = authenticate_user(form_data.username, form_data.password)
    return create_access_token(user)


# ---------------------------------------------------------------------------
# /ask  (authenticated + guardrailed)
# ---------------------------------------------------------------------------

@app.post("/ask", response_model=AskResponse)
async def ask(
    request: AskRequest,
    current_user: User = Depends(require_permission("query:assigned")),
):
    """
    Answer a question grounded in SEC filings.

    Security flow:
      1. JWT verified → current_user injected
      2. If ticker_filter provided → RBAC ticker check
      3. Guardrail chain (scope + safety) → block or pass
      4. Hybrid retrieval → RRF fusion → Cohere rerank → GPT-4o

    Requires role: ADMIN, ANALYST, or VIEWER (all can query).
    ANALYST/VIEWER restricted to their allowed_tickers.
    """
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty.")

    # Ticker-level RBAC check
    if request.ticker_filter:
        require_ticker_access(request.ticker_filter, current_user)

    # Guardrail pre-check
    guardrails: GuardrailChain = _state["guardrails"]
    guard_result = await guardrails.run(request.question)
    if not guard_result.passed:
        raise HTTPException(status_code=400, detail=guard_result.user_message)

    faiss_index   = _state["faiss_index"]
    bm25_index    = _state["bm25_index"]
    metadata      = _state["metadata"]
    openai_client = _state["openai_client"]
    langfuse      = _state["langfuse"]
    cohere_key    = os.environ.get("COHERE_API_KEY", "")

    # If the user has restricted tickers, filter metadata before retrieval
    effective_metadata = metadata
    if current_user.allowed_tickers:
        allowed = {t.upper() for t in current_user.allowed_tickers}
        effective_metadata = [
            m for m in metadata if m.get("ticker", "").upper() in allowed
        ]
        if not effective_metadata:
            raise HTTPException(
                status_code=403,
                detail="No indexed filings match your authorized tickers.",
            )

    vector_results = vector_search(
        request.question, faiss_index, effective_metadata, openai_client,
        top_k=request.top_k_retrieval,
    )
    bm25_results = bm25_search(
        request.question, bm25_index, effective_metadata,
        top_k=request.top_k_retrieval,
    )
    fused = reciprocal_rank_fusion([vector_results, bm25_results], top_n=40)

    if cohere_key:
        top_chunks = rerank(
            request.question, fused, cohere_key,
            top_n=request.top_n_rerank,
        )
    else:
        print("[api] No COHERE_API_KEY — skipping rerank, using top-N from fusion.")
        top_chunks = fused[: request.top_n_rerank]

    if not top_chunks:
        raise HTTPException(status_code=404, detail="No relevant filing content found.")

    result = generate_answer(
        query=request.question,
        chunks=top_chunks,
        openai_api_key=os.environ["OPENAI_API_KEY"],
        langfuse_client=langfuse,
        user_id=current_user.user_id,
    )

    return AskResponse(
        question=request.question,
        answer=result["answer"],
        claims=[Claim(**c) for c in result["claims"]],
        confidence_score=result["confidence_score"],
        confidence_reasoning=result["confidence_reasoning"],
        decision_recommendation=result["decision_recommendation"],
        data_sufficiency=result["data_sufficiency"],
        chunks_used=result["chunks_used"],
        latency_ms=result["latency_ms"],
        cost_usd=result["cost_usd"],
        input_tokens=result["input_tokens"],
        output_tokens=result["output_tokens"],
        queried_by=current_user.username,
        role=current_user.role.value,
    )


# ---------------------------------------------------------------------------
# /ingest/run  (ADMIN only)
# ---------------------------------------------------------------------------

@app.post("/ingest/run", response_model=IngestResponse)
async def run_ingest(
    current_user: User = Depends(require_permission("ingest:run")),
):
    """
    Trigger the ingestion pipeline via the API.
    Restricted to ADMIN role. In production, this kicks off a background task
    or an AWS Step Functions execution rather than running inline.
    """
    # Inline import avoids loading heavy ingestion deps at startup
    from src.ingestion.run import run_ingestion_pipeline
    try:
        run_ingestion_pipeline()
        return IngestResponse(status="ok", message="Ingestion completed successfully.")
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ---------------------------------------------------------------------------
# /health  (public)
# ---------------------------------------------------------------------------

@app.get("/health")
async def health():
    return {
        "status": "ok",
        "chunks_indexed": len(_state.get("metadata", [])),
        "version": "2.0.0",
    }
