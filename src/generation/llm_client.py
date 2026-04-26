"""
llm_client.py - GPT-4o wrapper with structured output, Langfuse v4 tracing,
                and LangSmith run logging.

Decision support:
  - Structured output via beta.chat.completions.parse enforces a typed schema.
  - Confidence = 0.7 * llm_confidence + 0.3 * mean(rerank_scores).
  - Returns claims, confidence_score, confidence_reasoning,
    decision_recommendation, data_sufficiency.

Observability dual-write:
  - Langfuse  → per-request cost + latency traces (real-time dashboard)
  - LangSmith → dataset tracking, eval runs, cost aggregations
"""

import os
import time
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from src.generation.prompt_builder import build_messages


# ---------------------------------------------------------------------------
# Pydantic output schema (enforced by OpenAI structured outputs)
# ---------------------------------------------------------------------------

class Claim(BaseModel):
    statement: str = Field(
        description="A single factual claim extracted verbatim or closely paraphrased from the filing"
    )
    citation: str = Field(
        description="Citation in format [TICKER, FORM_TYPE, FILING_DATE, Page N]"
    )


class StructuredAnswer(BaseModel):
    answer: str = Field(
        description="Full narrative answer with inline citations after every claim"
    )
    claims: list[Claim] = Field(
        description="Every factual claim from the answer, each paired with its citation"
    )
    llm_confidence: float = Field(
        description="How completely the retrieved context covers the question (0.0–1.0)",
        ge=0.0, le=1.0,
    )
    confidence_reasoning: str = Field(
        description="One or two sentences explaining the confidence level"
    )
    decision_recommendation: str = Field(
        description=(
            "Concise, actionable recommendation a financial analyst could act on, "
            "grounded only in the retrieved filings. State gaps if context is insufficient."
        )
    )
    data_sufficiency: Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"] = Field(
        description="Whether retrieved context fully, partially, or insufficiently covers the question"
    )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _retrieval_signal(chunks: list[dict]) -> float | None:
    scores = [c["rerank_score"] for c in chunks if c.get("rerank_score") is not None]
    return sum(scores) / len(scores) if scores else None


def _langsmith_log(
    run_name: str,
    inputs: dict,
    outputs: dict,
    metadata: dict,
    user_id: str | None,
) -> None:
    """
    Log a run to LangSmith using the low-level REST client.
    Silently no-ops if LANGCHAIN_API_KEY is not set.

    LangSmith auto-traces LangChain objects, but for non-LangChain calls
    (direct OpenAI SDK) we use the langsmith.Client() run API.
    """
    api_key = os.environ.get("LANGCHAIN_API_KEY")
    if not api_key:
        return
    try:
        from langsmith import Client as LangSmithClient
        import uuid

        ls = LangSmithClient(api_key=api_key)
        project = os.environ.get("LANGCHAIN_PROJECT", "finance-rag-production")
        run_id = str(uuid.uuid4())

        ls.create_run(
            id=run_id,
            name=run_name,
            run_type="llm",
            inputs=inputs,
            project_name=project,
            extra={"metadata": {**metadata, "user_id": user_id or "anonymous"}},
        )
        ls.update_run(
            run_id,
            outputs=outputs,
            end_time=__import__("datetime").datetime.utcnow(),
        )
    except Exception as exc:
        print(f"[langsmith] Logging failed (non-fatal): {exc}")


# ---------------------------------------------------------------------------
# Main function
# ---------------------------------------------------------------------------

def generate_answer(
    query: str,
    chunks: list[dict],
    openai_api_key: str,
    langfuse_client=None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
    user_id: str | None = None,
) -> dict:
    """
    Generate a structured, grounded, cited answer from top-K reranked chunks.

    Returns a dict with:
        answer, claims, confidence_score, confidence_reasoning,
        decision_recommendation, data_sufficiency,
        latency_ms, input_tokens, output_tokens, cost_usd, chunks_used, model
    """
    client = OpenAI(api_key=openai_api_key)
    messages = build_messages(query, chunks)

    start = time.perf_counter()
    response = client.beta.chat.completions.parse(
        model=model,
        messages=messages,
        response_format=StructuredAnswer,
        temperature=temperature,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    structured: StructuredAnswer = response.choices[0].message.parsed
    input_tokens  = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # GPT-4o pricing (2025): $5/1M input, $15/1M output
    cost_usd = round((input_tokens * 5 + output_tokens * 15) / 1_000_000, 6)

    retrieval_sig = _retrieval_signal(chunks)
    confidence_score = round(
        0.7 * structured.llm_confidence + 0.3 * retrieval_sig
        if retrieval_sig is not None
        else structured.llm_confidence,
        3,
    )

    result = {
        "answer":                  structured.answer,
        "claims":                  [c.model_dump() for c in structured.claims],
        "confidence_score":        confidence_score,
        "confidence_reasoning":    structured.confidence_reasoning,
        "decision_recommendation": structured.decision_recommendation,
        "data_sufficiency":        structured.data_sufficiency,
        "latency_ms":              latency_ms,
        "input_tokens":            input_tokens,
        "output_tokens":           output_tokens,
        "cost_usd":                cost_usd,
        "model":                   model,
        "chunks_used":             len(chunks),
    }

    # -----------------------------------------------------------------------
    # Langfuse — real-time per-request tracing
    # -----------------------------------------------------------------------
    if langfuse_client:
        try:
            obs = langfuse_client.start_observation(
                name="finance-rag-query",
                as_type="generation",
                input=messages,
                output=structured.answer,
                model=model,
                usage_details={"input": input_tokens, "output": output_tokens},
                cost_details={"total": cost_usd},
                metadata={
                    "latency_ms":       latency_ms,
                    "chunks_used":      len(chunks),
                    "confidence_score": confidence_score,
                    "data_sufficiency": structured.data_sufficiency,
                    "query":            query,
                    "user_id":          user_id or "anonymous",
                },
            )
            obs.end()
            langfuse_client.flush()
        except Exception as exc:
            print(f"[langfuse] Trace failed (non-fatal): {exc}")

    # -----------------------------------------------------------------------
    # LangSmith — eval dataset + cost dashboard logging
    # -----------------------------------------------------------------------
    _langsmith_log(
        run_name="finance-rag-query",
        inputs={"query": query, "chunks_used": len(chunks)},
        outputs={
            "answer":           structured.answer,
            "confidence_score": confidence_score,
            "data_sufficiency": structured.data_sufficiency,
            "cost_usd":         cost_usd,
        },
        metadata={
            "model":       model,
            "latency_ms":  latency_ms,
            "input_tokens":  input_tokens,
            "output_tokens": output_tokens,
        },
        user_id=user_id,
    )

    return result
