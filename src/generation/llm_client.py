"""
llm_client.py - OpenAI GPT-4o wrapper with structured output and Langfuse v4 tracing

Decision support extension:
- Uses OpenAI structured outputs (beta.chat.completions.parse) to enforce a typed schema.
- Confidence score blends LLM self-assessment (70%) + retrieval signal from Cohere rerank (30%).
- Returns claims, confidence_score, confidence_reasoning, decision_recommendation, data_sufficiency.
"""

import time
from typing import Literal

from openai import OpenAI
from pydantic import BaseModel, Field

from src.generation.prompt_builder import build_messages


class Claim(BaseModel):
    statement: str = Field(
        description="A single factual claim extracted verbatim or closely paraphrased from the filing"
    )
    citation: str = Field(
        description="Citation in format [TICKER, FORM_TYPE, FILING_DATE, Page N]"
    )


class StructuredAnswer(BaseModel):
    answer: str = Field(
        description="Full narrative answer to the question with inline citations after every claim"
    )
    claims: list[Claim] = Field(
        description="Every individual factual claim from the answer, each paired with its citation"
    )
    llm_confidence: float = Field(
        description="How completely the retrieved context covers the question (0.0-1.0)",
        ge=0.0,
        le=1.0,
    )
    confidence_reasoning: str = Field(
        description="One or two sentences explaining why this confidence level was assigned"
    )
    decision_recommendation: str = Field(
        description=(
            "Concise, actionable recommendation a financial analyst could act on, "
            "grounded only in the retrieved filings. State what is missing if context is insufficient."
        )
    )
    data_sufficiency: Literal["SUFFICIENT", "PARTIAL", "INSUFFICIENT"] = Field(
        description="Whether the retrieved context fully, partially, or insufficiently covers the question"
    )


def _retrieval_signal(chunks: list[dict]) -> float | None:
    """
    Derive a retrieval quality signal from Cohere rerank scores (0-1 scale).
    Returns None if no rerank scores are present (reranker was skipped).
    """
    scores = [
        c["rerank_score"]
        for c in chunks
        if c.get("rerank_score") is not None
    ]
    return sum(scores) / len(scores) if scores else None


def generate_answer(
    query: str,
    chunks: list[dict],
    openai_api_key: str,
    langfuse_client=None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
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
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # GPT-4o pricing (early 2025): $5/1M input, $15/1M output
    cost_usd = round((input_tokens * 5 + output_tokens * 15) / 1_000_000, 6)

    # Blend LLM self-assessed confidence (70%) with retrieval signal (30%).
    # If reranker was skipped (no rerank_score fields), use LLM confidence directly.
    retrieval_sig = _retrieval_signal(chunks)
    if retrieval_sig is not None:
        confidence_score = round(0.7 * structured.llm_confidence + 0.3 * retrieval_sig, 3)
    else:
        confidence_score = round(structured.llm_confidence, 3)

    result = {
        "answer": structured.answer,
        "claims": [c.model_dump() for c in structured.claims],
        "confidence_score": confidence_score,
        "confidence_reasoning": structured.confidence_reasoning,
        "decision_recommendation": structured.decision_recommendation,
        "data_sufficiency": structured.data_sufficiency,
        "latency_ms": latency_ms,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "cost_usd": cost_usd,
        "model": model,
        "chunks_used": len(chunks),
    }

    # Langfuse v4 tracing
    if langfuse_client:
        try:
            obs = langfuse_client.start_observation(
                name="finance-rag-query",
                as_type="generation",
                input=messages,
                output=structured.answer,
                model=model,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                },
                cost_details={"total": cost_usd},
                metadata={
                    "latency_ms": latency_ms,
                    "chunks_used": len(chunks),
                    "confidence_score": confidence_score,
                    "data_sufficiency": structured.data_sufficiency,
                    "query": query,
                },
            )
            obs.end()
            langfuse_client.flush()
        except Exception as e:
            print(f"[langfuse] Trace failed (non-fatal): {e}")

    return result
