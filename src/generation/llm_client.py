"""
llm_client.py - OpenAI GPT-4o wrapper with Langfuse v4 tracing
"""

import time
from openai import OpenAI
from src.generation.prompt_builder import build_messages


def generate_answer(
    query: str,
    chunks: list[dict],
    openai_api_key: str,
    langfuse_client=None,
    model: str = "gpt-4o",
    temperature: float = 0.0,
) -> dict:
    """
    Generate a grounded, cited answer from top-K reranked chunks.
    Logs to Langfuse v4 if client is provided.
    """
    client = OpenAI(api_key=openai_api_key)
    messages = build_messages(query, chunks)

    start = time.perf_counter()
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=temperature,
    )
    latency_ms = round((time.perf_counter() - start) * 1000, 1)

    answer = response.choices[0].message.content
    input_tokens = response.usage.prompt_tokens
    output_tokens = response.usage.completion_tokens

    # GPT-4o pricing (early 2025): $5/1M input, $15/1M output
    cost_usd = round((input_tokens * 5 + output_tokens * 15) / 1_000_000, 6)

    result = {
        "answer": answer,
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
                output=answer,
                model=model,
                usage_details={
                    "input": input_tokens,
                    "output": output_tokens,
                },
                cost_details={"total": cost_usd},
                metadata={
                    "latency_ms": latency_ms,
                    "chunks_used": len(chunks),
                    "query": query,
                },
            )
            obs.end()
            langfuse_client.flush()
        except Exception as e:
            print(f"[langfuse] Trace failed (non-fatal): {e}")

    return result
