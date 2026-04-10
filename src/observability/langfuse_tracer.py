"""
langfuse_tracer.py

Thin wrapper around the Langfuse client for consistent trace naming
and metadata across the pipeline. Designed to be imported and used
in llm_client.py and optionally in the retrieval layer.

Usage:
    from src.observability.langfuse_tracer import build_langfuse_client
    client = build_langfuse_client()   # returns None if keys not set
"""

import os


def build_langfuse_client():
    """
    Build and return a Langfuse client if credentials are present in env.
    Returns None silently if LANGFUSE_SECRET_KEY is not set - this allows
    the app to run without tracing in local dev without raising errors.
    """
    secret_key = os.environ.get("LANGFUSE_SECRET_KEY")
    public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")

    if not secret_key or not public_key:
        return None

    try:
        from langfuse import Langfuse
        client = Langfuse(
            secret_key=secret_key,
            public_key=public_key,
            host=os.environ.get("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        print("[langfuse] Tracing enabled.")
        return client
    except ImportError:
        print("[langfuse] Package not installed - run: pip install langfuse")
        return None
    except Exception as e:
        print(f"[langfuse] Failed to initialise (non-fatal): {e}")
        return None


def log_retrieval_trace(
    langfuse_client,
    trace_id: str,
    query: str,
    vector_count: int,
    bm25_count: int,
    fused_count: int,
    reranked_count: int,
    latency_ms: float,
) -> None:
    """
    Log retrieval pipeline metrics to an existing Langfuse trace.
    Call this after the retrieval step, before generation.
    """
    if langfuse_client is None:
        return
    try:
        langfuse_client.span(
            trace_id=trace_id,
            name="retrieval-pipeline",
            input={"query": query},
            output={
                "vector_results": vector_count,
                "bm25_results": bm25_count,
                "after_fusion": fused_count,
                "after_rerank": reranked_count,
            },
            metadata={"latency_ms": latency_ms},
        )
    except Exception as e:
        print(f"[langfuse] Span log failed (non-fatal): {e}")
