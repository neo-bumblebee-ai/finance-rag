"""
langfuse_tracer.py

Thin wrapper around the Langfuse v4 client.
Langfuse 4.x uses start_observation() instead of trace().
"""

import os


def build_langfuse_client():
    """
    Build and return a Langfuse client if credentials are present.
    Returns None silently if keys are not set.
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
        # Verify connection
        client.auth_check()
        print("[api] Langfuse tracing enabled.")
        return client
    except ImportError:
        print("[api] Langfuse not installed - tracing disabled.")
        return None
    except Exception as e:
        print(f"[api] Langfuse init failed (non-fatal): {e}")
        return None
