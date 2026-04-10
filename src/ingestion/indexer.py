"""
indexer.py

Builds and persists:
  1. FAISS vector index  - dense semantic search
  2. BM25 index          - keyword / sparse search
  3. Metadata JSON       - chunk provenance for citations

Both indexes are built from the same chunk list so they stay in sync.
Run once at ingestion time; subsequent queries load from disk.
"""

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from rank_bm25 import BM25Okapi

from src.ingestion.pdf_parser import Chunk

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIM = 1536

INDEX_DIR = Path("data/index")
FAISS_INDEX_PATH = INDEX_DIR / "faiss.index"
BM25_INDEX_PATH = INDEX_DIR / "bm25.pkl"
METADATA_PATH = INDEX_DIR / "metadata.json"


def _embed_texts(
    texts: list[str], client, batch_size: int = 100
) -> np.ndarray:
    """Embed texts in batches. Returns (N, 1536) float32 array."""
    all_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        response = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_embeddings.extend([item.embedding for item in response.data])
        print(f"[indexer] Embedded {min(i + batch_size, len(texts))}/{len(texts)}")
    return np.array(all_embeddings, dtype=np.float32)


def build_indexes(chunks: list[Chunk], openai_api_key: str) -> None:
    """
    Build FAISS and BM25 indexes from chunks and save to disk.
    Expensive - run once per corpus change.
    """
    from openai import OpenAI  # lazy import - only needed at build time
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    client = OpenAI(api_key=openai_api_key)
    texts = [c.text for c in chunks]
    metadata = [c.to_dict() for c in chunks]

    # FAISS - L2-normalised vectors give cosine similarity via IndexFlatIP
    print("[indexer] Building FAISS index...")
    embeddings = _embed_texts(texts, client)
    faiss.normalize_L2(embeddings)
    index = faiss.IndexFlatIP(EMBEDDING_DIM)
    index.add(embeddings)
    faiss.write_index(index, str(FAISS_INDEX_PATH))
    print(f"[indexer] FAISS: {index.ntotal} vectors -> {FAISS_INDEX_PATH}")

    # BM25
    print("[indexer] Building BM25 index...")
    tokenised = [t.lower().split() for t in texts]
    bm25 = BM25Okapi(tokenised)
    with open(BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print(f"[indexer] BM25 -> {BM25_INDEX_PATH}")

    # Metadata sidecar
    with open(METADATA_PATH, "w") as f:
        json.dump(metadata, f)
    print(f"[indexer] Metadata: {len(metadata)} chunks -> {METADATA_PATH}")


def load_indexes() -> tuple[faiss.Index, BM25Okapi, list[dict]]:
    """Load pre-built indexes from disk. Raises if not found."""
    for path in [FAISS_INDEX_PATH, BM25_INDEX_PATH, METADATA_PATH]:
        if not path.exists():
            raise FileNotFoundError(
                f"Index not found: {path}\n"
                "Run `make ingest` to build indexes first."
            )
    faiss_index = faiss.read_index(str(FAISS_INDEX_PATH))
    with open(BM25_INDEX_PATH, "rb") as f:
        bm25_index = pickle.load(f)
    with open(METADATA_PATH) as f:
        metadata = json.load(f)
    print(
        f"[indexer] Loaded - FAISS: {faiss_index.ntotal} vectors, "
        f"BM25 + {len(metadata)} chunks"
    )
    return faiss_index, bm25_index, metadata


def indexes_exist() -> bool:
    return all(
        p.exists() for p in [FAISS_INDEX_PATH, BM25_INDEX_PATH, METADATA_PATH]
    )
