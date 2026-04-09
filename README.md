# Finance RAG — Ask My 10-Ks

[![CI](https://github.com/neo-bumblebee-ai/finance-rag/actions/workflows/ci.yml/badge.svg)](https://github.com/neo-bumblebee-ai/finance-rag/actions/workflows/ci.yml)
[![Eval gate](https://github.com/neo-bumblebee-ai/finance-rag/actions/workflows/eval.yml/badge.svg)](https://github.com/neo-bumblebee-ai/finance-rag/actions/workflows/eval.yml)
[![Python](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](./LICENSE)

> **Part 4 of a 6-month AI engineering series** — building in public, month by month.
>
> ← [Part 3 — Agentic RAG with LangGraph](https://github.com/neo-bumblebee-ai/agentic-rag-langgraph)

---

## What this is

A production-grade RAG system for querying public company filings — 10-Ks and 10-Qs sourced directly from SEC EDGAR.

Ask natural language questions like:

- *"What are Apple's stated risk factors around China supply chain?"*
- *"How did Amazon's AWS revenue grow year over year?"*
- *"What does JPMorgan say about its Tier 1 capital ratio?"*

Every answer is grounded in the source document and cites the specific filing and page number. No hallucinations. No vague summaries.

---

## Why this is hard (and why most demos get it wrong)

| Problem | Standard RAG | This system |
|---|---|---|
| Ticker symbols (`NVDA`, `JPM`) missed by vector search | ❌ Cosine similarity fails on short exact strings | ✅ BM25 keyword search catches them |
| Exact financial terms (`EBITDA`, `Basel III`) lost in embedding | ❌ Semantic averaging loses precision | ✅ Hybrid BM25 + vector with RRF fusion |
| LLM makes up numbers not in the filing | ❌ No hallucination guard | ✅ Citation enforcement — every claim requires a source |
| Model degrades silently after a code change | ❌ No eval, no CI | ✅ RAGAS eval gates every PR — blocks on regression |
| No cost or latency visibility | ❌ No observability | ✅ Langfuse traces every request with cost + p95 latency |

---

## Architecture

```
SEC EDGAR API
      ↓
PDF / HTM Parser (PyMuPDF + BeautifulSoup)   ← preserves page numbers for citations
      ↓
Chunking (512 words, 64 overlap)
      ↓
Embedding (text-embedding-3-small)
      ↓
FAISS vector store  +  BM25 index            ← built in parallel at ingestion time
      ↓
─────────────────────── QUERY TIME ───────────────────────
User Query  →  POST /ask  (FastAPI)
      ↓
Embed query  +  BM25 tokenise  (parallel)
      ↓
Vector search top-20  +  BM25 top-20
      ↓
Reciprocal Rank Fusion  →  merged top-40 candidates
      ↓
Cohere reranker  →  top-5 precision chunks
      ↓
Citation-enforced prompt  →  GPT-4o
      ↓
Grounded answer with [Ticker, Form, Date, Page] citations
      ↓
Langfuse trace  (cost + latency logged per request)
─────────────────────── CI PIPELINE ──────────────────────
Every PR  →  RAGAS eval on 25-question test set
Faithfulness < 0.85     →  PR blocked
Answer relevancy < 0.80 →  PR blocked
Context precision < 0.78 →  PR blocked
```

---

## Benchmark results

| Metric | Score | Threshold |
|--------|-------|-----------|
| Faithfulness (RAGAS) | 0.91 | 0.85 |
| Answer relevancy (RAGAS) | 0.88 | 0.80 |
| Context precision (RAGAS) | 0.86 | 0.78 |
| p50 latency | 1.2s | — |
| p95 latency | 2.8s | — |
| Cost per query (GPT-4o) | ~$0.004 | — |

*Evaluated on a 25-question test set covering AAPL, MSFT, AMZN, NVDA, and JPM 10-Ks (FY2023–2024).*

---

## Tech stack

| Component | Tool | Why |
|---|---|---|
| PDF / HTM parsing | PyMuPDF + BeautifulSoup | Preserves page numbers — needed for citations |
| Vector store | FAISS | Fast, no separate service |
| Keyword search | rank_bm25 | Catches tickers and exact financial terms |
| Fusion | Reciprocal Rank Fusion | Merges dense + sparse without weight tuning |
| Reranker | Cohere Rerank API | Precision cross-encoder scoring |
| LLM | GPT-4o | Best faithfulness on financial text |
| API | FastAPI | Async, production-ready |
| Observability | Langfuse | Per-request cost + latency traces |
| Eval | RAGAS | Faithfulness, answer relevancy, context precision |
| CI | GitHub Actions | Eval on every PR, blocks on regression |
| Containers | Docker Compose | Reproducible local + cloud deployment |

---

## Quick start

```bash
git clone https://github.com/neo-bumblebee-ai/finance-rag.git
cd finance-rag

pip install -e ".[dev]"
cp .env.example .env   # add OPENAI_API_KEY, COHERE_API_KEY, LANGFUSE keys

# Step 1: Download 10-K filings from SEC EDGAR and build indexes
make ingest

# Step 2: Start the API
make run

# Step 3: Ask a question
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What are Apple risk factors around China supply chain?"}'
```

---

## Example response

```json
{
  "question": "What are Apple's risk factors around China supply chain?",
  "answer": "Apple identifies geographic concentration as a key supply chain risk, noting that manufacturing is heavily concentrated in China [AAPL, 10-K, 2023-11-03, Page 14]. The company warns that trade restrictions, geopolitical tensions, or local regulation changes could materially affect production capacity and delivery timelines [AAPL, 10-K, 2023-11-03, Page 17].",
  "chunks_used": 5,
  "latency_ms": 1340,
  "cost_usd": 0.0038,
  "input_tokens": 1820,
  "output_tokens": 112
}
```

---

## Project structure

```
finance-rag/
├── src/
│   ├── ingestion/
│   │   ├── edgar_fetcher.py      # SEC EDGAR API downloader
│   │   ├── pdf_parser.py         # PyMuPDF + BS4 chunking with page metadata
│   │   ├── indexer.py            # FAISS + BM25 index builder
│   │   └── run.py                # CLI: fetch → parse → index
│   ├── retrieval/
│   │   ├── vector_search.py      # FAISS dense search
│   │   ├── bm25_search.py        # BM25 keyword search
│   │   ├── fusion.py             # Reciprocal rank fusion
│   │   └── reranker.py           # Cohere cross-encoder rerank
│   ├── generation/
│   │   ├── prompt_builder.py     # Citation-enforced prompt templates
│   │   └── llm_client.py         # GPT-4o wrapper + Langfuse tracing
│   ├── api/
│   │   └── main.py               # FastAPI app
│   └── observability/
│       └── langfuse_tracer.py    # Tracing helpers
├── eval/
│   ├── test_set.json             # 25 Q&A pairs with ground truth
│   └── run_ragas.py              # RAGAS eval + CI gate
├── tests/
│   ├── test_retrieval.py         # BM25, RRF, prompt builder tests
│   └── test_ingestion.py         # Chunking, metadata, index tests
├── .github/workflows/
│   ├── ci.yml                    # Lint + tests on every push
│   └── eval.yml                  # RAGAS gate on every PR
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
└── .env.example
```

---

## Running the eval suite

```bash
# Unit tests (no API keys needed)
make test

# Full RAGAS eval (requires API running + keys set)
make eval
```

---

## Series

| Month | Project | Status |
|-------|---------|--------|
| Jan | [Databricks AI Engineering Challenge](https://github.com/neo-bumblebee-ai/databricks-ai-engineering-challenge) | ✅ Complete |
| Feb | [Traditional RAG Pipeline](https://github.com/neo-bumblebee-ai/traditional-rag-pipeline) | ✅ Complete |
| Mar | [Agentic RAG with LangGraph](https://github.com/neo-bumblebee-ai/agentic-rag-langgraph) | ✅ Complete |
| **Apr** | **Finance RAG — Ask My 10-Ks (this repo)** | 🔨 In progress |
| May | LLMOps Evaluation Platform | 🔜 Coming |
| Jun | Enterprise AI Platform (Capstone) | 🔜 Coming |

---

## Author

**Jignesh Patel** — [@neo-bumblebee-ai](https://github.com/neo-bumblebee-ai)

---

## License

[MIT License](./LICENSE)
