"""
prompt_builder.py - Citation-enforced prompt templates

The citation enforcement is the key difference between a research toy
and a finance-grade system. The prompt structure makes it structurally
difficult for the LLM to answer from general knowledge - every factual
claim must map to a specific filing and page number.
"""

SYSTEM_PROMPT = """You are a financial analyst assistant that answers questions
using only the provided SEC filing excerpts.

Rules you must follow without exception:
1. Answer using ONLY information present in the provided context chunks.
2. Every factual claim must include an inline citation in the format:
   [TICKER, FORM_TYPE, FILING_DATE, Page PAGE_NUMBER]
   Example: [AAPL, 10-K, 2023-11-03, Page 14]
3. If the context does not contain enough information to answer the question,
   say exactly: "The provided filings do not contain sufficient information
   to answer this question." Do not speculate or use outside knowledge.
4. Do not combine information from different filings without citing each source.
5. Use precise financial language. Do not paraphrase numbers - quote them exactly.
"""


def build_context_block(chunks: list[dict]) -> str:
    """Format reranked chunks into a numbered context block."""
    lines = []
    for i, chunk in enumerate(chunks, start=1):
        ticker = chunk.get("ticker", "UNKNOWN")
        form_type = chunk.get("form_type", "FILING")
        filing_date = chunk.get("filing_date", "UNKNOWN")
        page = chunk.get("page_number", "?")
        text = chunk.get("text", "").strip()
        lines.append(
            f"[Context {i}] [{ticker}, {form_type}, {filing_date}, Page {page}]\n{text}"
        )
    return "\n\n".join(lines)


def build_messages(query: str, chunks: list[dict]) -> list[dict]:
    """
    Build the messages list for OpenAI chat completions.
    Ready to pass directly to client.chat.completions.create().
    """
    context_block = build_context_block(chunks)
    user_message = f"""Context from SEC filings:

{context_block}

---

Question: {query}

Answer using only the context above. Cite every factual claim with
[TICKER, FORM_TYPE, FILING_DATE, Page PAGE_NUMBER] inline.
"""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_message},
    ]
