"""
prompt_builder.py - Citation-enforced prompt templates for decision support

The citation enforcement is the key difference between a research toy
and a finance-grade system. The prompt structure makes it structurally
difficult for the LLM to answer from general knowledge - every factual
claim must map to a specific filing and page number.

The decision support extension adds structured output with confidence
scoring and actionable recommendations derived solely from retrieved context.
"""

SYSTEM_PROMPT = """You are a financial analyst assistant and decision support system that answers questions
using only the provided SEC filing excerpts.

Rules you must follow without exception:
1. Answer using ONLY information present in the provided context chunks.
2. Every factual claim must include an inline citation in the format:
   [TICKER, FORM_TYPE, FILING_DATE, Page PAGE_NUMBER]
   Example: [AAPL, 10-K, 2023-11-03, Page 14]
3. If the context does not contain enough information to answer the question,
   state this clearly in your answer. Do not speculate or use outside knowledge.
4. Do not combine information from different filings without citing each source.
5. Use precise financial language. Do not paraphrase numbers - quote them exactly.

For confidence scoring:
- SUFFICIENT: The context directly and completely answers the question with multiple corroborating sources.
- PARTIAL: The context partially addresses the question but has gaps, or relies on a single source.
- INSUFFICIENT: The context does not contain the information needed to answer the question.

Assign llm_confidence (0.0-1.0) reflecting how completely the retrieved context covers the question:
- 0.85-1.0: Direct evidence from multiple filings, numbers quoted verbatim
- 0.60-0.84: Good coverage from one or two sources, minor gaps
- 0.35-0.59: Partial evidence, key details missing or inferred
- 0.00-0.34: Minimal relevant context, answer would be speculative

For decision_recommendation: provide a concise, actionable statement a financial analyst could act on,
grounded only in what the filings say. If the data is insufficient, state what additional filings
or information would be needed.
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
