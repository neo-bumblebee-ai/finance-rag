"""
Finance scope guardrail.

Uses a lightweight GPT-4o-mini classification call to determine whether an
incoming question is within the system's scope: public company financials,
SEC filings, earnings, risk factors, capital structure, and related topics.

Out-of-scope examples:
  - "Write me a poem about apples"
  - "What is the weather in New York?"
  - "How do I cook pasta?"

Cost note: gpt-4o-mini at ~$0.00015 / 1k input tokens makes this check
negligible (< $0.0001 per call).
"""

from __future__ import annotations

from dataclasses import dataclass

from langchain_core.output_parsers import JsonOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Output schema
# ---------------------------------------------------------------------------

class ScopeResult(BaseModel):
    in_scope: bool = Field(description="True if the question is finance/SEC-filing related")
    reason: str    = Field(description="One-sentence explanation of the decision")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SCOPE_SYSTEM = """\
You are a finance RAG scope classifier. Your only job is to decide whether a
user's question is relevant to this system, which answers questions about:

  • Public company SEC filings (10-K, 10-Q, 8-K)
  • Earnings, revenue, profitability, cash flow
  • Risk factors, regulatory disclosures
  • Capital structure, debt, equity, dividends
  • Business segments, geographic exposure
  • Executive compensation disclosures
  • Auditor opinions and accounting policies
  • Financial ratios (EBITDA, EPS, P/E, ROE, etc.)
  • Macroeconomic topics ONLY when discussed in a filing

Out of scope (respond with in_scope=false):
  • General knowledge questions unrelated to company financials
  • Coding, recipes, travel, entertainment, personal advice
  • Real-time market data, stock price predictions, trading signals
  • Legal / tax advice (beyond what the filing says)

Respond ONLY with valid JSON matching this schema:
{{"in_scope": <bool>, "reason": "<one sentence>"}}
"""

_SCOPE_HUMAN = "User question: {question}"

_SCOPE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SCOPE_SYSTEM),
    ("human", _SCOPE_HUMAN),
])


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class ScopeFilter:
    openai_api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
        )
        self._chain = _SCOPE_PROMPT | llm | JsonOutputParser()

    def check(self, question: str) -> ScopeResult:
        raw = self._chain.invoke({"question": question})
        return ScopeResult(**raw)
