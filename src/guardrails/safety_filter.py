"""
Safety guardrail: toxicity + brand safety.

Two checks run in a single LLM call to minimize latency:

  1. Toxicity — detects abusive, threatening, or harmful content.
  2. Brand safety — detects:
       • Competitor promotion ("use Bloomberg Terminal instead")
       • Defamatory / false claims about specific companies
       • Requests to generate misleading financial statements
       • Insider trading / market manipulation language

Uses gpt-4o-mini for cost efficiency (same reasoning as scope_filter).
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

class SafetyResult(BaseModel):
    safe: bool                  = Field(description="True if the content passes all checks")
    toxicity_flagged: bool      = Field(description="True if abusive or harmful content detected")
    brand_safety_flagged: bool  = Field(description="True if competitor promo, defamation, or manipulation detected")
    violations: list[str]       = Field(description="List of specific violation descriptions (empty if safe)")


# ---------------------------------------------------------------------------
# Prompt
# ---------------------------------------------------------------------------

_SAFETY_SYSTEM = """\
You are a content safety classifier for a professional financial research
platform used by institutional investors and compliance teams.

Evaluate the user message for two categories of violations:

TOXICITY — flag if the message contains:
  • Abusive, threatening, or harassing language
  • Explicit or graphic content
  • Self-harm or violence references

BRAND SAFETY — flag if the message contains:
  • Promotion of competing financial data products (Bloomberg, FactSet, etc.)
  • Requests to fabricate, falsify, or manipulate financial data
  • Content that could constitute market manipulation or insider trading
  • Defamatory or false claims about specific companies or executives
  • Requests to bypass compliance / regulatory controls

Respond ONLY with valid JSON matching this schema:
{{
  "safe": <bool>,
  "toxicity_flagged": <bool>,
  "brand_safety_flagged": <bool>,
  "violations": ["<description1>", ...]
}}

If safe=true, violations must be an empty array.
"""

_SAFETY_HUMAN = "User message: {question}"

_SAFETY_PROMPT = ChatPromptTemplate.from_messages([
    ("system", _SAFETY_SYSTEM),
    ("human", _SAFETY_HUMAN),
])


# ---------------------------------------------------------------------------
# Classifier
# ---------------------------------------------------------------------------

@dataclass
class SafetyFilter:
    openai_api_key: str
    model: str = "gpt-4o-mini"
    temperature: float = 0.0

    def __post_init__(self) -> None:
        llm = ChatOpenAI(
            model=self.model,
            temperature=self.temperature,
            openai_api_key=self.openai_api_key,
        )
        self._chain = _SAFETY_PROMPT | llm | JsonOutputParser()

    def check(self, question: str) -> SafetyResult:
        raw = self._chain.invoke({"question": question})
        return SafetyResult(**raw)
