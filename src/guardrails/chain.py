"""
Combined guardrail chain.

Runs scope check and safety check in parallel (via asyncio.gather), then
combines results into a single GuardrailResult.

Usage:
    guardrails = GuardrailChain(openai_api_key=key)
    result = await guardrails.run(question)
    if not result.passed:
        raise HTTPException(400, detail=result.block_reason)
"""

from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from pydantic import BaseModel

from src.guardrails.scope_filter import ScopeFilter, ScopeResult
from src.guardrails.safety_filter import SafetyFilter, SafetyResult


# ---------------------------------------------------------------------------
# Combined result
# ---------------------------------------------------------------------------

class GuardrailResult(BaseModel):
    passed: bool
    block_reason: str | None       = None   # human-readable if blocked
    scope: ScopeResult  | None     = None
    safety: SafetyResult | None    = None

    # Friendly HTTP-safe detail string for 400 responses
    @property
    def user_message(self) -> str:
        if self.passed:
            return "OK"
        if self.scope and not self.scope.in_scope:
            return (
                "This question is outside the scope of the Finance RAG system. "
                "Please ask questions about SEC filings, earnings, risk factors, "
                "capital structure, or related financial topics."
            )
        if self.safety and not self.safety.safe:
            return (
                "Your question was blocked by the content safety policy. "
                "Please rephrase and avoid content that violates professional standards."
            )
        return self.block_reason or "Request blocked."


# ---------------------------------------------------------------------------
# Chain
# ---------------------------------------------------------------------------

@dataclass
class GuardrailChain:
    openai_api_key: str
    skip_scope: bool  = False   # set True in tests to bypass LLM scope check
    skip_safety: bool = False   # set True in tests to bypass LLM safety check

    def __post_init__(self) -> None:
        self._scope  = ScopeFilter(openai_api_key=self.openai_api_key)
        self._safety = SafetyFilter(openai_api_key=self.openai_api_key)

    async def run(self, question: str) -> GuardrailResult:
        """
        Run both checks concurrently.  Returns GuardrailResult.
        Short-circuits to passed=True if both skip flags are set (test mode).
        """
        if self.skip_scope and self.skip_safety:
            return GuardrailResult(passed=True)

        # Run in executor threads (LangChain sync clients)
        loop = asyncio.get_event_loop()

        scope_task  = loop.run_in_executor(None, self._scope.check, question)  if not self.skip_scope  else asyncio.sleep(0)
        safety_task = loop.run_in_executor(None, self._safety.check, question) if not self.skip_safety else asyncio.sleep(0)

        scope_result, safety_result = await asyncio.gather(scope_task, safety_task)

        # Evaluate
        scope_ok  = self.skip_scope  or (isinstance(scope_result,  ScopeResult)  and scope_result.in_scope)
        safety_ok = self.skip_safety or (isinstance(safety_result, SafetyResult) and safety_result.safe)

        if not scope_ok:
            return GuardrailResult(
                passed=False,
                block_reason=f"Out of scope: {scope_result.reason}",
                scope=scope_result,
                safety=safety_result if isinstance(safety_result, SafetyResult) else None,
            )

        if not safety_ok:
            return GuardrailResult(
                passed=False,
                block_reason=f"Safety violation: {', '.join(safety_result.violations)}",
                scope=scope_result if isinstance(scope_result, ScopeResult) else None,
                safety=safety_result,
            )

        return GuardrailResult(
            passed=True,
            scope=scope_result  if isinstance(scope_result,  ScopeResult)  else None,
            safety=safety_result if isinstance(safety_result, SafetyResult) else None,
        )
