"""
openrouter_ai/pipeline.py

OpenRouter AI Pipeline — main orchestrator.

Wires all four stages together:
  1. OptimizerAgent  — compress & clarify the prompt
  2. ComplexityClassifier — score the task complexity
  3. SmartRouter     — pick the optimal model
  4. ExecutorAgent   — call the model, handle fallback
  + CreditSystem & AnalyticsEngine for rewards and telemetry

Usage:
    from openrouter_ai.pipeline import OpenRouterPipeline

    pipeline = OpenRouterPipeline(
        groq_api_key="gsk-...",
    )
    result = await pipeline.run(RoutingRequest(prompt="Explain quantum entanglement."))
    print(result.response_text)
"""

import uuid
import time
import os
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from openrouter_ai.models import RoutingRequest, RouteResponse, RoutingDecision
from openrouter_ai.agents.optimizer_agent import OptimizerAgent
from openrouter_ai.router.complexity_classifier import ComplexityClassifier
from openrouter_ai.router.smart_router import SmartRouter
from openrouter_ai.agents.executor_agent import ExecutorAgent
from openrouter_ai.utils.credits import CreditSystem
from openrouter_ai.utils.analytics import AnalyticsEngine

# Load secrets from openrouter_ai/.env regardless of current working directory.
load_dotenv(Path(__file__).resolve().parent / ".env")
load_dotenv()


class OpenRouterPipeline:
    """
    Top-level pipeline.  Instantiate once per application, reuse across requests.
    """

    def __init__(
        self,
        groq_api_key: str | None = None,
        system_prompt: str | None = None,
    ):
        self._groq_key = groq_api_key or os.getenv("GROQ_API_KEY", "")
        self._system  = system_prompt

        # Stage 1 — Optimizer
        self.optimizer   = OptimizerAgent(groq_api_key=self._groq_key)
        # Stage 2 — Classifier
        self.classifier  = ComplexityClassifier(groq_api_key=self._groq_key)
        # Stage 3 — Router
        self.router      = SmartRouter()
        # Stage 4 — Executor
        self.executor    = ExecutorAgent(groq_api_key=self._groq_key)
        # Support systems
        self.credits     = CreditSystem()
        self.analytics   = AnalyticsEngine()

    async def analyze_routing(self, request: RoutingRequest) -> RoutingDecision:
        """
        Stages 1–3 only: optional prompt rewrite, complexity score, model pick.
        Does not call the downstream executor (no second LLM for the answer).
        """
        if request.skip_optimizer_llm:
            optimized_prompt = self.optimizer.identity_optimize(request.prompt)
        else:
            optimized_prompt = await self.optimizer.optimize(request.prompt)

        complexity = self.classifier.classify(optimized_prompt.optimized)
        return self.router.route(
            complexity=complexity,
            optimized_prompt=optimized_prompt,
            prefer_cost=request.prefer_cost,
            prefer_speed=request.prefer_speed,
            max_budget_usd=request.max_budget_usd,
        )

    def analyze_routing_sync(self, request: RoutingRequest) -> RoutingDecision:
        import asyncio
        return asyncio.run(self.analyze_routing(request))

    async def run(
        self,
        request: RoutingRequest,
        *,
        on_routed: Optional[Callable[[RoutingDecision], None]] = None,
    ) -> RouteResponse:
        """
        Full async pipeline execution.

        ``on_routed`` is invoked once after the router chooses a model (for UIs / logging).
        """
        request_id = str(uuid.uuid4())
        user_id    = request.user_id or "anonymous"
        t_pipeline_start = time.perf_counter()

        decision = await self.analyze_routing(request)
        complexity = decision.complexity
        optimized_prompt = decision.optimized_prompt

        if on_routed is not None:
            on_routed(decision)

        # ── Stage 4: Execute ───────────────────────────────────────────────────
        exec_result = await self.executor.execute(decision, system_prompt=self._system)

        total_latency_ms = (time.perf_counter() - t_pipeline_start) * 1000

        # ── Compute actual cost ────────────────────────────────────────────────
        from openrouter_ai.router.smart_router import _estimate_cost
        actual_cost = _estimate_cost(
            exec_result["model_used"],
            exec_result["input_tokens"],
            exec_result["output_tokens"],
        )

        # ── Credits ────────────────────────────────────────────────────────────
        credits_earned = self.credits.record_request(
            user_id=user_id,
            request_id=request_id,
            actual_cost_usd=actual_cost,
            input_tokens=exec_result["input_tokens"],
            output_tokens=exec_result["output_tokens"],
        )

        # ── Analytics ──────────────────────────────────────────────────────────
        self.analytics.record_event(
            request_id=request_id,
            user_id=user_id,
            model_used=exec_result["model_used"],
            complexity_level=complexity.level.value,
            complexity_score=complexity.score,
            compression_ratio=optimized_prompt.compression_ratio,
            tokens_original=optimized_prompt.tokens_original,
            tokens_optimized=optimized_prompt.tokens_optimized,
            input_tokens=exec_result["input_tokens"],
            output_tokens=exec_result["output_tokens"],
            actual_cost_usd=actual_cost,
            latency_ms=total_latency_ms,
            credits_earned=credits_earned,
        )

        # Update router's live metrics (EMA latency)
        self.router.update_metrics(
            model=exec_result["model_used"],
            latency_ms=exec_result["latency_ms"],
            rate_limit_remaining=999,   # replace with real value from API headers
        )

        return RouteResponse(
            request_id=request_id,
            decision=decision,
            response_text=exec_result["response_text"],
            actual_cost_usd=round(actual_cost, 6),
            actual_latency_ms=round(total_latency_ms, 1),
            tokens_used=exec_result["input_tokens"] + exec_result["output_tokens"],
            input_tokens=exec_result["input_tokens"],
            output_tokens=exec_result["output_tokens"],
            credits_earned=round(credits_earned, 4),
        )

    def run_sync(
        self,
        request: RoutingRequest,
        *,
        on_routed: Optional[Callable[[RoutingDecision], None]] = None,
    ) -> RouteResponse:
        """Synchronous wrapper for scripts / REPL usage."""
        import asyncio
        return asyncio.run(self.run(request, on_routed=on_routed))


def run_pipeline(
    prompt: str,
    groq_api_key: str | None = None,
    user_id: str | None = None,
    prefer_cost: bool = True,
    prefer_speed: bool = False,
    max_budget_usd: float | None = None,
    system_prompt: str | None = None,
    skip_optimizer_llm: bool = False,
    on_routed: Optional[Callable[[RoutingDecision], None]] = None,
) -> RouteResponse:
    """Convenience helper for demo/tests to execute the full pipeline sync."""
    pipeline = OpenRouterPipeline(groq_api_key=groq_api_key, system_prompt=system_prompt)
    request = RoutingRequest(
        prompt=prompt,
        user_id=user_id,
        prefer_cost=prefer_cost,
        prefer_speed=prefer_speed,
        max_budget_usd=max_budget_usd,
        skip_optimizer_llm=skip_optimizer_llm,
    )
    return pipeline.run_sync(request, on_routed=on_routed)
