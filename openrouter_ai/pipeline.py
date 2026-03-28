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
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Optional

from dotenv import load_dotenv

from openrouter_ai.models import RoutingRequest, RouteResponse, RoutingDecision
from openrouter_ai.agents.optimizer_agent import OptimizerAgent
from openrouter_ai.router.complexity_classifier import ComplexityClassifier
from openrouter_ai.router.smart_router import (
    SmartRouter,
    _estimate_cost,
    brain_central_model,
    catalog_latency_ms,
)
from openrouter_ai.agents.executor_agent import ExecutorAgent
from openrouter_ai.utils.credits import CreditSystem
from openrouter_ai.utils.analytics import AnalyticsEngine
from openrouter_ai.utils.dashboard_store import (
    append_event,
    infer_query_type,
    routing_intelligence_score,
)

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

        brain = brain_central_model()
        actual_cost = _estimate_cost(
            exec_result["model_used"],
            exec_result["input_tokens"],
            exec_result["output_tokens"],
        )
        baseline_est_cost = _estimate_cost(
            brain,
            exec_result["input_tokens"],
            exec_result["output_tokens"],
        )
        baseline_est_lat = catalog_latency_ms(brain)
        est_savings = baseline_est_cost - actual_cost
        est_lat_delta = total_latency_ms - baseline_est_lat

        baseline_live_cost: float | None = None
        baseline_live_latency: float | None = None
        baseline_live_preview: str | None = None
        if request.run_baseline_live and self._groq_key:
            try:
                br = await self.executor.execute_for_model(
                    optimized_prompt.optimized,
                    brain,
                    system_prompt=self._system,
                    max_tokens=512,
                )
                baseline_live_cost = round(
                    _estimate_cost(brain, br["input_tokens"], br["output_tokens"]),
                    6,
                )
                baseline_live_latency = float(br["latency_ms"])
                baseline_live_preview = (br["response_text"] or "")[:800]
            except Exception:
                pass

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
            brain_central_model=brain.value,
            baseline_est_cost_usd=baseline_est_cost,
            est_savings_vs_brain_usd=est_savings,
        )

        feats = {k: float(v) for k, v in complexity.features.items()}
        qtype = infer_query_type(feats, request.prompt)
        riq = routing_intelligence_score(
            complexity.level.value,
            exec_result["model_used"].value,
            brain.value,
            True,
            est_savings,
        )
        append_event(
            {
                "ts": datetime.now(timezone.utc).isoformat(),
                "request_id": request_id,
                "user_id": user_id,
                "query_type": qtype,
                "complexity_level": complexity.level.value,
                "complexity_score": complexity.score,
                "complexity_features": feats,
                "rationale": decision.rationale,
                "routed_model": exec_result["model_used"].value,
                "brain_central_model": brain.value,
                "routed_cost_usd": round(actual_cost, 8),
                "routed_latency_ms": round(total_latency_ms, 2),
                "executor_only_latency_ms": round(float(exec_result["latency_ms"]), 2),
                "baseline_est_cost_usd": round(baseline_est_cost, 8),
                "baseline_est_latency_ms": round(baseline_est_lat, 2),
                "est_savings_vs_brain_usd": round(est_savings, 8),
                "est_latency_delta_vs_brain_ms": round(est_lat_delta, 2),
                "baseline_live_cost_usd": baseline_live_cost,
                "baseline_live_latency_ms": baseline_live_latency,
                "success": True,
                "input_tokens": exec_result["input_tokens"],
                "output_tokens": exec_result["output_tokens"],
                "compression_ratio": optimized_prompt.compression_ratio,
                "router_confidence": round(complexity.score, 4),
                "routing_intelligence_score": round(riq, 2),
                "tokens_original": optimized_prompt.tokens_original,
                "tokens_optimized": optimized_prompt.tokens_optimized,
            }
        )

        # Update router's live metrics (EMA latency)
        self.router.update_metrics(
            model=exec_result["model_used"],
            latency_ms=exec_result["latency_ms"],
            rate_limit_remaining=999,
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
            brain_central_model=brain.value,
            baseline_est_cost_usd=round(baseline_est_cost, 6),
            baseline_est_latency_ms=round(baseline_est_lat, 1),
            est_savings_vs_brain_usd=round(est_savings, 6),
            est_latency_delta_vs_brain_ms=round(est_lat_delta, 1),
            baseline_live_cost_usd=baseline_live_cost,
            baseline_live_latency_ms=baseline_live_latency,
            baseline_live_response_preview=baseline_live_preview,
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
