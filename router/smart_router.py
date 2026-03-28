"""
openrouter_ai/router/smart_router.py

Smart Router — Stage 3 of the pipeline.

Given the ComplexityScore and user preferences (cost vs quality vs speed),
selects the optimal ModelChoice and builds a RoutingDecision.

Catalogue: up to 10 Groq chat models (see https://console.groq.com/docs/models).
Central “brain” model (baseline for savings math): llama-3.3-70b-versatile by default,
override with env ``ROUTER_BRAIN_MODEL`` (full model id string).
"""

import os
from typing import Any, List, Optional

from openrouter_ai.models import (
    ComplexityLevel,
    ComplexityScore,
    ModelChoice,
    ModelMetrics,
    RoutingDecision,
    OptimizedPrompt,
)


def brain_central_model() -> ModelChoice:
    """Largest / anchor model used as the single-LLM baseline in dashboard comparisons."""
    raw = os.getenv("ROUTER_BRAIN_MODEL", "").strip()
    if raw:
        for m in ModelChoice:
            if m.value == raw:
                return m
    return ModelChoice.LLAMA3_70B_8192


# $/1M input and output from Groq docs (env override = USD per 1K tokens for that component)
def _per_1k_from_1m(per_1m: float) -> float:
    return per_1m / 1000.0


# ── Model catalogue ───────────────────────────────────────────────────────────
_MODEL_CATALOGUE: dict[ModelChoice, ModelMetrics] = {
    ModelChoice.LLAMA3_8B_8192: ModelMetrics(
        model=ModelChoice.LLAMA3_8B_8192,
        input_cost_per_1k=float(os.getenv("LLAMA3_8B_INPUT_COST", str(_per_1k_from_1m(0.05)))),
        output_cost_per_1k=float(os.getenv("LLAMA3_8B_OUTPUT_COST", str(_per_1k_from_1m(0.08)))),
        avg_latency_ms=700,
        quality_score=0.78,
        rate_limit_remaining=1000,
    ),
    ModelChoice.GPT_OSS_20B: ModelMetrics(
        model=ModelChoice.GPT_OSS_20B,
        input_cost_per_1k=float(os.getenv("GPT_OSS_20B_INPUT_COST", str(_per_1k_from_1m(0.075)))),
        output_cost_per_1k=float(os.getenv("GPT_OSS_20B_OUTPUT_COST", str(_per_1k_from_1m(0.30)))),
        avg_latency_ms=850,
        quality_score=0.82,
        rate_limit_remaining=1000,
    ),
    ModelChoice.GPT_OSS_SAFEGUARD_20B: ModelMetrics(
        model=ModelChoice.GPT_OSS_SAFEGUARD_20B,
        input_cost_per_1k=float(os.getenv("GPT_OSS_SAFE_INPUT_COST", str(_per_1k_from_1m(0.075)))),
        output_cost_per_1k=float(os.getenv("GPT_OSS_SAFE_OUTPUT_COST", str(_per_1k_from_1m(0.30)))),
        avg_latency_ms=880,
        quality_score=0.80,
        rate_limit_remaining=1000,
    ),
    ModelChoice.MIXTRAL_8X7B_32768: ModelMetrics(
        model=ModelChoice.MIXTRAL_8X7B_32768,
        input_cost_per_1k=float(os.getenv("SCOUT_INPUT_COST", str(_per_1k_from_1m(0.11)))),
        output_cost_per_1k=float(os.getenv("SCOUT_OUTPUT_COST", str(_per_1k_from_1m(0.34)))),
        avg_latency_ms=1100,
        quality_score=0.87,
        rate_limit_remaining=1000,
    ),
    ModelChoice.MIXTRAL_LEGACY_8X7B: ModelMetrics(
        model=ModelChoice.MIXTRAL_LEGACY_8X7B,
        input_cost_per_1k=float(os.getenv("MIXTRAL_INPUT_COST", str(_per_1k_from_1m(0.11)))),
        output_cost_per_1k=float(os.getenv("MIXTRAL_OUTPUT_COST", str(_per_1k_from_1m(0.34)))),
        avg_latency_ms=1200,
        quality_score=0.84,
        rate_limit_remaining=1000,
    ),
    ModelChoice.LLAMA4_MAVERICK: ModelMetrics(
        model=ModelChoice.LLAMA4_MAVERICK,
        input_cost_per_1k=float(os.getenv("MAVERICK_INPUT_COST", str(_per_1k_from_1m(0.18)))),
        output_cost_per_1k=float(os.getenv("MAVERICK_OUTPUT_COST", str(_per_1k_from_1m(0.50)))),
        avg_latency_ms=1400,
        quality_score=0.90,
        rate_limit_remaining=1000,
    ),
    ModelChoice.GPT_OSS_120B: ModelMetrics(
        model=ModelChoice.GPT_OSS_120B,
        input_cost_per_1k=float(os.getenv("GPT_OSS_120B_INPUT_COST", str(_per_1k_from_1m(0.15)))),
        output_cost_per_1k=float(os.getenv("GPT_OSS_120B_OUTPUT_COST", str(_per_1k_from_1m(0.60)))),
        avg_latency_ms=1600,
        quality_score=0.91,
        rate_limit_remaining=1000,
    ),
    ModelChoice.QWEN25_32B: ModelMetrics(
        model=ModelChoice.QWEN25_32B,
        input_cost_per_1k=float(os.getenv("QWEN25_INPUT_COST", str(_per_1k_from_1m(0.20)))),
        output_cost_per_1k=float(os.getenv("QWEN25_OUTPUT_COST", str(_per_1k_from_1m(0.50)))),
        avg_latency_ms=1500,
        quality_score=0.88,
        rate_limit_remaining=1000,
    ),
    ModelChoice.QWEN3_32B: ModelMetrics(
        model=ModelChoice.QWEN3_32B,
        input_cost_per_1k=float(os.getenv("QWEN3_INPUT_COST", str(_per_1k_from_1m(0.29)))),
        output_cost_per_1k=float(os.getenv("QWEN3_OUTPUT_COST", str(_per_1k_from_1m(0.59)))),
        avg_latency_ms=1700,
        quality_score=0.89,
        rate_limit_remaining=1000,
    ),
    ModelChoice.LLAMA3_70B_8192: ModelMetrics(
        model=ModelChoice.LLAMA3_70B_8192,
        input_cost_per_1k=float(os.getenv("LLAMA3_70B_INPUT_COST", str(_per_1k_from_1m(0.59)))),
        output_cost_per_1k=float(os.getenv("LLAMA3_70B_OUTPUT_COST", str(_per_1k_from_1m(0.79)))),
        avg_latency_ms=2500,
        quality_score=0.96,
        rate_limit_remaining=1000,
    ),
    ModelChoice.EDGE_LOCAL_LLAMA3: ModelMetrics(
        model=ModelChoice.EDGE_LOCAL_LLAMA3,
        input_cost_per_1k=0.0,
        output_cost_per_1k=0.0,
        avg_latency_ms=3000,
        quality_score=0.75,
        rate_limit_remaining=9999,
    ),
}


def cost_ladder() -> list[ModelChoice]:
    """Cheapest → most expensive for a reference token mix (used for budget downgrades)."""
    models = [m for m in _MODEL_CATALOGUE.keys() if m != ModelChoice.EDGE_LOCAL_LLAMA3]
    models.sort(key=lambda m: _estimate_cost(m, 500, 200))
    return models


def _estimate_cost(model: ModelChoice, input_tokens: int, output_tokens: int = 200) -> float:
    m = _MODEL_CATALOGUE[model]
    return (
        input_tokens / 1000 * m.input_cost_per_1k + output_tokens / 1000 * m.output_cost_per_1k
    )


def catalog_latency_ms(model: ModelChoice) -> float:
    return float(_MODEL_CATALOGUE[model].avg_latency_ms)


def comparison_for_prompt(
    selected: ModelChoice,
    input_tokens: int,
    output_tokens: int,
) -> str:
    sel = _MODEL_CATALOGUE[selected]
    sel_cost = _estimate_cost(selected, input_tokens, output_tokens)
    lines: list[str] = [
        "Model lineup for this I/O size (catalog estimates — your pick marked):",
    ]
    for m in cost_ladder():
        mm = _MODEL_CATALOGUE[m]
        c = _estimate_cost(m, input_tokens, output_tokens)
        tag = "  ← routed here" if m == selected else ""
        lines.append(
            f"  • {m.value}{tag}: ~${c:.6f} | quality {mm.quality_score:.2f} | "
            f"p50 latency ~{mm.avg_latency_ms:.0f} ms"
        )

    cheaper = [
        m
        for m in cost_ladder()
        if m != selected and _estimate_cost(m, input_tokens, output_tokens) < sel_cost
    ]
    stronger = [
        m for m in cost_ladder() if m != selected and _MODEL_CATALOGUE[m].quality_score > sel.quality_score
    ]

    why: list[str] = [
        f"\nWhy {selected.value}?",
        f"  The router matched your complexity tier and cost/speed preferences to this checkpoint.",
        f"  For ~{input_tokens} input + ~{output_tokens} output tokens, its estimated bill is ~${sel_cost:.6f}.",
    ]
    if cheaper:
        why.append(
            f"  Cheaper options exist ({', '.join(c.value for c in cheaper[:5])}) but may under-power "
            "this tier or conflict with the quality/speed matrix for your score."
        )
    if stronger:
        why.append(
            f"  Stronger models ({', '.join(s.value for s in stronger[:5])}) score higher on quality but "
            "cost more and are slower — used when complexity is higher or defaults favor quality."
        )
    if not cheaper and not stronger:
        why.append("  Other models are either more expensive or lower quality for this profile; this is the sweet spot.")

    return "\n".join(lines + why)


def model_comparison_table(
    selected: ModelChoice,
    input_tokens: int,
    output_tokens: int,
) -> List[dict[str, Any]]:
    rows: List[dict[str, Any]] = []
    brain = brain_central_model()
    for m in cost_ladder():
        mm = _MODEL_CATALOGUE[m]
        rows.append(
            {
                "model": m.value,
                "brain_baseline": m == brain,
                "routed_here": m == selected,
                "est_cost_usd": round(_estimate_cost(m, input_tokens, output_tokens), 8),
                "quality_0_1": mm.quality_score,
                "p50_latency_ms": int(mm.avg_latency_ms),
            }
        )
    return rows


class SmartRouter:
    """
    Stateless router: given complexity + preferences, returns a RoutingDecision.
    """

    _MATRIX: dict[ComplexityLevel, dict[str, ModelChoice]] = {
        ComplexityLevel.SIMPLE: {
            "cost": ModelChoice.LLAMA3_8B_8192,
            "speed": ModelChoice.LLAMA3_8B_8192,
            "default": ModelChoice.LLAMA3_8B_8192,
        },
        ComplexityLevel.MEDIUM: {
            "cost": ModelChoice.LLAMA3_8B_8192,
            "speed": ModelChoice.MIXTRAL_8X7B_32768,
            "default": ModelChoice.GPT_OSS_20B,
        },
        ComplexityLevel.COMPLEX: {
            "cost": ModelChoice.GPT_OSS_120B,
            "speed": ModelChoice.LLAMA3_70B_8192,
            "default": ModelChoice.LLAMA3_70B_8192,
        },
    }

    def route(
        self,
        complexity: ComplexityScore,
        optimized_prompt: OptimizedPrompt,
        prefer_cost: bool = True,
        prefer_speed: bool = False,
        max_budget_usd: Optional[float] = None,
        requires_edge: bool = False,
    ) -> RoutingDecision:

        if requires_edge:
            selected = ModelChoice.EDGE_LOCAL_LLAMA3
        else:
            mode = "speed" if prefer_speed else ("cost" if prefer_cost else "default")
            selected = self._MATRIX[complexity.level][mode]

            ladder = cost_ladder()
            if max_budget_usd is not None:
                for model in ladder:
                    est = _estimate_cost(model, optimized_prompt.tokens_optimized)
                    if est <= max_budget_usd:
                        selected = model
                        break

            metrics = _MODEL_CATALOGUE[selected]
            if metrics.rate_limit_remaining < 5:
                for fallback in ladder:
                    if _MODEL_CATALOGUE[fallback].rate_limit_remaining >= 5:
                        selected = fallback
                        break

        metrics = _MODEL_CATALOGUE[selected]
        est_cost = _estimate_cost(selected, optimized_prompt.tokens_optimized)
        rationale = self._build_rationale(selected, complexity, prefer_cost, prefer_speed, est_cost)

        return RoutingDecision(
            selected_model=selected,
            complexity=complexity,
            optimized_prompt=optimized_prompt,
            estimated_cost_usd=round(est_cost, 6),
            estimated_latency_ms=metrics.avg_latency_ms,
            rationale=rationale,
        )

    @staticmethod
    def _build_rationale(
        model: ModelChoice,
        complexity: ComplexityScore,
        prefer_cost: bool,
        prefer_speed: bool,
        est_cost: float,
    ) -> str:
        prefs = []
        if prefer_cost:
            prefs.append("cost-optimised")
        if prefer_speed:
            prefs.append("speed-optimised")
        if not prefs:
            prefs.append("quality-optimised")
        return (
            f"Complexity={complexity.level.value} (score={complexity.score:.2f}). "
            f"Mode: {', '.join(prefs)}. "
            f"Selected {model.value} — estimated ${est_cost:.5f} per request."
        )

    def get_model_metrics(self) -> dict:
        return {k.value: v.model_dump() for k, v in _MODEL_CATALOGUE.items()}

    def update_metrics(self, model: ModelChoice, latency_ms: float, rate_limit_remaining: int):
        m = _MODEL_CATALOGUE[model]
        _MODEL_CATALOGUE[model] = m.model_copy(
            update={
                "avg_latency_ms": 0.8 * m.avg_latency_ms + 0.2 * latency_ms,
                "rate_limit_remaining": rate_limit_remaining,
            }
        )
