"""
openrouter_ai/router/smart_router.py

Smart Router — Stage 3 of the pipeline.

Given the ComplexityScore and user preferences (cost vs quality vs speed),
selects the optimal ModelChoice and builds a RoutingDecision.

Selection matrix (Groq Llama family)
────────────────────────────────────────────────────────────────
Complexity   prefer_cost=True          prefer_speed=True        default
─────────────────────────────────────────────────────────────────
SIMPLE       llama-3.1-8b-instant      llama-3.1-8b-instant     llama-3.1-8b-instant
MEDIUM       llama-3.1-8b-instant      Llama 4 Scout            Llama 4 Scout
COMPLEX      Llama 4 Scout             llama-3.3-70b-versatile  llama-3.3-70b-versatile
─────────────────────────────────────────────────────────────────

The router also applies a budget guard: if estimated cost exceeds
max_budget_usd, it downgrades to the next cheaper model.
"""

import os
from typing import Any, List, Optional

from openrouter_ai.models import (
    ComplexityLevel, ComplexityScore, ModelChoice,
    ModelMetrics, RoutingDecision, OptimizedPrompt,
)


# ── Model catalogue ───────────────────────────────────────────────────────────
# Costs in USD per 1K tokens; latency in ms (rough p50 estimates).
_MODEL_CATALOGUE: dict[ModelChoice, ModelMetrics] = {
    ModelChoice.LLAMA3_8B_8192: ModelMetrics(
        model=ModelChoice.LLAMA3_8B_8192,
        input_cost_per_1k=float(os.getenv("LLAMA3_8B_INPUT_COST", "0.00005")),
        output_cost_per_1k=float(os.getenv("LLAMA3_8B_OUTPUT_COST", "0.00008")),
        avg_latency_ms=700,
        quality_score=0.80,
        rate_limit_remaining=1000,
    ),
    ModelChoice.MIXTRAL_8X7B_32768: ModelMetrics(
        model=ModelChoice.MIXTRAL_8X7B_32768,
        input_cost_per_1k=float(os.getenv("MIXTRAL_8X7B_INPUT_COST", "0.00011")),
        output_cost_per_1k=float(os.getenv("MIXTRAL_8X7B_OUTPUT_COST", "0.00034")),
        avg_latency_ms=1100,
        quality_score=0.87,
        rate_limit_remaining=1000,
    ),
    ModelChoice.LLAMA3_70B_8192: ModelMetrics(
        model=ModelChoice.LLAMA3_70B_8192,
        input_cost_per_1k=float(os.getenv("LLAMA3_70B_INPUT_COST", "0.00059")),
        output_cost_per_1k=float(os.getenv("LLAMA3_70B_OUTPUT_COST", "0.00079")),
        avg_latency_ms=2500,
        quality_score=0.95,
        rate_limit_remaining=1000,
    ),
}

# Cost order (cheapest → most expensive)
_COST_LADDER = [
    ModelChoice.LLAMA3_8B_8192,
    ModelChoice.MIXTRAL_8X7B_32768,
    ModelChoice.LLAMA3_70B_8192,
]


def _estimate_cost(model: ModelChoice, input_tokens: int, output_tokens: int = 200) -> float:
    m = _MODEL_CATALOGUE[model]
    return (
        input_tokens  / 1000 * m.input_cost_per_1k +
        output_tokens / 1000 * m.output_cost_per_1k
    )


def comparison_for_prompt(
    selected: ModelChoice,
    input_tokens: int,
    output_tokens: int,
) -> str:
    """
    Short explanation of why ``selected`` fits this request vs other catalog models
    (cost / quality / latency using the same token counts).
    """
    sel = _MODEL_CATALOGUE[selected]
    sel_cost = _estimate_cost(selected, input_tokens, output_tokens)
    lines: list[str] = [
        "Model lineup for this I/O size (catalog estimates — your pick marked):",
    ]
    for m in _COST_LADDER:
        mm = _MODEL_CATALOGUE[m]
        c = _estimate_cost(m, input_tokens, output_tokens)
        tag = "  ← routed here" if m == selected else ""
        lines.append(
            f"  • {m.value}{tag}: ~${c:.6f} | quality {mm.quality_score:.2f} | "
            f"p50 latency ~{mm.avg_latency_ms:.0f} ms"
        )

    cheaper = [m for m in _COST_LADDER if m != selected and _estimate_cost(m, input_tokens, output_tokens) < sel_cost]
    stronger = [m for m in _COST_LADDER if m != selected and _MODEL_CATALOGUE[m].quality_score > sel.quality_score]

    why: list[str] = [
        f"\nWhy {selected.value}?",
        f"  The router matched your complexity tier and cost/speed preferences to this checkpoint.",
        f"  For ~{input_tokens} input + ~{output_tokens} output tokens, its estimated bill is ~${sel_cost:.6f}.",
    ]
    if cheaper:
        why.append(
            f"  Cheaper options exist ({', '.join(c.value for c in cheaper)}) but would under-power "
            "this tier or conflict with the quality/speed matrix for your score."
        )
    if stronger:
        why.append(
            f"  Stronger models ({', '.join(s.value for s in stronger)}) score higher on quality but cost more "
            "and are slower — used when complexity is higher or defaults favor quality."
        )
    if not cheaper and not stronger:
        why.append("  Other models are either more expensive or lower quality for this profile; this is the sweet spot.")

    return "\n".join(lines + why)


def model_comparison_table(
    selected: ModelChoice,
    input_tokens: int,
    output_tokens: int,
) -> List[dict[str, Any]]:
    """Rows for dashboards / Streamlit (cost, quality, latency vs catalog)."""
    rows: List[dict[str, Any]] = []
    for m in _COST_LADDER:
        mm = _MODEL_CATALOGUE[m]
        rows.append(
            {
                "model": m.value,
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

    In production you'd inject live metrics (latency, rate-limit remaining)
    from a monitoring service instead of the static catalogue above.
    """

    # ── Selection matrix ──────────────────────────────────────────────────────
    _MATRIX: dict[ComplexityLevel, dict[str, ModelChoice]] = {
        ComplexityLevel.SIMPLE: {
            "cost":    ModelChoice.LLAMA3_8B_8192,
            "speed":   ModelChoice.LLAMA3_8B_8192,
            "default": ModelChoice.LLAMA3_8B_8192,
        },
        ComplexityLevel.MEDIUM: {
            "cost":    ModelChoice.LLAMA3_8B_8192,
            "speed":   ModelChoice.MIXTRAL_8X7B_32768,
            "default": ModelChoice.MIXTRAL_8X7B_32768,
        },
        ComplexityLevel.COMPLEX: {
            "cost":    ModelChoice.MIXTRAL_8X7B_32768,
            "speed":   ModelChoice.LLAMA3_70B_8192,
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
    ) -> RoutingDecision:

        # 1. Pick initial model from matrix
        mode = "speed" if prefer_speed else ("cost" if prefer_cost else "default")
        selected = self._MATRIX[complexity.level][mode]

        # 2. Budget guard — downgrade if over limit
        if max_budget_usd is not None:
            for model in _COST_LADDER:
                est = _estimate_cost(model, optimized_prompt.tokens_optimized)
                if est <= max_budget_usd:
                    selected = model
                    break

        # 3. Rate-limit guard — skip exhausted models
        metrics = _MODEL_CATALOGUE[selected]
        if metrics.rate_limit_remaining < 5:
            # fall back to next cheapest available
            for fallback in _COST_LADDER:
                if _MODEL_CATALOGUE[fallback].rate_limit_remaining >= 5:
                    selected = fallback
                    break

        metrics  = _MODEL_CATALOGUE[selected]
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
        if prefer_cost:  prefs.append("cost-optimised")
        if prefer_speed: prefs.append("speed-optimised")
        if not prefs:    prefs.append("quality-optimised")
        return (
            f"Complexity={complexity.level.value} (score={complexity.score:.2f}). "
            f"Mode: {', '.join(prefs)}. "
            f"Selected {model.value} — estimated ${est_cost:.5f} per request."
        )

    def get_model_metrics(self) -> dict:
        """Expose catalogue for monitoring dashboards."""
        return {k.value: v.model_dump() for k, v in _MODEL_CATALOGUE.items()}

    def update_metrics(self, model: ModelChoice, latency_ms: float, rate_limit_remaining: int):
        """Live update from the execution layer after each real call."""
        m = _MODEL_CATALOGUE[model]
        # Exponential moving average for latency
        _MODEL_CATALOGUE[model] = m.model_copy(update={
            "avg_latency_ms": 0.8 * m.avg_latency_ms + 0.2 * latency_ms,
            "rate_limit_remaining": rate_limit_remaining,
        })