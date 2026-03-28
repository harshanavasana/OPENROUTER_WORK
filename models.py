from enum import Enum
from pydantic import BaseModel
from typing import Dict, Optional


class ComplexityLevel(str, Enum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"


class ComplexityScore(BaseModel):
    level: ComplexityLevel
    score: float
    reasoning: str
    features: Dict[str, float]


class ModelChoice(str, Enum):
    """
    Groq chat model ids (production + preview). See https://console.groq.com/docs/models
    Pricing defaults match Groq docs ($/1M tokens → env overrides per 1K in smart_router).
    """

    # Tier: fast / cheap
    LLAMA3_8B_8192 = "llama-3.1-8b-instant"
    GPT_OSS_20B = "openai/gpt-oss-20b"
    GPT_OSS_SAFEGUARD_20B = "openai/gpt-oss-safeguard-20b"
    # Mid (MoE / general) — MIXTRAL_* name is legacy; Groq id is Llama 4 Scout
    MIXTRAL_8X7B_32768 = "meta-llama/llama-4-scout-17b-16e-instruct"
    MIXTRAL_LEGACY_8X7B = "mixtral-8x7b-32768"
    LLAMA4_MAVERICK = "meta-llama/llama-4-maverick-17b-128e-instruct"
    GPT_OSS_120B = "openai/gpt-oss-120b"
    QWEN25_32B = "qwen-2.5-32b"
    QWEN3_32B = "qwen/qwen3-32b"
    # Flagship — default “brain” baseline for routing comparisons
    LLAMA3_70B_8192 = "llama-3.3-70b-versatile"


class ModelMetrics(BaseModel):
    model: ModelChoice
    input_cost_per_1k: float
    output_cost_per_1k: float
    avg_latency_ms: float
    quality_score: float
    rate_limit_remaining: int


class OptimizedPrompt(BaseModel):
    original: str
    optimized: str
    compression_ratio: float
    tokens_original: int
    tokens_optimized: int


class RoutingDecision(BaseModel):
    selected_model: ModelChoice
    complexity: ComplexityScore
    optimized_prompt: OptimizedPrompt
    estimated_cost_usd: float
    estimated_latency_ms: float
    rationale: str


class RoutingRequest(BaseModel):
    prompt: str
    user_id: Optional[str] = None
    prefer_cost: bool = True
    prefer_speed: bool = False
    max_budget_usd: Optional[float] = None
    # Skip the LLM rewrite step (faster; routing uses raw prompt as "optimized")
    skip_optimizer_llm: bool = False
    # Second Groq call using the brain model for live A/B (2× cost/latency)
    run_baseline_live: bool = False


class RouteResponse(BaseModel):
    request_id: str
    decision: RoutingDecision
    response_text: str
    actual_cost_usd: float
    actual_latency_ms: float
    tokens_used: int
    input_tokens: int
    output_tokens: int
    credits_earned: float
    # Catalog baseline: same tokens on the central “brain” model (no extra API call)
    brain_central_model: str = ""
    baseline_est_cost_usd: float = 0.0
    baseline_est_latency_ms: float = 0.0
    est_savings_vs_brain_usd: float = 0.0
    est_latency_delta_vs_brain_ms: float = 0.0
    # Optional live baseline (when run_baseline_live=True)
    baseline_live_cost_usd: Optional[float] = None
    baseline_live_latency_ms: Optional[float] = None
    baseline_live_response_preview: Optional[str] = None


class InputRequest(BaseModel):
    text: str


class Classification(BaseModel):
    intent: str
    length: int
    complexity: Optional[str] = None
    score: Optional[float] = None


class OptimizedRequest(BaseModel):
    original_text: str
    classification: Classification
    priority: str
    processing_mode: str


class RouteInfo(BaseModel):
    route: str
    model: str
    processing_mode: str
    input: str


class ExecutionResult(BaseModel):
    route: str
    output: str
    mode: str
    model: str
    fallback_used: bool = False
