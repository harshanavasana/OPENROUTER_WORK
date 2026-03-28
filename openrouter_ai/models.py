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
    # Groq production / preview Meta Llama ids (see https://console.groq.com/docs/models)
    LLAMA3_8B_8192 = "llama-3.1-8b-instant"
    # Mid tier: Llama 4 Scout (Mixtral slot kept for stable enum / cost-ladder ordering)
    MIXTRAL_8X7B_32768 = "meta-llama/llama-4-scout-17b-16e-instruct"
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
