import os
import sys
import pytest

# Repo root (parent of openrouter_ai/) must be on path for `import openrouter_ai`.
_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _repo_root)

from openrouter_ai.router.complexity_classifier import ComplexityClassifier
from openrouter_ai.router.smart_router import SmartRouter
from openrouter_ai.agents.optimizer_agent import OptimizerAgent
from openrouter_ai.agents.executor_agent import ExecutorAgent
from openrouter_ai.models import OptimizedPrompt, ComplexityScore, ComplexityLevel


def test_classifier_heuristic_fallback():
    cls = ComplexityClassifier(groq_api_key="")
    score = cls.classify("What is the capital of France?")
    assert isinstance(score, ComplexityScore)
    assert score.level in (ComplexityLevel.SIMPLE, ComplexityLevel.MEDIUM, ComplexityLevel.COMPLEX)


def test_router_simple_model():
    router = SmartRouter()
    optimized_prompt = OptimizedPrompt(
        original="What is HTTP?",
        optimized="What is HTTP?",
        compression_ratio=0.0,
        tokens_original=4,
        tokens_optimized=4,
    )
    complexity = ComplexityScore(
        level=ComplexityLevel.SIMPLE,
        score=0.1,
        reasoning="simple",
        features={
            "token_count": 4,
            "has_code_block": False,
            "has_math": False,
            "has_multi_step": False,
            "has_comparison": False,
            "has_analysis": False,
            "has_creative": False,
            "has_reasoning": False,
            "has_factual": True,
            "unique_word_ratio": 1.0,
        },
    )
    decision = router.route(complexity=complexity, optimized_prompt=optimized_prompt)
    assert decision.selected_model is not None
    assert decision.selected_model != ""


def test_optimizer_requires_key():
    opt = OptimizerAgent(groq_api_key="")
    with pytest.raises(ValueError):
        opt.optimize_sync("Explain quicksort.")


def test_executor_requires_key():
    exec_agent = ExecutorAgent(groq_api_key="")
    optimized_prompt = OptimizedPrompt(
        original="Test",
        optimized="Test",
        compression_ratio=0.0,
        tokens_original=1,
        tokens_optimized=1,
    )
    decision = type("D", (), {"optimized_prompt": optimized_prompt, "selected_model": None})()
    with pytest.raises(ValueError):
        exec_agent.execute_sync(decision)

