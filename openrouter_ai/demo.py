import sys
from pathlib import Path

# Repo root must be on sys.path (so `import openrouter_ai` works when cwd is this folder).
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent / ".env")

from openrouter_ai.pipeline import run_pipeline
from openrouter_ai.router.smart_router import comparison_for_prompt


def print_routing_report(result) -> None:
    """Pretty-print suggested model, reasoning, tokens, and comparison vs alternatives."""
    d = result.decision
    op = d.optimized_prompt
    c = d.complexity

    print()
    print("========== ROUTING ==========")
    print(f"Suggested model : {d.selected_model.value}")
    print(f"Complexity      : {c.level.value} (score {c.score:.3f})")
    print(f"Classifier says : {c.reasoning}")
    print(f"Router says     : {d.rationale}")
    print()
    print("========== TOKENS ==========")
    print(f"Original prompt : {op.tokens_original} tokens (tiktoken est.)")
    print(f"After optimize  : {op.tokens_optimized} tokens "
          f"(compression {op.compression_ratio * 100:.1f}%)")
    print(f"LLM call (actual): {result.input_tokens} in + {result.output_tokens} out "
          f"(total {result.tokens_used})")
    print()
    print("========== VS OTHER MODELS ==========")
    print(
        comparison_for_prompt(
            d.selected_model,
            result.input_tokens,
            result.output_tokens,
        )
    )
    print()
    print("========== MODEL REPLY ==========")
    print(result.response_text.strip())
    print()
    print(f"(cost ~${result.actual_cost_usd:.6f}, latency {result.actual_latency_ms:.0f} ms, "
          f"credits +{result.credits_earned})")
    print()


def main():
    print("openrouter_ai demo — enter a prompt, or 'quit' to exit")
    print("Tests:  .venv\\Scripts\\python.exe -m pytest")
    print("UI:     .venv\\Scripts\\python.exe -m streamlit run ui\\brain_app.py")
    while True:
        try:
            text = input("-> ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break

        if text.lower() in ("quit", "exit", ""):
            if text.lower() in ("quit", "exit"):
                break
            continue

        try:
            result = run_pipeline(text)
            print_routing_report(result)
        except Exception as e:
            print(f"\nError: {e}\n")


if __name__ == "__main__":
    main()
