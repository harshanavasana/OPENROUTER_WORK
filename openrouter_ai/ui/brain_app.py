"""
LLM routing "brain" — visualizes complexity, cost, latency, tokens, and model choice.

Run from repo root:
  python -m streamlit run openrouter_ai/ui/brain_app.py

From openrouter_ai/:
  python -m streamlit run ui/brain_app.py
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

# Repo root on path
_REPO = Path(__file__).resolve().parent.parent.parent
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")
load_dotenv()

import streamlit as st

from openrouter_ai.pipeline import OpenRouterPipeline
from openrouter_ai.models import RoutingRequest
from openrouter_ai.router.smart_router import model_comparison_table, comparison_for_prompt

st.set_page_config(page_title="LLM Brain Router", layout="wide", initial_sidebar_state="expanded")


def _fmt_features(features: dict) -> str:
    lines = []
    for k, v in sorted(features.items()):
        lines.append(f"- **{k}**: `{v}`")
    return "\n".join(lines) if lines else "_No feature flags_"


def main() -> None:
    st.title("LLM brain router")
    st.caption(
        "Flow: **prompt rewrite (optional)** → **complexity** (features + ML score) → "
        "**router** (cost · latency · tokens · tier) → **one Groq Llama** to answer."
    )

    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key:
        st.warning(
            "Set `GROQ_API_KEY` in `openrouter_ai/.env` for answers. "
            "**Routing preview only** works without a key if you enable **Skip LLM prompt rewrite**."
        )

    with st.sidebar:
        st.header("Preferences")
        prefer_cost = st.toggle("Prefer lower **cost**", value=True)
        prefer_speed = st.toggle("Prefer **speed**", value=False)
        budget = st.text_input("Max budget (USD / request, optional)", placeholder="e.g. 0.002")
        skip_opt = st.toggle(
            "**Skip** prompt rewrite LLM (faster — one fewer Groq call)",
            value=False,
            help="Uses your text as-is for classification & routing.",
        )
        preview_only = st.toggle(
            "**Routing preview only** (no answer model)",
            value=False,
            help="Classify + route locally and show the brain; no executor call.",
        )
        st.divider()
        st.markdown(
            "Extra speed: set `GROQ_OPTIMIZER_SKIP_BELOW_TOKENS=16` in `.env` so tiny prompts "
            "skip the rewriter automatically."
        )

    prompt = st.text_area("Your prompt / question", height=140, placeholder="Ask anything…")

    max_budget: float | None = None
    if budget.strip():
        try:
            max_budget = float(budget.strip())
        except ValueError:
            st.error("Budget must be a number.")
            return

    col_run, _ = st.columns([1, 4])
    run = col_run.button("Run brain", type="primary", use_container_width=True)

    if not run or not prompt.strip():
        st.info("Enter a prompt and click **Run brain**.")
        return

    req = RoutingRequest(
        prompt=prompt.strip(),
        prefer_cost=prefer_cost,
        prefer_speed=prefer_speed,
        max_budget_usd=max_budget,
        skip_optimizer_llm=skip_opt,
    )
    pipe = OpenRouterPipeline()

    try:
        result = None
        status = st.empty()

        if preview_only:
            status.info("Running: classify + route (no executor)…")
            decision = pipe.analyze_routing_sync(req)
            exec_in = decision.optimized_prompt.tokens_optimized
            exec_out = 200
            status.success("Done — routing preview only.")
        else:
            def on_routed(d):
                status.info(
                    f"Routed → **{d.selected_model.value}** · {d.complexity.level.value} · "
                    f"~${d.estimated_cost_usd:.6f} · ~{d.estimated_latency_ms:.0f} ms p50 — generating answer…"
                )

            result = pipe.run_sync(req, on_routed=on_routed)
            decision = result.decision
            exec_in, exec_out = result.input_tokens, result.output_tokens
            status.success("Done — full pipeline.")

        # --- Brain dashboard ---
        st.divider()
        st.subheader("Brain output — suggested model & drivers")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Suggested model", decision.selected_model.value, help="Groq Llama id used as the worker")
        c2.metric("Complexity", f"{decision.complexity.level.value} ({decision.complexity.score:.2f})")
        c3.metric("Est. $ (router)", f"{decision.estimated_cost_usd:.6f}")
        c4.metric("Catalog latency p50", f"{decision.estimated_latency_ms:.0f} ms")

        op = decision.optimized_prompt
        st.markdown("**Why this model?** The router combines your **complexity tier**, **token size**, and **cost vs speed** toggles with a fixed catalog (cheaper 8B → mid Llama 4 Scout → 70B for heavy prompts).")

        t1, t2 = st.columns(2)
        with t1:
            st.markdown("**Token size**")
            st.write(f"- Original prompt: **{op.tokens_original}** tokens (tiktoken)")
            st.write(f"- After rewrite: **{op.tokens_optimized}** (−**{op.compression_ratio * 100:.1f}%** vs original)")
        with t2:
            if result is not None:
                st.markdown("**LLM call (actual)**")
                st.write(
                    f"– **{result.input_tokens}** in + **{result.output_tokens}** out "
                    f"= **{result.tokens_used}** total"
                )
            else:
                st.markdown("**Comparison table uses**")
                st.write(f"– **{exec_in}** in + **{exec_out}** out (estimated out tokens if not executed)")

        st.markdown("**Classifier (brain signal)**")
        st.info(decision.complexity.reasoning)

        with st.expander("Raw feature signals", expanded=False):
            st.markdown(_fmt_features(decision.complexity.features))

        st.markdown("**Router narrative**")
        st.info(decision.rationale)

        st.markdown("**All models vs this prompt size** (est. cost · quality · latency)")
        st.dataframe(
            model_comparison_table(decision.selected_model, exec_in, exec_out),
            use_container_width=True,
            hide_index=True,
        )

        with st.expander("Longer comparison text", expanded=False):
            st.text(comparison_for_prompt(decision.selected_model, exec_in, exec_out))

        if op.optimized.strip() != op.original.strip():
            with st.expander("Rewritten prompt", expanded=False):
                st.code(op.optimized, language="text")

        if result is not None:
            st.divider()
            st.subheader("Assistant reply")
            st.write(result.response_text.strip())
            st.caption(
                f"Spend ~${result.actual_cost_usd:.6f} · wall clock {result.actual_latency_ms:.0f} ms · credits +{result.credits_earned}"
            )

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")


if __name__ == "__main__":
    main()
