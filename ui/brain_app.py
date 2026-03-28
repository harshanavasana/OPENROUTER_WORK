"""
LLM routing UI: live router + optimization dashboard.

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
from openrouter_ai.router.smart_router import brain_central_model, comparison_for_prompt, model_comparison_table
from openrouter_ai.ui.dashboard_panel import render_dashboard

st.set_page_config(
    page_title="AI Router — Live + Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)


def _fmt_features(features: dict) -> str:
    lines = []
    for k, v in sorted(features.items()):
        lines.append(f"- **{k}**: `{v}`")
    return "\n".join(lines) if lines else "_No feature flags_"


def render_router_tab() -> None:
    st.subheader("Live router")
    st.caption(
        "Flow: **prompt rewrite (optional)** → **complexity** (features + ML score) → "
        "**router** (10-model Groq catalogue) → **one model** to answer. "
        "Each full run appends a row to the **dashboard** event log."
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
        run_live_baseline = st.toggle(
            "**Live A/B**: also call brain model (2× Groq cost)",
            value=False,
            help="Runs the central brain model after the routed answer for apples-to-apples latency/cost on the same prompt.",
        )
        st.divider()
        st.markdown(
            "Baseline for savings: **`"
            + brain_central_model().value
            + "`** — set `ROUTER_BRAIN_MODEL` to override."
        )
        st.markdown(
            "Optional: `GROQ_OPTIMIZER_SKIP_BELOW_TOKENS=16` in `.env` skips rewriter on tiny prompts."
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
        run_baseline_live=run_live_baseline,
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
            status.success("Done — routing preview only (nothing written to dashboard log).")
        else:
            def on_routed(d):
                status.info(
                    f"Routed → **{d.selected_model.value}** · {d.complexity.level.value} · "
                    f"~${d.estimated_cost_usd:.6f} · ~{d.estimated_latency_ms:.0f} ms p50 — generating answer…"
                )

            result = pipe.run_sync(req, on_routed=on_routed)
            decision = result.decision
            exec_in, exec_out = result.input_tokens, result.output_tokens
            status.success("Done — full pipeline (dashboard log updated).")

        st.divider()
        st.subheader("Brain output — suggested model & drivers")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Suggested model", decision.selected_model.value, help="Groq model id used as the worker")
        c2.metric("Complexity", f"{decision.complexity.level.value} ({decision.complexity.score:.2f})")
        c3.metric("Est. $ (router)", f"{decision.estimated_cost_usd:.6f}")
        c4.metric("Catalog latency p50", f"{decision.estimated_latency_ms:.0f} ms")
        if result is not None:
            c5.metric("Est. savings vs brain", f"${result.est_savings_vs_brain_usd:.6f}")

        if result is not None:
            st.markdown("**Baseline vs routed (catalog estimates on same tokens)**")
            b1, b2, b3, b4 = st.columns(4)
            b1.metric("Brain model", result.brain_central_model)
            b2.metric("Brain est. $", f"{result.baseline_est_cost_usd:.6f}")
            b3.metric("Brain est. latency (catalog)", f"{result.baseline_est_latency_ms:.0f} ms")
            b4.metric("Δ latency (routed wall − brain cat.)", f"{result.est_latency_delta_vs_brain_ms:+.0f} ms")
            if result.baseline_live_cost_usd is not None:
                st.markdown("**Live A/B (second Groq call)**")
                l1, l2 = st.columns(2)
                l1.metric("Brain live $ (est.)", f"{result.baseline_live_cost_usd:.6f}")
                l2.metric("Brain live latency", f"{result.baseline_live_latency_ms:.0f} ms")
                if result.baseline_live_response_preview:
                    with st.expander("Brain reply preview (truncated)"):
                        st.text(result.baseline_live_response_preview)

        op = decision.optimized_prompt
        st.markdown(
            "**Why this model?** The router combines your **complexity tier**, **token size**, and **cost vs speed** "
            "toggles with a **10-model** Groq catalogue (see dashboard for full comparison table)."
        )

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

        st.markdown("**All catalog models vs this prompt size** (est. cost · quality · latency)")
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


def main() -> None:
    st.title("AI model routing optimization")
    tab_live, tab_dash = st.tabs(["Live router", "Optimization dashboard"])
    with tab_live:
        render_router_tab()
    with tab_dash:
        render_dashboard()


if __name__ == "__main__":
    main()
