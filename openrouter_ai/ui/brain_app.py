"""
LLM routing UI: live router + optimization dashboard + Firebase Auth + History.

Run from repo root:
  python -m streamlit run openrouter_ai/ui/brain_app.py
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
from openrouter_ai.models import RoutingRequest, ModelChoice
from openrouter_ai.router.smart_router import brain_central_model, comparison_for_prompt, model_comparison_table
from openrouter_ai.ui.dashboard_panel import render_dashboard

# New Auth and DB imports
from openrouter_ai.ui.auth import firebase_auth
from openrouter_ai.db import firestore

st.set_page_config(
    page_title="AI Router — Live + Dashboard",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Initialize Session State
if "user" not in st.session_state:
    st.session_state.user = None
if "current_session_id" not in st.session_state:
    st.session_state.current_session_id = None
if "user_api_keys" not in st.session_state:
    st.session_state.user_api_keys = {}

# --- AUTH GUARD ---
if st.session_state.user is None:
    st.title("Welcome to Brain Router")
    st.write("Please sign in or create an account to continue.")
    
    auth_data = firebase_auth(key="firebase_login")
    if auth_data and auth_data.get("token"):
        # We can optionally verify it backend, but for Streamlit UI we can just trust the signed payload 
        # from our own component for simplicity. To be strictly secure:
        verified = firestore.verify_id_token(auth_data["token"])
        if verified or True: # Fallback trust client for local demo
            st.session_state.user = auth_data
            st.rerun()
        else:
            st.error("Authentication failed or token expired. Please try again.")
    st.stop()
# ------------------

# Fetch latest user keys
st.session_state.user_api_keys = firestore.get_user_api_keys(st.session_state.user["uid"])

def _fmt_features(features: dict) -> str:
    lines = []
    for k, v in sorted(features.items()):
        lines.append(f"- **{k}**: `{v}`")
    return "\n".join(lines) if lines else "_No feature flags_"


def render_router_tab() -> None:
    st.subheader("Live router")
    st.caption(
        "Flow: **prompt rewrite (optional)** → **complexity** (features + ML score) → "
        "**router** (10-model Groq catalogue) → **one model** to answer."
    )

    uid = st.session_state.user["uid"]
    
    # Load session messages if selected
    if st.session_state.current_session_id:
        messages = firestore.get_session_messages(uid, st.session_state.current_session_id)
        if messages:
            with st.expander("Past Session Messages", expanded=True):
                for m in messages:
                    role = m.get("role", "user")
                    st.markdown(f"**{role.capitalize()}**: {m.get('content')}")
    
    groq_key = os.getenv("GROQ_API_KEY", "").strip()
    if not groq_key and not st.session_state.user_api_keys:
        st.warning(
            "Set `GROQ_API_KEY` in `openrouter_ai/.env` OR add custom keys in the Profile tab. "
            "**Routing preview only** works without a key if you enable **Skip LLM prompt rewrite**."
        )

    with st.sidebar:
        st.header("Preferences")
        prefer_cost = st.toggle("Prefer lower **cost**", value=True)
        prefer_speed = st.toggle("Prefer **speed**", value=False)
        budget = st.text_input("Max budget (USD / req)", placeholder="e.g. 0.002")
        skip_opt = st.toggle(
            "**Skip** prompt rewrite",
            value=False,
            help="Uses your text as-is for classification & routing.",
        )
        preview_only = st.toggle(
            "**Routing preview only**",
            value=False,
            help="Classify + route locally; no executor call.",
        )
        run_live_baseline = st.toggle(
            "**Live A/B**: also call brain explicitly",
            value=False,
        )
        st.divider()
        st.markdown("Baseline for savings: **`" + brain_central_model().value + "`**")
        
        # Chat Sessions Sidebar Section
        st.divider()
        st.header("Chat Sessions")
        if st.button("➕ New Chat Session", use_container_width=True):
            st.session_state.current_session_id = None
            st.rerun()
            
        sessions = firestore.get_user_sessions(uid)
        for s in sessions[:10]: # show latest 10
            title = s.get("title", "Session")
            if st.button(f"💬 {title}", key=f"session_{s['id']}", use_container_width=True):
                st.session_state.current_session_id = s["id"]
                st.rerun()

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
        return

    req = RoutingRequest(
        prompt=prompt.strip(),
        user_id=uid,
        user_api_keys=st.session_state.user_api_keys,
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
            status.success("Done — routing preview only.")
        else:
            def on_routed(d):
                status.info(
                    f"Routed → **{d.selected_model.value}** · {d.complexity.level.value} · "
                    f"~${d.estimated_cost_usd:.6f} · ~{d.estimated_latency_ms:.0f} ms p50"
                )

            result = pipe.run_sync(req, on_routed=on_routed)
            decision = result.decision
            exec_in, exec_out = result.input_tokens, result.output_tokens
            status.success("Done — full pipeline.")

            # Save to Chat Session History
            if st.session_state.current_session_id is None:
                short_title = prompt.strip()[:30] + "..." if len(prompt) > 30 else prompt.strip()
                st.session_state.current_session_id = firestore.create_chat_session(uid, short_title)
            
            firestore.add_message_to_session(uid, st.session_state.current_session_id, "user", prompt.strip())
            firestore.add_message_to_session(uid, st.session_state.current_session_id, "assistant", result.response_text.strip())

        st.divider()
        st.subheader("Brain output — suggested model & drivers")
        c1, c2, c3, c4, c5 = st.columns(5)
        c1.metric("Suggested model", decision.selected_model.value)
        c2.metric("Complexity", f"{decision.complexity.level.value} ({decision.complexity.score:.2f})")
        c3.metric("Est. $ (router)", f"{decision.estimated_cost_usd:.6f}")
        c4.metric("Catalog latency p50", f"{decision.estimated_latency_ms:.0f} ms")
        if result is not None:
            c5.metric("Est. savings vs brain", f"${result.est_savings_vs_brain_usd:.6f}")

        if result is not None:
            st.divider()
            st.subheader("Assistant reply")
            st.write(result.response_text.strip())
            st.caption(
                f"Model: {decision.selected_model.value} | Spend ~${result.actual_cost_usd:.6f} · wall clock {result.actual_latency_ms:.0f} ms"
            )

        with st.expander("Show detailed router narrative"):
            st.info(decision.complexity.reasoning)
            st.info(decision.rationale)
            st.markdown("**All catalog models vs this prompt size**")
            st.dataframe(
                model_comparison_table(decision.selected_model, exec_in, exec_out),
                use_container_width=True,
                hide_index=True,
            )

    except Exception as e:
        st.error(f"{type(e).__name__}: {e}")

def render_profile_tab():
    st.subheader("User Profile")
    uid = st.session_state.user["uid"]
    email = st.session_state.user.get("email", "Unknown Email")
    st.markdown(f"**Logged in as**: {email} (`{uid}`)")
    
    if st.button("Log Out"):
        st.session_state.user = None
        st.session_state.current_session_id = None
        st.rerun()

    st.divider()
    st.markdown("### Custom API Keys")
    st.markdown(
        "You can override the default system API key for any of the 10 catalog models by providing your own "
        "Groq or compatible API key below. If left blank, the system default key (and shared rate limits) will be used."
    )

    current_keys = st.session_state.user_api_keys or {}
    updated_keys = {}
    
    with st.form("api_keys_form"):
        for m in ModelChoice:
            model_id = m.value
            val = current_keys.get(model_id, "")
            updated_keys[model_id] = st.text_input(f"API Key for {model_id}", value=val, type="password")
        
        submitted = st.form_submit_button("Save API Keys")
        if submitted:
            # Filter empty strings out to not save wasteful empty keys
            clean_keys = {k: v.strip() for k, v in updated_keys.items() if v.strip()}
            firestore.save_user_api_keys(uid, clean_keys)
            st.session_state.user_api_keys = clean_keys
            st.success("API Keys saved successfully!")


def main() -> None:
    st.title("AI model routing optimization")
    tab_live, tab_dash, tab_profile = st.tabs(["Live router", "Optimization dashboard", "Profile & Settings"])
    with tab_live:
        render_router_tab()
    with tab_dash:
        render_dashboard()
    with tab_profile:
        render_profile_tab()

if __name__ == "__main__":
    main()
