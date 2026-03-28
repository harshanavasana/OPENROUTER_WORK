"""
Routing optimization dashboard — Altair + Streamlit (no Plotly; avoids large wheel downloads on slow networks).

Loads events from ``openrouter_ai/data/dashboard_events.jsonl`` (see ``dashboard_store``).
"""

from __future__ import annotations

import os
from datetime import datetime, timedelta, timezone
from typing import List

import altair as alt
import pandas as pd
import streamlit as st

from openrouter_ai.models import ModelChoice
from openrouter_ai.router.smart_router import brain_central_model, model_comparison_table
from openrouter_ai.utils.dashboard_store import clear_events, load_events, seed_synthetic_events


def _to_df(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    if "ts" in df.columns:
        df["ts"] = pd.to_datetime(df["ts"], utc=True, errors="coerce")
    return df


def _normalize_log_df(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing columns for older JSONL rows or partial writes."""
    if df.empty:
        return df
    out = df.copy()
    if "routed_cost_usd" not in out.columns:
        if "actual_cost_usd" in out.columns:
            out["routed_cost_usd"] = pd.to_numeric(out["actual_cost_usd"], errors="coerce").fillna(0.0)
        else:
            out["routed_cost_usd"] = 0.0
    if "routed_latency_ms" not in out.columns:
        if "latency_ms" in out.columns:
            out["routed_latency_ms"] = pd.to_numeric(out["latency_ms"], errors="coerce").fillna(0.0)
        else:
            out["routed_latency_ms"] = 0.0
    if "baseline_est_cost_usd" not in out.columns:
        out["baseline_est_cost_usd"] = out["routed_cost_usd"]
    if "baseline_est_latency_ms" not in out.columns:
        out["baseline_est_latency_ms"] = 2500.0
    if "est_savings_vs_brain_usd" not in out.columns:
        out["est_savings_vs_brain_usd"] = (
            pd.to_numeric(out["baseline_est_cost_usd"], errors="coerce").fillna(0.0)
            - pd.to_numeric(out["routed_cost_usd"], errors="coerce").fillna(0.0)
        )
    if "success" not in out.columns:
        out["success"] = True
    if "routing_intelligence_score" not in out.columns:
        out["routing_intelligence_score"] = 70.0
    return out


def _numeric(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series([default] * len(df))
    return pd.to_numeric(df[col], errors="coerce").fillna(default)


def render_dashboard() -> None:
    st.header("AI model routing optimization")
    st.caption(
        "Baseline = catalog cost & latency for the **central brain** model on the same tokens; "
        "routed = your ML router + execution. Logged on each full pipeline run."
    )

    brain = brain_central_model()
    st.markdown(f"**Central brain (baseline anchor):** `{brain.value}` — override with `ROUTER_BRAIN_MODEL` in `.env`.")

    c_tool, c_tool2, c_tool3 = st.columns(3)
    with c_tool:
        if st.button("Load synthetic demo data", help="Fills the log with sample points for charts"):
            seed_synthetic_events(48)
            st.rerun()
    with c_tool2:
        if st.button("Clear event log", type="secondary"):
            clear_events()
            st.rerun()
    with c_tool3:
        st.caption(f"Events file: `{os.getenv('DASHBOARD_EVENTS_FILE', 'openrouter_ai/data/dashboard_events.jsonl')}`")

    df = _normalize_log_df(_to_df(load_events()))
    if df.empty:
        st.info("No events yet. Run prompts in the **Live router** tab (full pipeline), or click **Load synthetic demo data**.")
        return

    # Filters
    st.subheader("Filters")
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        types = sorted(df["query_type"].dropna().unique().tolist()) if "query_type" in df.columns else []
        sel_types = st.multiselect("Query type", types, default=types)
    with fc2:
        levels = sorted(df["complexity_level"].dropna().unique().tolist()) if "complexity_level" in df.columns else []
        sel_levels = st.multiselect("Complexity", levels, default=levels)
    with fc3:
        models = sorted(df["routed_model"].dropna().unique().tolist()) if "routed_model" in df.columns else []
        sel_models = st.multiselect("Routed model", models, default=models)
    with fc4:
        hours = st.selectbox("Time window", [6, 24, 72, 168, 9999], format_func=lambda x: "All time" if x > 900 else f"Last {x} h")
    if "ts" in df.columns and hours < 9000:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        df = df[df["ts"] >= cutoff]
    if sel_types and "query_type" in df.columns:
        df = df[df["query_type"].isin(sel_types)]
    if sel_levels and "complexity_level" in df.columns:
        df = df[df["complexity_level"].isin(sel_levels)]
    if sel_models and "routed_model" in df.columns:
        df = df[df["routed_model"].isin(sel_models)]

    if df.empty:
        st.warning("No rows after filters.")
        return

    df = df.sort_values("ts") if "ts" in df.columns else df

    routed_cost = _numeric(df, "routed_cost_usd")
    base_cost = _numeric(df, "baseline_est_cost_usd")
    routed_lat = _numeric(df, "routed_latency_ms")
    base_lat = _numeric(df, "baseline_est_latency_ms")
    savings = _numeric(df, "est_savings_vs_brain_usd")
    in_tok = _numeric(df, "input_tokens")
    out_tok = _numeric(df, "output_tokens")
    success = df["success"] if "success" in df.columns else pd.Series([True] * len(df))

    total_base = float(base_cost.sum())
    total_routed = float(routed_cost.sum())
    total_savings = float(savings.sum())
    pct_saved = (100.0 * total_savings / total_base) if total_base > 0 else 0.0

    span_s = 1.0
    if "ts" in df.columns and len(df) > 1:
        span_s = max((df["ts"].max() - df["ts"].min()).total_seconds(), 1.0)
    rps = len(df) / span_s

    lat_p95_r = float(routed_lat.quantile(0.95))
    lat_p99_r = float(routed_lat.quantile(0.99))
    lat_p95_b = float(base_lat.quantile(0.95))
    lat_p99_b = float(base_lat.quantile(0.99))

    cps = float(routed_cost[success].sum()) / max(int(success.sum()), 1)
    eff = float(savings.sum()) / max(float(routed_lat.mean()), 1.0)

    st.markdown("---")
    h1, h2, h3, h4, h5 = st.columns(5)
    h1.metric("Cumulative savings (est.)", f"${total_savings:.4f}", delta=f"{pct_saved:.1f}% vs brain")
    h2.metric("Avg latency (routed)", f"{routed_lat.mean():.0f} ms", delta=f"p95 {lat_p95_r:.0f} ms", delta_color="inverse")
    h3.metric("Catalog brain p95 latency", f"{lat_p95_b:.0f} ms", help="Baseline latency from static catalog")
    h4.metric("Throughput", f"{rps:.3f} req/s", help="Over visible time span")
    h5.metric("Routing intelligence (avg)", f"{_numeric(df, 'routing_intelligence_score').mean():.1f}")

    st.markdown("#### AI-style insights (rule-based)")
    ins = []
    if pct_saved > 5:
        ins.append(f"You saved about **{pct_saved:.1f}%** on estimated spend vs always using the brain model.")
    if lat_p95_r < lat_p95_b:
        ins.append(f"Routed **p95 latency** ({lat_p95_r:.0f} ms) is below catalog brain p95 ({lat_p95_b:.0f} ms) on this slice.")
    if len(routed_lat) > 3 and float(routed_lat.iloc[-1]) > lat_p95_r * 1.2:
        ins.append("**Anomaly:** latest request latency is elevated vs recent p95 — check model or prompt size.")
    top_m = df["routed_model"].value_counts().head(1)
    if not top_m.empty:
        ins.append(f"Most-used model: **{top_m.index[0]}** ({int(top_m.iloc[0])} requests).")
    for line in ins:
        st.markdown(f"- {line}")

    tab_a, tab_b, tab_c, tab_d = st.tabs(
        ["Trends & comparison", "Distributions & frontier", "Flow & utilization", "Explainability & drill-down"]
    )

    with tab_a:
        if "ts" in df.columns:
            dfp = df.copy()
            dfp["savings_cum"] = savings.cumsum().to_numpy()
            ch1 = (
                alt.Chart(dfp)
                .mark_line(strokeWidth=3)
                .encode(
                    alt.X("ts:T", title="Time"),
                    alt.Y("savings_cum:Q", title="USD"),
                    color=alt.value("#10b981"),
                )
                .properties(height=320, title="Cumulative estimated savings vs brain")
            )
            st.altair_chart(ch1, use_container_width=True)

            dfp["routed_ms"] = routed_lat.to_numpy()
            dfp["brain_ms"] = base_lat.to_numpy()
            ch_r = (
                alt.Chart(dfp)
                .mark_circle(size=45, color="#6366f1")
                .encode(
                    alt.X("ts:T", title="Time"),
                    alt.Y("routed_ms:Q", title="ms"),
                )
            )
            ch_b = (
                alt.Chart(dfp)
                .mark_line(strokeDash=[5, 5], color="#94a3b8")
                .encode(
                    alt.X("ts:T"),
                    alt.Y("brain_ms:Q"),
                )
            )
            st.altair_chart(
                (ch_b + ch_r).properties(height=280, title="Routed latency (points) vs brain catalog (dashed line)"),
                use_container_width=True,
            )

        mode = st.radio("Comparison style", ["Side-by-side bars", "Overlay lines"], horizontal=True)
        agg = pd.DataFrame(
            {
                "metric": ["Total $ (est.)", "Mean latency ms", "p95 latency ms"],
                "baseline": [total_base, base_lat.mean(), lat_p95_b],
                "routed": [total_routed, routed_lat.mean(), lat_p95_r],
            }
        )
        aggm = agg.melt(id_vars="metric", var_name="system", value_name="value")
        if mode == "Side-by-side bars":
            ch3 = (
                alt.Chart(aggm)
                .mark_bar()
                .encode(
                    alt.X("metric:N", title=None),
                    alt.Y("value:Q"),
                    alt.XOffset("system:N"),
                    alt.Color("system:N"),
                )
                .properties(height=380, title="Baseline vs routed")
            )
        else:
            ch3 = (
                alt.Chart(aggm)
                .mark_line(point=True)
                .encode(
                    alt.X("metric:N", title=None),
                    alt.Y("value:Q"),
                    alt.Color("system:N"),
                )
                .properties(height=380, title="Baseline vs routed")
            )
        st.altair_chart(ch3, use_container_width=True)

    with tab_b:
        df_sc = df.copy()
        df_sc["_tok"] = (in_tok + out_tok).to_numpy()
        sc_base = alt.Chart(df_sc).mark_circle().encode(
            alt.X("routed_latency_ms:Q", title="Routed latency (ms)"),
            alt.Y("routed_cost_usd:Q", title="Routed cost ($)"),
            alt.Size("_tok:Q", title="tokens", scale=alt.Scale(range=[20, 400])),
        )
        tip = ["routed_model", "routed_cost_usd", "routed_latency_ms"]
        if "request_id" in df_sc.columns:
            tip = ["request_id", "complexity_level", "query_type"] + tip
        if "routed_model" in df_sc.columns:
            enc = sc_base.encode(alt.Color("routed_model:N"), tooltip=tip)
        else:
            enc = sc_base.encode(color=alt.value("#6366f1"), tooltip=tip)
        st.altair_chart(enc.properties(height=400, title="Cost vs latency (size ∝ tokens)"), use_container_width=True)

        box_df = pd.DataFrame(
            {
                "ms": pd.concat([df["routed_latency_ms"], df["baseline_est_latency_ms"]], ignore_index=True),
                "series": (["Routed"] * len(df) + ["Brain catalog"] * len(df)),
            }
        )
        ch_v = (
            alt.Chart(box_df)
            .mark_boxplot(size=50, extent="min-max")
            .encode(
                alt.X("series:N", title=None),
                alt.Y("ms:Q", title="Latency (ms)"),
            )
            .properties(height=360, title="Latency distribution")
        )
        st.altair_chart(ch_v, use_container_width=True)

        st.markdown("**Innovation metrics (this slice)**")
        m1, m2 = st.columns(2)
        m1.metric("Cost per successful response", f"${cps:.6f}")
        m2.metric("Latency-adjusted savings density", f"{eff:.6f} $/ms (sum savings / mean latency)")

    with tab_c:
        if "query_type" in df.columns and "routed_model" in df.columns:
            heat = df.groupby(["query_type", "routed_model"], observed=False).size().reset_index(name="n")
            if not heat.empty:
                ch_h = (
                    alt.Chart(heat)
                    .mark_rect()
                    .encode(
                        alt.X("routed_model:N", title="Model"),
                        alt.Y("query_type:N", title="Query type"),
                        alt.Color("n:Q", title="Requests"),
                    )
                    .properties(height=max(200, 28 * heat["query_type"].nunique()), title="Query type × model")
                )
                st.altair_chart(ch_h, use_container_width=True)

        if all(c in df.columns for c in ("query_type", "complexity_level", "routed_model")):
            dsk = df.copy()
            qc = dsk.groupby(["query_type", "complexity_level"], observed=False).size().reset_index(name="n")
            cm = dsk.groupby(["complexity_level", "routed_model"], observed=False).size().reset_index(name="n")
            flows = []
            for _, r in qc.iterrows():
                flows.append({"stage": "query → complexity", "from": str(r["query_type"]), "to": str(r["complexity_level"]), "n": int(r["n"])})
            for _, r in cm.iterrows():
                flows.append(
                    {
                        "stage": "complexity → model",
                        "from": str(r["complexity_level"]),
                        "to": str(r["routed_model"]),
                        "n": int(r["n"]),
                    }
                )
            st.markdown("**Routing flow** (Sankey-style data as table — no extra chart deps)")
            st.dataframe(pd.DataFrame(flows).sort_values("n", ascending=False), use_container_width=True, hide_index=True)

        st.markdown("**Model utilization**")
        if "routed_model" in df.columns:
            um = df["routed_model"].value_counts().reset_index()
            um.columns = ["model", "count"]
            ch_u = (
                alt.Chart(um)
                .mark_arc(innerRadius=50)
                .encode(
                    alt.Theta("count:Q", stack=True),
                    alt.Color("model:N"),
                    alt.Tooltip(["model", "count"]),
                )
                .properties(height=320, width=320, title="Share by model")
            )
            st.altair_chart(ch_u, use_container_width=True)

    with tab_d:
        st.markdown("**Latest request snapshot** (most recent event in filter)")
        last = df.iloc[-1]
        st.json(
            {
                "request_id": last.get("request_id"),
                "query_type": last.get("query_type"),
                "complexity": last.get("complexity_level"),
                "score": last.get("complexity_score"),
                "routed_model": last.get("routed_model"),
                "brain_would_be": last.get("brain_central_model"),
                "why (rationale)": (str(last.get("rationale"))[:500] + "…")
                if last.get("rationale") and len(str(last.get("rationale"))) > 500
                else last.get("rationale"),
                "router_confidence_proxy": last.get("router_confidence"),
                "feature_signals": last.get("complexity_features"),
                "est_savings_vs_brain_usd": last.get("est_savings_vs_brain_usd"),
            },
            expanded=False,
        )
        st.markdown("**Feature importance (absolute value, latest row)**")
        feats = last.get("complexity_features") or {}
        if isinstance(feats, dict) and feats:
            fv = pd.DataFrame([{"feature": k, "abs": abs(float(v))} for k, v in feats.items()])
            fv = fv.sort_values("abs", ascending=False)
            ch_f = (
                alt.Chart(fv)
                .mark_bar()
                .encode(
                    alt.X("abs:Q", title="|value|"),
                    alt.Y("feature:N", sort="-x", title=None),
                )
                .properties(height=min(400, 24 * len(fv)), title="Signal strength")
            )
            st.altair_chart(ch_f, use_container_width=True)

        st.markdown("**Replay: scrub events**")
        if len(df) > 1:
            ev_i = st.slider("Event index", 0, len(df) - 1, len(df) - 1)
        else:
            ev_i = 0
        row = df.iloc[ev_i]
        st.write(
            f"`{row.get('request_id')}` · {row.get('routed_model')} · "
            f"${row.get('routed_cost_usd')} · {row.get('routed_latency_ms')} ms"
        )

    with st.expander("Judge / presentation mode — compact KPI strip", expanded=False):
        st.metric("Total savings", f"${total_savings:.4f}", f"{pct_saved:.1f}%")
        st.metric("Requests analyzed", len(df))
        succ = success.astype(bool) if hasattr(success, "astype") else success
        st.metric("Success rate", f"{100.0 * float(succ.mean()):.1f}%")

    st.markdown("---")
    st.subheader("Full Groq catalogue vs reference token size")
    ref_in = int(in_tok.iloc[-1]) if len(in_tok) else 200
    ref_out = int(out_tok.iloc[-1]) if len(out_tok) else 200
    last_model = str(df.iloc[-1]["routed_model"]) if "routed_model" in df.columns and len(df) else ""
    selected_for_table = brain
    for m in ModelChoice:
        if m.value == last_model:
            selected_for_table = m
            break
    st.dataframe(
        pd.DataFrame(model_comparison_table(selected_for_table, ref_in, ref_out)),
        use_container_width=True,
        hide_index=True,
    )
