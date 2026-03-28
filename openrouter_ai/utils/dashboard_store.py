"""
Append-only JSONL event log for the routing optimization dashboard.

Path: ``openrouter_ai/data/dashboard_events.jsonl``
Disable with env ``DASHBOARD_LOG=0``.
"""

from __future__ import annotations

import json
import os
import random
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_DEFAULT_FILE = _DATA_DIR / "dashboard_events.jsonl"


def events_path() -> Path:
    raw = os.getenv("DASHBOARD_EVENTS_FILE", "").strip()
    return Path(raw) if raw else _DEFAULT_FILE


def logging_enabled() -> bool:
    return os.getenv("DASHBOARD_LOG", "1").lower() not in ("0", "false", "no")


def append_event(record: Dict[str, Any]) -> None:
    if not logging_enabled():
        return
    path = events_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, default=str, ensure_ascii=False)
    with open(path, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_events(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    path = events_path()
    if not path.is_file():
        return []
    rows: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    if limit is not None:
        rows = rows[-limit:]
    return rows


def clear_events() -> None:
    path = events_path()
    if path.is_file():
        path.unlink()


def infer_query_type(features: Dict[str, Any], text: str) -> str:
    """Coarse bucket for heatmaps / filters."""
    if features.get("has_code_block"):
        return "code"
    if features.get("has_math"):
        return "math"
    if features.get("has_comparison") or features.get("has_analysis"):
        return "analysis"
    if features.get("has_creative"):
        return "creative"
    if features.get("has_factual"):
        return "factual"
    if features.get("has_reasoning"):
        return "reasoning"
    if len(text or "") > 800:
        return "long_context"
    return "general"


def routing_intelligence_score(
    complexity_level: str,
    routed_model: str,
    brain_model: str,
    success: bool,
    savings_usd: float,
) -> float:
    """
    0–100 heuristic: reward using cheaper-than-brain when complexity is low,
    and any positive savings on success.
    """
    base = 55.0
    if success:
        base += 25.0
    if savings_usd > 0:
        base += min(15.0, savings_usd * 5000)
    if complexity_level == "simple" and routed_model != brain_model:
        base += 5.0
    return max(0.0, min(100.0, base))


def seed_synthetic_events(n: int = 48) -> None:
    """Demo data for an empty dashboard (hackathon walkthrough)."""
    if not logging_enabled():
        return
    brain = "llama-3.3-70b-versatile"
    models = [
        "llama-3.1-8b-instant",
        "openai/gpt-oss-20b",
        "meta-llama/llama-4-scout-17b-16e-instruct",
        "openai/gpt-oss-120b",
        "qwen/qwen3-32b",
        brain,
    ]
    levels = ["simple", "medium", "complex"]
    qtypes = ["general", "code", "analysis", "factual", "reasoning"]
    now = time.time()
    for i in range(n):
        ts = now - (n - i) * 120 + random.uniform(-20, 20)
        lvl = random.choice(levels)
        routed = random.choice(models[:4] if lvl != "complex" else models[2:])
        in_tok = random.randint(80, 1200)
        out_tok = random.randint(120, 600)
        # catalog-ish costs
        def est(m: str, inn: int, out: int) -> float:
            rates = {
                "llama-3.1-8b-instant": (0.00005, 0.00008),
                "openai/gpt-oss-20b": (0.000075, 0.00030),
                "meta-llama/llama-4-scout-17b-16e-instruct": (0.00011, 0.00034),
                "openai/gpt-oss-120b": (0.00015, 0.00060),
                "qwen/qwen3-32b": (0.00029, 0.00059),
                brain: (0.00059, 0.00079),
            }
            ri, ro = rates.get(m, rates[brain])
            return inn / 1000 * ri + out / 1000 * ro

        base_c = est(brain, in_tok, out_tok)
        route_c = est(routed, in_tok, out_tok)
        lat_r = random.uniform(400, 2200)
        lat_b = random.uniform(800, 3200)
        success = random.random() > 0.06
        append_event(
            {
                "ts": datetime.fromtimestamp(ts, tz=timezone.utc).isoformat(),
                "request_id": f"syn-{i}",
                "query_type": random.choice(qtypes),
                "complexity_level": lvl,
                "complexity_score": {"simple": 0.25, "medium": 0.55, "complex": 0.82}[lvl]
                + random.uniform(-0.05, 0.05),
                "routed_model": routed,
                "brain_central_model": brain,
                "routed_cost_usd": round(route_c, 8),
                "routed_latency_ms": round(lat_r, 1),
                "baseline_est_cost_usd": round(base_c, 8),
                "baseline_est_latency_ms": round(lat_b, 1),
                "est_savings_vs_brain_usd": round(base_c - route_c, 8),
                "est_latency_delta_vs_brain_ms": round(lat_r - lat_b, 1),
                "success": success,
                "input_tokens": in_tok,
                "output_tokens": out_tok,
                "compression_ratio": random.uniform(0.0, 0.25),
                "router_confidence": random.uniform(0.5, 0.95),
                "routing_intelligence_score": random.uniform(60, 95),
                "synthetic": True,
            }
        )
