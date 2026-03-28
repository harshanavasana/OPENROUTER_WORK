"""
Groq Cloud chat via the OpenAI-compatible endpoint (Llama / Meta models only in this project).

Docs: https://console.groq.com/docs/openai
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, List, Dict, Optional

import requests

DEFAULT_GROQ_BASE = "https://api.groq.com/openai/v1"


@dataclass(frozen=True)
class GroqChatResult:
    text: str
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None


def _chat_url() -> str:
    base = os.getenv("GROQ_OPENAI_BASE", DEFAULT_GROQ_BASE).rstrip("/")
    return f"{base}/chat/completions"


def groq_chat_completion_full(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout: int = 120,
) -> GroqChatResult:
    """
    POST chat/completions. Returns text and token usage when the API includes it.

    ``model`` must be a Groq model id (e.g. llama-3.1-8b-instant).
    """
    if not api_key:
        raise ValueError("GROQ_API_KEY is required for Groq calls")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
    }
    payload: Dict[str, Any] = {
        "model": model,
        "messages": messages,
        "temperature": temperature,
        "max_tokens": max_tokens,
    }

    resp = requests.post(_chat_url(), json=payload, headers=headers, timeout=timeout)
    if not resp.ok:
        detail = (resp.text or "")[:800]
        raise RuntimeError(
            f"Groq chat API HTTP {resp.status_code} for model {model!r}. "
            f"Body: {detail}"
        )
    data = resp.json()

    choices = data.get("choices") if isinstance(data, dict) else None
    if not choices:
        raise RuntimeError(f"Unexpected Groq response (no choices): {repr(data)[:500]}")

    choice0 = choices[0]
    msg = choice0.get("message") if isinstance(choice0, dict) else None
    text_out: Optional[str] = None
    if isinstance(msg, dict) and msg.get("content") is not None:
        text_out = str(msg["content"]).strip()

    if text_out is None and isinstance(choice0, dict) and choice0.get("text"):
        text_out = str(choice0["text"]).strip()

    if text_out is None:
        raise RuntimeError(f"Unexpected Groq choice shape: {choice0!r}")

    pt: Optional[int] = None
    ct: Optional[int] = None
    usage = data.get("usage") if isinstance(data, dict) else None
    if isinstance(usage, dict):
        if usage.get("prompt_tokens") is not None:
            pt = int(usage["prompt_tokens"])
        if usage.get("completion_tokens") is not None:
            ct = int(usage["completion_tokens"])

    return GroqChatResult(text=text_out, prompt_tokens=pt, completion_tokens=ct)


def groq_chat_completion(
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    *,
    max_tokens: int = 1024,
    temperature: float = 0.2,
    timeout: int = 120,
) -> str:
    """POST chat/completions; assistant text only (see ``groq_chat_completion_full`` for usage)."""
    return groq_chat_completion_full(
        api_key,
        model,
        messages,
        max_tokens=max_tokens,
        temperature=temperature,
        timeout=timeout,
    ).text
