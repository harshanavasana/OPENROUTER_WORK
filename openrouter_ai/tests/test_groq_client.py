"""Unit tests for Groq OpenAI-compatible client (no network)."""

import os
import sys

_repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, _repo_root)

import pytest
from unittest.mock import patch, MagicMock

from openrouter_ai.utils.groq_client import groq_chat_completion, _chat_url


def test_groq_chat_completion_parses_message_content():
    mock_resp = MagicMock()
    mock_resp.raise_for_status = MagicMock()
    mock_resp.json.return_value = {
        "choices": [
            {"message": {"role": "assistant", "content": "  hello world  "}},
        ]
    }
    with patch("openrouter_ai.utils.groq_client.requests.post", return_value=mock_resp) as post:
        out = groq_chat_completion(
            "fake-key",
            "llama-3.1-8b-instant",
            [{"role": "user", "content": "hi"}],
            max_tokens=64,
            temperature=0.0,
        )
    assert out == "hello world"
    args, kwargs = post.call_args
    assert kwargs["json"]["model"] == "llama-3.1-8b-instant"
    assert kwargs["headers"]["Authorization"] == "Bearer fake-key"


def test_groq_chat_completion_requires_key():
    with pytest.raises(ValueError, match="GROQ_API_KEY"):
        groq_chat_completion("", "llama-3.1-8b-instant", [{"role": "user", "content": "x"}])


def test_chat_url_suffix():
    assert _chat_url().endswith("/chat/completions")


@pytest.mark.skipif(not os.getenv("GROQ_API_KEY"), reason="Set GROQ_API_KEY for live Groq smoke test")
def test_groq_live_llama_smoke():
    """Calls Groq with Llama 3.1 8B (requires network + valid key)."""
    out = groq_chat_completion(
        os.environ["GROQ_API_KEY"],
        "llama-3.1-8b-instant",
        [{"role": "user", "content": "Reply with exactly the word: pong"}],
        max_tokens=32,
        temperature=0.0,
        timeout=60,
    )
    assert len(out) > 0
