"""
openrouter_ai/agents/executor_agent.py

LLM Executor — Stage 4 of the pipeline.

Takes the RoutingDecision and the optimized prompt, calls Groq-hosted Llama models
via the OpenAI-compatible chat endpoint, and returns the response with token counts
and latency.

If the call fails (rate limit / API error), the failure is raised for the caller to handle.
"""

from __future__ import annotations

import asyncio
import time
from functools import partial
from typing import Dict, List, Optional

import tiktoken  # type: ignore[import-untyped]

from openrouter_ai.models import ModelChoice, RoutingDecision
from openrouter_ai.utils.groq_client import groq_chat_completion_full


class ExecutorAgent:
    """
    Executes the selected LLM call using Groq (Meta Llama models).

    Args:
        groq_api_key: Groq API key
    """

    def __init__(self, groq_api_key: str):
        self._groq_key = groq_api_key
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _build_messages(
        self,
        user_prompt: str,
        system_prompt: Optional[str],
    ) -> List[Dict[str, str]]:
        messages: List[Dict[str, str]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages

    def _token_counts_fallback(self, messages: List[Dict[str, str]], response_text: str) -> tuple[int, int]:
        prompt_for_count = "\n\n".join(m["content"] for m in messages)
        input_tokens = len(self._enc.encode(prompt_for_count))
        output_tokens = len(self._enc.encode(response_text))
        return input_tokens, output_tokens

    async def execute(
        self,
        decision: RoutingDecision,
        system_prompt: Optional[str] = None,
    ) -> dict:
        """
        Async execution with Groq chat completions.

        The HTTP request runs in a thread pool so this coroutine does not block the event loop.

        Returns a dict:
          {
              "response_text": str,
              "model_used":    ModelChoice,
              "latency_ms":    float,
              "input_tokens":  int,
              "output_tokens": int,
          }
        """
        if not self._groq_key:
            raise ValueError("GROQ_API_KEY is required for Groq calls")

        user_text = decision.optimized_prompt.optimized
        messages = self._build_messages(user_text, system_prompt)
        model_id = decision.selected_model.value

        t0 = time.perf_counter()
        # partial avoids brittle *args/**kwargs forwarding to to_thread across Python versions
        call = partial(
            groq_chat_completion_full,
            self._groq_key,
            model_id,
            messages,
            max_tokens=1024,
            temperature=0.3,
            timeout=120,
        )
        result = await asyncio.to_thread(call)
        latency_ms = (time.perf_counter() - t0) * 1000

        fb_in, fb_out = self._token_counts_fallback(messages, result.text)
        input_tokens = result.prompt_tokens if result.prompt_tokens is not None else fb_in
        output_tokens = result.completion_tokens if result.completion_tokens is not None else fb_out

        return {
            "response_text": result.text.strip(),
            "model_used": decision.selected_model,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    async def execute_for_model(
        self,
        user_prompt: str,
        model: ModelChoice,
        system_prompt: Optional[str] = None,
        *,
        max_tokens: int = 1024,
        temperature: float = 0.3,
    ) -> dict:
        """Run Groq chat with an explicit model (e.g. live baseline / A-B)."""
        if not self._groq_key:
            raise ValueError("GROQ_API_KEY is required for Groq calls")

        messages = self._build_messages(user_prompt, system_prompt)
        model_id = model.value

        t0 = time.perf_counter()
        call = partial(
            groq_chat_completion_full,
            self._groq_key,
            model_id,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=120,
        )
        result = await asyncio.to_thread(call)
        latency_ms = (time.perf_counter() - t0) * 1000

        fb_in, fb_out = self._token_counts_fallback(messages, result.text)
        input_tokens = result.prompt_tokens if result.prompt_tokens is not None else fb_in
        output_tokens = result.completion_tokens if result.completion_tokens is not None else fb_out

        return {
            "response_text": result.text.strip(),
            "model_used": model,
            "latency_ms": round(latency_ms, 1),
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
        }

    def execute_sync(self, decision: RoutingDecision, system_prompt: Optional[str] = None) -> dict:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(self.execute(decision, system_prompt))
        raise RuntimeError(
            "ExecutorAgent.execute_sync() cannot be called from inside a running event loop; "
            "await execute() instead."
        )
