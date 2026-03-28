"""
openrouter_ai/agents/optimizer_agent.py

LLM Optimizer Agent — Stage 1 of the pipeline.

Receives the raw user prompt and uses a lightweight Groq-hosted Llama model
to produce a semantically equivalent but token-efficient version.
This "decompression" step removes redundancy, expands abbreviations that
confuse models, and strips filler language — saving cost on every
downstream call.

Architecture:
    User prompt
        │
        ▼
    OptimizerAgent (LangChain LLMChain)
        │
        ├─ returns: optimized prompt (str)
        └─ returns: OptimizedPrompt metadata
"""

import asyncio
import os
import tiktoken

from openrouter_ai.models import OptimizedPrompt
from openrouter_ai.utils.groq_client import groq_chat_completion

OPTIMIZER_SYSTEM = """You are a prompt optimization engine. Your ONLY job is to rewrite
the user's prompt so that it:
1. Removes all filler words, hedges, and redundant phrases
2. Makes the intent crystal-clear and unambiguous
3. Is as short as possible without losing any meaning
4. Expands unclear abbreviations that a language model might misread

Output ONLY the rewritten prompt — no explanations, no preamble, no quotes.
If the prompt is already optimal, output it unchanged."""

OPTIMIZER_HUMAN = "Optimize this prompt:\n\n{raw_prompt}"


class OptimizerAgent:
    """
    Stage-1 agent: rewrites the user prompt for maximum efficiency.

    Uses Groq model with temperature=0 for deterministic, reproducible rewrites.
    """

    def __init__(self, groq_api_key: str, model: str | None = None):
        self._groq_key = groq_api_key
        self._model = model or os.getenv("GROQ_OPTIMIZER_MODEL", "llama-3.1-8b-instant")
        self._enc = tiktoken.get_encoding("cl100k_base")

    def _count_tokens(self, text: str) -> int:
        return len(self._enc.encode(text))

    def identity_optimize(self, raw_prompt: str) -> OptimizedPrompt:
        """No API call — use raw text as the optimized prompt (fast path)."""
        t = self._count_tokens(raw_prompt)
        return OptimizedPrompt(
            original=raw_prompt,
            optimized=raw_prompt,
            compression_ratio=0.0,
            tokens_original=t,
            tokens_optimized=t,
        )

    def _groq_call(self, user_content: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
        messages = [
            {"role": "system", "content": OPTIMIZER_SYSTEM},
            {"role": "user", "content": user_content},
        ]
        return groq_chat_completion(
            self._groq_key,
            self._model,
            messages,
            max_tokens=max_tokens,
            temperature=temperature,
            timeout=90,
        )

    async def optimize(self, raw_prompt: str) -> OptimizedPrompt:
        """
        Async: returns an OptimizedPrompt with the rewritten text
        and compression metadata.
        """
        if not self._groq_key:
            raise ValueError("GROQ_API_KEY is required for Groq calls")

        # Skip extra latency for very short prompts (one fewer Groq round-trip).
        min_skip = int(os.getenv("GROQ_OPTIMIZER_SKIP_BELOW_TOKENS", "16"))
        if min_skip > 0 and self._count_tokens(raw_prompt.strip()) < min_skip:
            return self.identity_optimize(raw_prompt)

        instruction = "Rewrite the following user prompt (output only the rewritten text):\n\n" + raw_prompt

        optimized_text = await asyncio.to_thread(
            self._groq_call, instruction, 512, 0.0
        )
        optimized_text = optimized_text.strip()

        tokens_original = self._count_tokens(raw_prompt)
        tokens_optimized = self._count_tokens(optimized_text)

        if tokens_optimized >= tokens_original:
            optimized_text = raw_prompt
            tokens_optimized = tokens_original

        compression_ratio = (
            (tokens_original - tokens_optimized) / tokens_original
            if tokens_original > 0 else 0.0
        )

        return OptimizedPrompt(
            original=raw_prompt,
            optimized=optimized_text,
            compression_ratio=round(compression_ratio, 4),
            tokens_original=tokens_original,
            tokens_optimized=tokens_optimized,
        )

    def optimize_sync(self, raw_prompt: str) -> OptimizedPrompt:
        """Synchronous wrapper for non-async contexts."""
        import asyncio
        return asyncio.run(self.optimize(raw_prompt))