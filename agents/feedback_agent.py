"""
openrouter_ai/agents/feedback_agent.py
Feedback Agent — Analyzes past executions to find routing mistakes 
and refines the Complexity Classifier using TOON.
"""
import asyncio
import numpy as np
from openrouter_ai.utils.toon_parser import ToonParser
from openrouter_ai.utils.groq_client import groq_chat_completion_full
from openrouter_ai.router.complexity_classifier import ComplexityClassifier, _extract_features

class FeedbackAgent:
    """
    Background worker that samples routing logs, asks Groq Llama for ground-truth
    scores using TOON, and accumulates data for RF retraining.
    """
    def __init__(self, groq_api_key: str, classifier: ComplexityClassifier):
        self._key = groq_api_key
        self._classifier = classifier
        self._history = []

    async def evaluate_and_learn(self, prompt: str, generated_text: str):
        """Asynchronously call the LLM to verify routing metadata."""
        sys_prompt = (
            "You are an AI Routing Judge. Rate the user prompt's complexity as simple, medium, or complex.\n"
            "Respond STRICTLY using TOON format with standard Level key.\n"
            "Example:\nKey | Value\n--- | ---\nLevel | simple"
        )
        
        toon_input = ToonParser.encode({
            "Prompt": prompt[:300],
            "ResponseLength": len(generated_text)
        })
        
        messages = [
            {"role": "system", "content": sys_prompt},
            {"role": "user", "content": toon_input}
        ]
        
        try:
            # Note: Running this via `to_thread` the same way Executor uses Groq APIs
            result = await asyncio.to_thread(
                groq_chat_completion_full,
                self._key,
                "llama-3.1-8b-instant",  # fast cheap model for verification
                messages,
                max_tokens=64,
                temperature=0.0,
                timeout=15,
            )
            data = ToonParser.decode(result.text)
            ground_truth = data.get("Level", "").lower()
            if ground_truth in ["simple", "medium", "complex"]:
                self._accumulate_and_retrain(prompt, ground_truth)
        except Exception as e:
            # We silently swallow background feedback errors (like rate limits)
            pass

    def _accumulate_and_retrain(self, prompt: str, label: str):
        self._history.append((prompt, label))
        
        # In a real app this might be 1000s; we use 5 for demonstration so it triggers fast
        if len(self._history) >= 5:
            # Generate the features
            X = np.array([_extract_features(p) for p, _ in self._history])
            y = np.array([l for _, l in self._history])
            
            # Re-train the RF
            self._classifier.retrain(X, y)
            print(f"[FeedbackAgent] Self-learned and retrained the router with {len(y)} new examples!")
            
            # Dump the history batch
            self._history.clear()
