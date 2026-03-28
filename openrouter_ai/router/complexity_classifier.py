"""
openrouter_ai/router/complexity_classifier.py

Complexity Classifier — Stage 2 of the pipeline.

Uses a two-layer approach:
  a) Fast heuristic feature extraction (no LLM call)
  b) A trained sklearn RandomForest for the final score

Feature vector (14 features):
  - token_count              : number of tokens in the optimized prompt
  - sentence_count           : number of sentences
  - avg_sentence_len         : avg tokens per sentence
  - question_count           : number of "?" characters
  - has_code_block           : 1 if ``` present
  - has_math                 : 1 if LaTeX-style or equations detected
  - has_multi_step           : 1 if "step", "then", "finally", "next" detected
  - has_comparison           : 1 if "compare", "difference", "vs" detected
  - has_analysis             : 1 if "analyse", "evaluate", "critique" detected
  - has_creative             : 1 if "write a", "create", "generate" detected
  - has_reasoning            : 1 if "why", "explain", "how does" detected
  - has_factual              : 1 if "what is", "who is", "when" detected
  - max_word_len             : longest word length (proxy for domain specificity)
  - unique_word_ratio        : vocabulary richness
"""

import re
import os
import json
import pickle
import requests
import numpy as np

from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder

from openrouter_ai.models import ComplexityLevel, ComplexityScore


# ── Keyword sets ──────────────────────────────────────────────────────────────
_MULTI_STEP = {"step", "then", "finally", "next", "after", "before", "first", "second", "third"}
_COMPARISON = {"compare", "comparison", "difference", "vs", "versus", "contrast", "better"}
_ANALYSIS   = {"analyse", "analyze", "evaluate", "critique", "assess", "review", "audit"}
_CREATIVE   = {"write", "create", "generate", "compose", "draft", "design", "build", "make"}
_REASONING  = {"why", "explain", "how does", "how do", "reason", "cause", "effect", "mechanism"}
_FACTUAL    = {"what is", "what are", "who is", "who are", "when", "where", "define"}


def _extract_features(text: str) -> np.ndarray:
    """Extract the 14-element feature vector from a prompt string."""
    words = text.lower().split()
    sentences = re.split(r'[.!?]+', text.strip())
    sentences = [s for s in sentences if s.strip()]

    token_count       = len(words)
    sentence_count    = max(len(sentences), 1)
    avg_sentence_len  = token_count / sentence_count
    question_count    = text.count("?")
    has_code_block    = int("```" in text)
    has_math          = int(bool(re.search(r'[$\\][a-zA-Z{]|=\s*\d|∑|∫|√', text)))
    has_multi_step    = int(bool(_MULTI_STEP & set(words)))
    has_comparison    = int(bool(_COMPARISON & set(words)))
    has_analysis      = int(bool(_ANALYSIS & set(words)))
    has_creative      = int(bool(_CREATIVE & set(words)))
    has_reasoning     = int(any(p in text.lower() for p in _REASONING))
    has_factual       = int(any(p in text.lower() for p in _FACTUAL))
    max_word_len      = max((len(w) for w in words), default=0)
    unique_word_ratio = len(set(words)) / max(token_count, 1)

    return np.array([
        token_count, sentence_count, avg_sentence_len, question_count,
        has_code_block, has_math, has_multi_step, has_comparison,
        has_analysis, has_creative, has_reasoning, has_factual,
        max_word_len, unique_word_ratio,
    ], dtype=float)


def _heuristic_score(features: np.ndarray) -> float:
    """
    Rule-based fallback score (0–1) when the ML model is unavailable.
    Combines normalised token count with binary feature flags.
    """
    token_count = features[0]
    flags = features[4:]          # binary indicators start at index 4
    base  = min(token_count / 300, 1.0) * 0.4
    bonus = float(np.sum(flags)) / len(flags) * 0.6
    return round(base + bonus, 4)


class ComplexityClassifier:
    """
    Classifies prompt complexity into SIMPLE / MEDIUM / COMPLEX.

    On first use, if no trained model file exists, it trains a small
    synthetic RandomForest and persists it next to this file.  In
    production, replace the synthetic training data with real labelled
    examples from your usage logs.
    """

    MODEL_PATH = os.path.join(os.path.dirname(__file__), "complexity_rf.pkl")

    SIMPLE_THRESH  = float(os.getenv("SIMPLE_COMPLEXITY_THRESHOLD",  "0.35"))
    COMPLEX_THRESH = float(os.getenv("COMPLEX_COMPLEXITY_THRESHOLD", "0.70"))

    def __init__(self, groq_api_key: str | None = None):
        # groq_api_key is accepted for API compatibility; classification is local (sklearn).
        _ = groq_api_key
        self._model: RandomForestClassifier | None = None
        self._le = LabelEncoder()
        self._le.fit(["simple", "medium", "complex"])
        self._load_or_train()

    # ── Training ───────────────────────────────────────────────────────────────

    def _synthetic_training_data(self):
        """
        Minimal synthetic dataset.  Replace / augment with real labelled data.
        Each tuple: (prompt_text, label)
        """
        examples = [
            # simple
            ("What is the capital of France?", "simple"),
            ("Translate hello to Spanish.", "simple"),
            ("What does API stand for?", "simple"),
            ("Give me a synonym for happy.", "simple"),
            ("What time is it in Tokyo?", "simple"),
            ("Define machine learning.", "simple"),
            ("Who wrote Hamlet?", "simple"),
            ("Convert 100 USD to EUR.", "simple"),
            # medium
            ("Explain how HTTPS works.", "medium"),
            ("Compare REST and GraphQL APIs.", "medium"),
            ("Write a Python function to reverse a string.", "medium"),
            ("Summarise this paragraph: The quick brown fox...", "medium"),
            ("What are the pros and cons of microservices?", "medium"),
            ("Explain the difference between SQL and NoSQL databases.", "medium"),
            ("How does TCP/IP work?", "medium"),
            ("Write a cover letter for a software engineer role.", "medium"),
            # complex
            ("Analyse the ethical implications of AI in healthcare and provide a framework for governance.", "complex"),
            ("Write a distributed rate-limiter in Go using Redis with a sliding window algorithm.", "complex"),
            ("Compare transformer architectures across GPT-4, Claude, and Gemini in detail.", "complex"),
            ("Critique the following research paper methodology and suggest improvements: ...", "complex"),
            ("Design a multi-region, fault-tolerant data pipeline for 1M events/sec.", "complex"),
            ("Explain the mathematical intuition behind backpropagation step by step.", "complex"),
            ("Write a business case for migrating our monolith to microservices with ROI analysis.", "complex"),
            ("Create a comprehensive study plan for passing the AWS Solutions Architect exam.", "complex"),
        ]
        X = np.array([_extract_features(p) for p, _ in examples])
        y = np.array([label for _, label in examples])
        return X, y

    def _train_and_save(self):
        X, y = self._synthetic_training_data()
        self._model = RandomForestClassifier(n_estimators=100, random_state=42)
        self._model.fit(X, y)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)

    def _load_or_train(self):
        if os.path.exists(self.MODEL_PATH):
            with open(self.MODEL_PATH, "rb") as f:
                self._model = pickle.load(f)
        else:
            self._train_and_save()

    def retrain(self, X: np.ndarray, y: np.ndarray):
        """
        Online retraining hook — call this from the feedback loop
        with accumulated labelled examples.
        """
        self._model = RandomForestClassifier(n_estimators=200, random_state=42)
        self._model.fit(X, y)
        with open(self.MODEL_PATH, "wb") as f:
            pickle.dump(self._model, f)

    # ── Inference ──────────────────────────────────────────────────────────────

    def classify(self, prompt: str) -> ComplexityScore:
        features = _extract_features(prompt)

        if self._model is not None:
            proba = self._model.predict_proba([features])[0]
            classes = self._model.classes_
            # Build a weighted score: simple=0, medium=0.5, complex=1
            weight_map = {"simple": 0.0, "medium": 0.5, "complex": 1.0}
            score = sum(proba[i] * weight_map.get(cls, 0.5) for i, cls in enumerate(classes))
        else:
            score = _heuristic_score(features)

        score = float(np.clip(score, 0.0, 1.0))

        if score < self.SIMPLE_THRESH:
            level = ComplexityLevel.SIMPLE
        elif score < self.COMPLEX_THRESH:
            level = ComplexityLevel.MEDIUM
        else:
            level = ComplexityLevel.COMPLEX

        reasoning = self._build_reasoning(features, level, score)

        return ComplexityScore(
            level=level,
            score=round(score, 4),
            reasoning=reasoning,
            features={
                "token_count":       int(features[0]),
                "has_code_block":    bool(features[4]),
                "has_math":          bool(features[5]),
                "has_multi_step":    bool(features[6]),
                "has_comparison":    bool(features[7]),
                "has_analysis":      bool(features[8]),
                "has_creative":      bool(features[9]),
                "has_reasoning":     bool(features[10]),
                "has_factual":       bool(features[11]),
                "unique_word_ratio": round(float(features[13]), 3),
            },
        )

    @staticmethod
    def _build_reasoning(features, level: ComplexityLevel, score: float) -> str:
        parts = [f"Complexity score {score:.2f} → {level.value}."]
        if features[4]: parts.append("Code block detected.")
        if features[5]: parts.append("Mathematical notation detected.")
        if features[6]: parts.append("Multi-step instructions detected.")
        if features[7]: parts.append("Comparison/contrast task detected.")
        if features[8]: parts.append("Analysis/evaluation task detected.")
        if features[9]: parts.append("Creative generation task detected.")
        if features[0] > 150: parts.append(f"Long prompt ({int(features[0])} tokens).")
        return " ".join(parts)