import sys
from pathlib import Path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from openrouter_ai.pipeline import run_pipeline

print("Running pipeline for DSA question...")
res = run_pipeline("Write a mathematically rigorous proof of Godel's First Incompleteness Theorem, and then apply its logical structure to formulate a paradox resolving the Halting problem in computer science. Ensure you use strict formal logic notation.")

print("\n\n=== RESULT ===")
print("COMPLEXITY:", res.decision.complexity.level.value)
print("COMPLEXITY SCORE:", res.decision.complexity.score)
print("SELECTED_MODEL:", res.decision.selected_model.value)
print("================\n")
