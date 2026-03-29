import sys
from pathlib import Path
import warnings

# Repo root must be on sys.path
_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

# Ignore scikit-learn warnings for cleaner output
warnings.filterwarnings('ignore')

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from openrouter_ai.pipeline import run_pipeline
import numpy as np

class MemoryRAG:
    """In-memory VectorDB using TF-IDF for lightweight continuous chat."""
    def __init__(self):
        self.documents = []  # Stores formatted strings of past exchanges
        self.vectorizer = TfidfVectorizer(stop_words='english')
    
    def add_exchange(self, user_text: str, ai_response: str):
        # We store just enough to remember the code or reasoning
        doc = f"User asked: {user_text}\nAI answered: {ai_response}"
        self.documents.append(doc)
        
    def retrieve(self, query: str, top_k: int = 1, threshold: float = 0.05) -> str:
        if not self.documents:
            return ""
        
        try:
            # Fit-transform all documents + the query
            tfidf_matrix = self.vectorizer.fit_transform(self.documents + [query])
            
            # Cosine similarity between query (last row) and all documents (all previous rows)
            cosine_similarities = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1]).flatten()
            
            # Find best matches
            best_indices = np.argsort(cosine_similarities)[::-1]
            
            retrieved_context = []
            for idx in best_indices[:top_k]:
                if cosine_similarities[idx] > threshold:
                    retrieved_context.append(self.documents[idx])
                    
            if retrieved_context:
                return "\n\n---\n".join(retrieved_context)
            return ""
        except ValueError:
            # Vectorizer can throw ValueError if the prompt has no recognizable words
            return ""

def main():
    print("==================================================")
    print(" AI Router — Continuous Chat with In-Memory RAG   ")
    print("==================================================")
    print("Type 'quit' or 'exit' to end the session.")
    print("This script clears memory when restarted and")
    print("does NOT alter your regular dashboard.")
    print("==================================================\n")
    
    rag_db = MemoryRAG()
    
    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye.")
            break
            
        if user_input.lower() in ("quit", "exit", ""):
            if user_input.lower() in ("quit", "exit"):
                break
            continue
            
        # 1. Retrieve any relevant context from RAG DB
        context = rag_db.retrieve(user_input, top_k=1, threshold=0.01)
        
        # 2. Build the router prompt
        if context:
            print("\n  [RAG System]: Found relevant past conversation. Injecting context...\n  " + "-"*48)
            final_prompt = (
                "You are an AI assistant in a continuous conversation. "
                "Use the following retrieved past conversation context if it helps answer the user's new question:\n\n"
                f"--- PAST CONTEXT ---\n{context[:1000]}...\n--- END PAST CONTEXT ---\n\n"
                f"User Question: {user_input}"
            )
        else:
            final_prompt = user_input
            
        # 3. Send to pipeline
        try:
            print("  [Router System]: Analyzing complexity and routing request to optimal LLM model...")
            result = run_pipeline(final_prompt)
            
            model_used = result.decision.selected_model.value
            ai_reply = result.response_text.strip()
            
            print(f"\nAI ({model_used}):")
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━")
            print(ai_reply)
            print("━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\n")
            
            # 4. Save this exchange to our Vector DB for future retrieval
            rag_db.add_exchange(user_input, ai_reply)
            
        except Exception as e:
            print(f"\nError processing request: {e}\n")

if __name__ == "__main__":
    main()
