import asyncio
import os
import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv
load_dotenv(Path(__file__).resolve().parent / ".env")

from openrouter_ai.pipeline import OpenRouterPipeline
from openrouter_ai.models import RoutingRequest, ModelChoice
from openrouter_ai.router.smart_router import brain_central_model, _MODEL_CATALOGUE
from openrouter_ai.utils.groq_client import groq_chat_completion_full

async def ask_llm_router(prompt: str, api_key: str, brain_model: str) -> ModelChoice:
    """Asks the brain LLM to act as a router and pick the best model for the prompt."""
    
    catalog_str = "\n".join([f"- {m.value}: quality={_MODEL_CATALOGUE[m].quality_score}, speed={_MODEL_CATALOGUE[m].avg_latency_ms}ms" for m in ModelChoice if m != ModelChoice.EDGE_LOCAL_LLAMA3])
    
    sys_prompt = (
        "You are an AI Routing Agent. Your job is to select the absolute best model strictly from the provided Catalogue to answer the User Prompt.\n"
        f"Catalogue:\n{catalog_str}\n\n"
        "Consider the complexity of the prompt. Simple prompts should use faster/weaker models. Complex prompts should use higher quality models.\n"
        "REPLY ONLY WITH THE EXACT MODEL ID STRING from the catalogue. No other text."
    )
    
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": f"User Prompt: {prompt}"}]
    
    try:
        response = await asyncio.to_thread(groq_chat_completion_full, api_key, brain_model, messages, max_tokens=50, temperature=0.1)
        choice_str = response.text.strip().strip('"').strip("'")
        
        # Verify choice
        for m in ModelChoice:
            if m.value == choice_str:
                return m
        
        # Fallback if hallucinated
        print(f"  [LLM Router Warning] Hallucinated/invalid model string: {choice_str}. Falling back to brain model.")
        return ModelChoice.LLAMA3_70B_8192
    except Exception as e:
        print(f"Error in LLM Router: {e}")
        return ModelChoice.LLAMA3_70B_8192

async def judge_results(prompt: str, ans_a: str, ans_b: str, api_key: str, brain_model: str) -> str:
    """Uses the central brain LLM to blind-judge the two results."""
    sys_prompt = (
        "You are an impartial, expert evaluator of AI responses. "
        "You will be given a User Prompt, and two answers: [Answer A] and [Answer B].\n"
        "Your job is to determine which answer is better (more accurate, helpful, and concise).\n"
        "First, write a 1-sentence rationale. Second, output the winner exactly as 'WINNER: A' or 'WINNER: B' (or 'WINNER: TIE')."
    )
    user_text = f"User Prompt: {prompt}\n\n[Answer A]:\n{ans_a}\n\n[Answer B]:\n{ans_b}\n"
    
    messages = [{"role": "system", "content": sys_prompt}, {"role": "user", "content": user_text}]
    
    try:
        response = await asyncio.to_thread(groq_chat_completion_full, api_key, brain_model, messages, max_tokens=256, temperature=0.3)
        return response.text.strip()
    except Exception as e:
        return f"Judging error: {e}"

async def main():
    print("==================================================")
    print(" 🥊 ML Router vs LLM Router: Head-to-Head Evaluator")
    print("==================================================")
    
    print("Ready to compare the speed/cost efficient ML Router against a brilliant but slow LLM Router.")
    prompt = input("\nEnter a prompt to test: ").strip()
    if not prompt:
        print("Empty prompt. Exiting.")
        return

    api_key = os.getenv("GROQ_API_KEY", "")
    if not api_key:
        print("Missing GROQ_API_KEY in .env")
        return
        
    brain_model = brain_central_model().value
    pipeline = OpenRouterPipeline()
    
    print("\n[1/3] Running ML Router (Stage 1)...")
    # Contestant A (ML) - Using default (balanced) preference so it has full capability
    req = RoutingRequest(prompt=prompt, prefer_cost=False, prefer_speed=False) 
    ml_result = await pipeline.run(req)
    ml_choice = ml_result.decision.selected_model
    ans_a = ml_result.response_text
    print(f"  → ML Router Picked: {ml_choice.value}")
    
    print("\n[2/3] Running LLM Router (Stage 2)...")
    # Contestant B (LLM)
    llm_choice = await ask_llm_router(prompt, api_key, brain_model)
    print(f"  → LLM Router Picked: {llm_choice.value}")
    
    if llm_choice == ml_choice:
        print("\n🏆 WOW! Both routers picked the exact same model! No need to fight.")
        return
        
    print(f"  → Executing LLM Router's choice ({llm_choice.value})...")
    llm_exec_result = await pipeline.executor.execute_for_model(prompt, llm_choice)
    ans_b = llm_exec_result["response_text"]
    
    print("\n[3/3] Blind Judging (Stage 3)...")
    verdict = await judge_results(prompt, ans_a, ans_b, api_key, brain_model)
    
    print("\n================== RESULTS ==================")
    print(f"Contestant A (ML Router): {ml_choice.value}")
    print(f"Contestant B (LLM Router): {llm_choice.value}")
    print("---------------------------------------------")
    print("The Judge (Llama 3.3 70B) says:")
    print(verdict)
    print("=============================================\n")

if __name__ == "__main__":
    asyncio.run(main())
