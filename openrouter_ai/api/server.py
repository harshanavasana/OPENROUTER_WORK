import sys
from pathlib import Path

_repo_root = Path(__file__).resolve().parent.parent.parent
if str(_repo_root) not in sys.path:
    sys.path.insert(0, str(_repo_root))

from dotenv import load_dotenv

load_dotenv(Path(__file__).resolve().parent.parent / ".env")

from fastapi import FastAPI
from openrouter_ai.pipeline import run_pipeline
from openrouter_ai.models import InputRequest

app = FastAPI()


@app.post("/route")
def route(request: InputRequest):
    result = run_pipeline(request.text)
    return result


@app.get("/")
def root():
    return {"status": "openrouter_ai running"}
