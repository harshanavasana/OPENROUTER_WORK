"""Load .env from openrouter_ai/ so integration tests see GROQ_API_KEY when present."""

import os

try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None

if load_dotenv:
    _env_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    load_dotenv(os.path.join(_env_dir, ".env"))
