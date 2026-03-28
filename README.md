# AI Router

Simple AI routing reference project with modules for classification, optimization, routing, execution, and logging.

## Structure

- `app/main.py` - orchestrator
- `app/classifier.py` - intent classification
- `app/optimizer.py` - request optimization
- `app/router.py` - route selection
- `app/executor.py` - placeholder route execution
- `app/logger.py` - simple JSON event logger
- `app/data/logs.json` - runtime logs
- `app/ui/streamlit_app.py` - Streamlit interface

## Setup

1. Create and activate a venv:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run CLI router:

```powershell
python app\main.py
```

4. Run Streamlit app:

```powershell
streamlit run app\ui\streamlit_app.py
```

## Notes

- `app/logger.py` writes to `C:/ai-router/app/data/logs.json`; adjust path for other install locations.
- This is a starter template; integrate with real NLP models and routing backends as needed.
