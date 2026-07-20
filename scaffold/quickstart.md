# Quickstart

Requires **Python 3.12** and two keys: `OPENAI_API_KEY`, `PINECONE_API_KEY` (free Starter tier).

## 1. Install

```bash
cd scaffold
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
cp .env.example .env    # then fill in the two API keys
```

## 2. Test (optional)

```bash
pip install -r requirements-dev.txt
pytest
```

Runs the offline suite in `tests/` (includes the access-control tests) — no API keys needed.

## 3. Build the knowledge base

```bash
python -m knowledge_assistant.ingestion.pipeline            # incremental: skips unchanged documents
python -m knowledge_assistant.ingestion.pipeline --rebuild  # full re-ingestion
```

Document content changed → plain run. Pipeline config changed (chunk size, overlap, embed model) → `--rebuild`

## 4. Run

```bash
uvicorn knowledge_assistant.api.main:app --port 8000    # terminal 1
streamlit run frontend/app.py                           # terminal 2
```

Sign in as a user in the sidebar and click a **Quick test** — each case notes who is entitled to see the answer.

## 5. Evaluate (optional)

```bash
python -m knowledge_assistant.evaluation.question_gen         # build the question pool
python -m knowledge_assistant.evaluation.evaluate             # metrics + access sweep
python -m knowledge_assistant.evaluation.evaluate --top-k 8   # one high-k run; the dashboard recomputes every smaller k
```

Results show on the app's **Evaluation Dashboard** page; per-request telemetry on **Request Details**. Logs: `logs/app.jsonl`.
