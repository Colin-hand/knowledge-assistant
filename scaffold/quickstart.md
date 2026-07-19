# Quickstart — Internal Knowledge Assistant (POC)

Access-aware RAG over MCP: Streamlit UI → FastAPI → agent (intent → compress → generate) → Knowledge MCP server → Pinecone. Full design: [doc/implementation/implementation.md](doc/implementation/implementation.md).

## 1. Setup

Requires **Python 3.12**

```bash
cd scaffold

# create & activate an environment, e.g. with venv:
python3.12 -m venv .venv && source .venv/bin/activate

pip install -r requirements.txt        # runtime dependencies
pip install -r requirements-dev.txt    # dev-only: pytest, ruff (skip if just running the app)
pip install -e .                       # installs src/knowledge_assistant as an importable package
```

Copy `.env.example` → `.env` and fill in:

- `OPENAI_API_KEY` (provided key)
- `PINECONE_API_KEY` (free Starter account — the index is created automatically on first ingestion)

## 2. Ingest the knowledge base

```bash
python -m knowledge_assistant.ingestion.pipeline            # incremental
python -m knowledge_assistant.ingestion.pipeline --rebuild  # from scratch
```

Prints a report (ingested / skipped / failed / cost). Only manifest-listed PDFs are ingested — the manifest is the ACL authority.

## 3. Run

```bash
uvicorn knowledge_assistant.api.main:app --port 8000        # terminal 1
streamlit run frontend/app.py                               # terminal 2
```

Pick a user in the sidebar (sets the bearer token) and ask away. Try:

- *"What's the current Growth plan price?"* as Sam → conflict flag ($79 playbook vs $99 pricing sheet)
- *"What's our brand tagline?"* as Maria → cites Brand Guidelines **v3**, not the archived v2
- *"What are the compensation bands?"* as Maria → uniform "no accessible information" (no leak)
- the exec-only all-hands paragraph → answerable only as Erin

## 4. Tests & evaluation

```bash
pytest                                                            # offline; includes the access-control suite

# question pool (needs .env):
python -m knowledge_assistant.evaluation.question_gen             # generate for docs with no questions yet
python -m knowledge_assistant.evaluation.question_gen --all       # regenerate the whole pool
python -m knowledge_assistant.evaluation.question_gen --doc u_maria/brand-guidelines-v3.pdf

# evaluation (needs .env + ingested index + question pool):
python -m knowledge_assistant.evaluation.evaluate                 # full pool
python -m knowledge_assistant.evaluation.evaluate --num-questions 20 --seed 42   # random sample
python -m knowledge_assistant.evaluation.evaluate --mode e2e --num-questions 10   # simulate real users through the full agent path (costly)
```

Question bank lives in `eval/question_bank/` (committed); the UI offers it as suggested questions per user. The **📊 Evaluation Dashboard** page in the Streamlit app visualizes the latest run. The eval writes `eval/runs/`: `report.json` (hit-rate@k, MRR, reversed-relevancy mean, access-control pass/fail, cost), `results.json` (per-query detail for visualization), and `reversed_questions.json`. It exits non-zero on any access leak.

Logs (JSONL, with per-call token/cost/latency telemetry): `logs/app.jsonl`.
