# Implementation Plan — Internal Knowledge Assistant (POC)

> Derived from `doc/solution_design/Solution_Architecture.png` and the challenge `README.md`.
> Grounded in the actual scaffold data: `data/users.json`, `data/pdfs/manifest.json`, 12 PDFs.

---

## 0. Global Decisions & Conventions

### 0.1 Runtime & tooling

| Concern | Decision |
|---|---|
| Backend language | Python 3.12 — any environment manager (venv, conda, uv); install via `requirements.txt` + `pip install -e .` |
| LLM provider | OpenAI — `OPENAI_MODEL=gpt-5.6-luna`, `OPENAI_EMBED_MODEL=text-embedding-3-small` |
| API layer | FastAPI + uvicorn between the Streamlit UI and the agent |
| Config | `pydantic-settings` reading `.env` (single source of truth; no scattered `os.getenv`) |
| Data contracts | Pydantic v2 models shared across all layers (`models.py`) |
| Observability | Structured logging + per-call cost/latency telemetry (§0.5); MLflow tracking is a planned next step |
| Lint/format | `ruff` (lint + format) — dev-only |
| Tests | `pytest`; access-control tests are mandatory, everything else best-effort for POC |
| Dependencies | `requirements.txt` contains **only packages actually imported at runtime** (§7); dev tools in `requirements-dev.txt` |

### 0.2 Project structure

```
scaffold/
├── frontend/
│   ├── app.py                      # Streamlit UI (thin — talks to FastAPI over HTTP only)
│   └── pages/
│       ├── 1_request_details.py        # per-request telemetry drill-down (reads logs/)
│       └── 2_evaluation_dashboard.py   # eval results dashboard (reads eval/runs/)
├── src/
│   └── knowledge_assistant/
│       ├── __init__.py
│       ├── config.py               # Settings (pydantic-settings)
│       ├── log.py                  # logging setup (trace-id aware)
│       ├── telemetry.py            # LLM call wrapper: tokens, cost, latency (§0.5)
│       ├── models.py               # User, DocumentMeta, Chunk, SearchResult, AgentAnswer, Citation
│       ├── api/
│       │   ├── main.py             # FastAPI app factory, middleware (trace id, error handler)
│       │   ├── routes.py           # POST /chat, GET /healthz
│       │   ├── schemas.py          # request/response models (reuse models.py types)
│       │   └── deps.py             # Bearer-token extraction dependency
│       ├── iam/
│       │   └── service.py          # token → User(roles) resolution from users.json
│       ├── ingestion/              # KB creation/update pipeline (§5)
│       │   ├── loader.py           # PDF → page-anchored, NFKC-normalized text
│       │   ├── markers.py          # confidential-marker regex (shared: chunker + enricher)
│       │   ├── chunker.py          # heading-aware section splitting
│       │   ├── enricher.py         # manifest metadata, is_global flag, exec-only chunk override
│       │   └── pipeline.py         # CLI: python -m knowledge_assistant.ingestion.pipeline
│       ├── vectorstore/
│       │   ├── base.py             # VectorStore protocol (swap-friendly)
│       │   ├── pinecone_store.py   # Pinecone serverless impl (§0.3)
│       │   └── memory_store.py     # in-memory impl, same filter semantics (tests)
│       ├── retrieval.py            # merged dual-scope ACL search (shared by MCP server + eval)
│       ├── mcp_server/
│       │   └── server.py           # FastMCP: search_knowledge
│       ├── agent/
│       │   ├── intent.py           # intent gate (classify + query rewrite)
│       │   ├── compressor.py       # per-chunk relevance extraction, parallel (§3.4)
│       │   ├── generator.py        # grounded, cited answer synthesis
│       │   ├── reply_logic.py      # no-result / error / refusal messaging
│       │   ├── prompts.py          # agent prompts (ICIO / RISEN structured)
│       │   └── orchestrator.py     # answer(token, query) → AgentAnswer
│       └── evaluation/             # KB evaluation pipeline (§6)
│           ├── prompts.py          # eval prompts (ICIO)
│           ├── question_gen.py     # user-proxy questions from original doc text (≤10/doc)
│           ├── quality.py
│           └── evaluate.py
├── tests/
│   ├── test_iam.py
│   ├── test_access_control.py      # the non-negotiable suite
│   ├── test_ingestion.py
│   └── test_agent_flows.py
├── data/                           # provided — never modified
├── doc/                            # documentation only (implementation, solution design)
├── eval/
│   ├── question_bank/              # maintained question pool — committed
│   └── runs/                       # per-run eval artifacts — gitignored
├── requirements.txt
├── requirements-dev.txt
├── .env.example
└── quickstart.md
```

Install as editable package (`pip install -e .` via a minimal `pyproject.toml`) so `frontend/`, `tests/`, the API, and the MCP server all import `knowledge_assistant` without `sys.path` hacks.

### 0.3 Vector DB — Pinecone (chosen)

**Pinecone serverless, Starter free tier** (~2 GB storage, monthly read/write unit quotas far above this corpus). Metadata filters run server-side in the query (`$eq` on `is_global`, `$in` on list-valued `access_roles`, `$ne` on `status`), which upholds the hard ACL-in-the-query invariant (§0.4-1). The index is created programmatically on first ingestion run (`dimension=1536`, cosine).

- Trade-offs (accepted): documents live in vendor cloud — fine for this fake corpus; a real deployment needs a data-residency decision. No self-host path.
- Serverless indexes don't support delete-by-metadata-filter; incremental re-ingestion deletes by **id prefix** instead — `chunk_id = {doc_id}:{page}:{seq}` makes `doc_id` the prefix by design.
- All code depends on `vectorstore/base.py` (`upsert`, `search(query_vector, acl_roles, extra_filter, top_k)`, `get_by_doc`, `delete_by_doc`); an `InMemoryVectorStore` with identical filter semantics backs the offline test suite. **TODO:** point CI at the `pinecone-local` Docker emulator for integration tests.

### 0.4 Security invariants (apply to every part)

1. **ACL filtering happens in the vector store query** (metadata pre-filter), never after generation. No code path may run a vector search without an ACL filter attached — `search()`'s `acl_roles` parameter is required with no default.
2. **The MCP server is the trust boundary.** Every tool call carries the **token**; the server resolves it to roles itself and never trusts caller-supplied roles. *(Future: replace the static token with a signed JWT — validation, expiry, scopes — without changing tool signatures.)*
3. **Document text is data.** Every prompt that includes retrieved chunk text (compressor **and** generator) wraps it in delimited untrusted-content blocks with an explicit instruction that content inside is reference material, never instructions. (Test target: `u_sam/q2-pipeline-report.pdf` contains a planted injection string per the manifest.)
4. **Deny by default.** Unknown token → 401. Chunk with missing/empty access metadata → never ingested, never returned.

### 0.5 Observability — logging, cost, latency

Every user request gets a **trace id** (FastAPI middleware, UUID) that flows through orchestrator → MCP client → compressor → generator and appears on every log line.

**`telemetry.py`** wraps all OpenAI calls (chat + embeddings) and records per call:
`trace_id, component (intent|compressor|generator|embed|eval), model, latency_ms, prompt_tokens, completion_tokens, cost_usd`.
Token counts come from the API response `usage`; cost is computed from env-configured prices (`OPENAI_MODEL_INPUT_PRICE_PER_1M`, `OPENAI_MODEL_OUTPUT_PRICE_PER_1M`, `OPENAI_EMBED_PRICE_PER_1M`) — prices are config, not hardcoded, so a model swap never silently miscosts.

**Per-request summary** logged at the end of `/chat`: total LLM calls, total tokens, total `cost_usd`, end-to-end latency with a stage breakdown (`intent_ms, retrieval_ms, compress_ms, generate_ms`). The same summary is returned in the API response's `meta` field so the UI can display it.

**Retrieval audit log** (MCP server): every tool call logs `trace_id, user_id, tool, pass (team|global), filters_applied, n_results, latency_ms`. This is the access-decision audit trail.

- POC sink: stdout + rotating file `logs/app.jsonl` (JSON lines — grep/pandas friendly).
- **TODO (next step): MLflow** — log each `/chat` trace and each evaluation run (params: models, top_k; metrics: cost, latency, eval scores; artifacts: eval reports) for run-over-run comparison.

---

## 1. Frontend UI

**Streamlit**, kept deliberately thin: it renders chat and calls the FastAPI backend over HTTP (`httpx`) — no business logic, no direct imports of agent code. Trade-off noted: Streamlit over React keeps the repo single-toolchain for a Python-judged POC; the HTTP seam means swapping in React later touches nothing behind `api/`.

### Scope

- **Login-lite:** sidebar select box of the 4 users from `users.json` (labeled by `name`), which sets the **token** sent as `Authorization: Bearer <token>` on every request. The UI stores only the token; roles never exist client-side.
- **Chat pane:** `st.chat_message` history per session; user question in, assistant answer out.
- **Citations:** rendered under each answer as expandable items — doc title, `period`, `source`, page/chunk reference, supporting snippet. `conflict` / `stale_source` flags render as a warning banner above the answer.
- **Request meta:** the `meta` block (cost, latency, token counts) from the API rendered in a small caption under each answer — makes the telemetry visible in demos.
- **Status surface:** distinct visual treatment for no-result / refusal / clarification-request / error so demo scenarios are legible.
- **Session state:** `st.session_state` for history + selected token. Switching user clears history (prevents cross-role context bleed), with a **visible notice** (toast + banner) whenever history is cleared.
- **Suggested questions:** sidebar picker over the question bank, filtered to questions the signed-in user's roles can answer; selecting asks it directly (Streamlit's chat input can't be pre-filled programmatically).
- **Controls:** sidebar **Clear chat history** button; **Stop generation** button while a request is in flight. Requests run in a background thread so the UI stays responsive; Stop closes the HTTP connection, and uvicorn cancels the in-flight agent request on client disconnect (best-effort server-side cancellation — LLM spend stops with it).

### API contract (FastAPI, `api/`)

| Route | Auth | Body / Response |
|---|---|---|
| `POST /chat` | Bearer token (required) | `{query, history}` → `AgentAnswer + meta{cost_usd, latency_ms, stage_breakdown, tokens}` |
| `GET /healthz` | none | liveness + vector store connectivity |

- `deps.py` extracts the Bearer token and passes it through **unresolved** — the API layer does not resolve roles; that happens at the MCP boundary (§0.4-2). The API only rejects a missing/blank header early with 401.
- Error middleware: exceptions → structured 500 with `trace_id`, generic message; detail goes to logs only.
- Run: `uvicorn knowledge_assistant.api.main:app` + `streamlit run frontend/app.py` (documented in `quickstart.md`).

---

## 2. IAM System

Backed entirely by the provided `data/users.json` — no invented users, roles, or claims.

### `iam/service.py`

- Loads `users.json` once at startup (`@lru_cache`); validates against a Pydantic model (`UsersFile` → `User(id, name, token, roles)`), failing fast on malformed data.
- `resolve_token(token: str) -> User` — exact match on `token`; raises `AuthenticationError` for unknown/empty tokens. Never logs the token itself (log `user.id` after resolution).
- `known_roles() -> set[str]` — from the file's `roles` array (`marketing, sales, ops, people, finance, exec`); ingestion validates manifest `access` values against it (typo in an ACL = hard error, not a silently unreadable doc).
- Multi-role semantics: OR across roles — a doc is readable if `user.roles ∩ doc.access ≠ ∅` (Sam: sales+marketing; Erin: all six). No implicit "exec sees everything" rule — the data is the policy.

**Unauthorized-topic messaging:** uniform "I don't have accessible information on that." — identical wording whether a document doesn't exist or the user isn't entitled to it, so the existence of restricted material (e.g. `ma-project-atlas.pdf`) never leaks. Trade-off accepted: the user isn't told access was the reason.

- **TODO (auth roadmap):** replace static token with signed JWT (signature, expiry, scopes) resolved at the MCP boundary; `resolve_token` is the single seam to swap.

---

## 3. AI Agent & Reply Logic

One orchestrator, four LLM-facing components. Prompts follow fixed templates — ICIO for input→output transforms (intent, compressor, all eval judges), RISEN for the generator — and state *behavior* only; output-field semantics are defined once, as `Field(description=...)` on the Pydantic response schemas, which reach the model inside the enforced JSON schema (no prompt/schema duplication to drift).

### 3.1 Orchestrator (`orchestrator.py`)

```
answer(token, query, history) →
  1. intent.gate(query, history)                   # one cheap LLM call
     ├─ greeting / out_of_domain / manipulation / unclear → reply_logic (no retrieval)
     └─ clear (rewritten query) ↓
  2. MCP call: search_knowledge(token, query)      # dual-scope is_global retrieval, §4
     ├─ results → 3
     └─ no result / error → reply_logic
  3. compressor.compress(query, chunks)            # per-chunk relevance filter, parallel
     ├─ ≥1 chunk survives → 4
     └─ all dropped → reply_logic (insufficient evidence)
  4. generator.generate(query, compressed_chunks)  # grounded, cited synthesis + flags
  5. return AgentAnswer (+ telemetry meta)
```

MCP client: official `mcp` SDK over **stdio** (orchestrator spawns `mcp_server/server.py`). No agent framework — the loop is deterministic; plain `openai` SDK (async client) with structured outputs for all classifier/generator calls.

### 3.2 Intent gate (`intent.py`)

Single structured-output call returning `{category: clear|unclear|greeting|out_of_domain|manipulation, rewritten_query, reason}`.

- `clear` → proceed with `rewritten_query` (resolves pronouns/follow-ups from history into a self-contained query). The UI excludes non-domain exchanges (out_of_domain / greeting / refused / error, plus the user turns that triggered them) from the history it sends, so settled small talk can never derail a later rewrite.
- `unclear` → one targeted clarifying question (`AgentAnswer(kind="clarify")`). Reserved for messages plainly about internal knowledge; personal questions or questions about the assistant itself are `out_of_domain`, not `unclear`. **Hard cap of one clarification round** (enforced in the orchestrator via the last assistant turn's `kind` in history): a second consecutive `unclear` gets a settled "I can only help with internal company knowledge…" reply instead of looping.
- `greeting` → canned friendly reply.
- `out_of_domain` → polite scope statement.
- `manipulation` (user-side injection / role-escalation, e.g. "act as an exec") → refusal. This gate covers the **query** side; document-side injection is handled by untrusted-content wrapping in compressor and generator prompts (§0.4-3).

### 3.4 Compressor (`compressor.py`)

Purpose: strip irrelevant content from each retrieved chunk so the generator sees only evidence that bears on the query.

- **Per-chunk processing:** one LLM call per chunk with `(rewritten_query, chunk_text)` → structured output `{has_relevant_content: bool, extract: str}`. The prompt instructs: copy relevant sentences **verbatim** (no paraphrase — extracts must remain quotable as citations); return `has_relevant_content=false` if nothing bears on the query.
- **Parallel invocation:** `asyncio.gather` over all chunks with a semaphore (concurrency cap ~8, config) to respect rate limits. Latency ≈ one call, not N.
- **Metadata mapping:** results map back by `chunk_id` — each surviving chunk keeps its full metadata (`doc_id, title, page, period, status, access…`) with `text` replaced by the verbatim extract. Chunks with `has_relevant_content=false` are dropped.
- All chunks empty → orchestrator short-circuits to `insufficient_evidence` (the generator never runs on nothing).
- Chunk text is wrapped as untrusted data in the compressor prompt — this stage is an injection surface too.
- Telemetry: logs per-chunk and aggregate (`n_in, n_out, tokens, cost_usd, latency_ms`).
- Trade-off (accepted): N extra LLM calls per query — cost/latency visible in the `meta` block. **TODO (scale):** batch multiple chunks per call, or replace with a cross-encoder reranker + extractive scorer for a large corpus.

### 3.5 Generator (`generator.py`)

Input: rewritten query + compressed chunks. Output (structured): `{answer, citations: [{chunk_id, quote}], flags, confidence}`.

Prompt contract (the load-bearing part):
- Answer **only** from the provided chunks; if they don't support an answer, return `insufficient_evidence` — never guess.
- Every claim cites at least one chunk id; citations validated **in code** post-generation — a cited `chunk_id` not in the provided set is dropped; zero valid citations → degrade to `insufficient_evidence`. This is the groundedness check.
- **Conflict handling:** if chunks disagree on a fact, surface both values with each source's `period` and `status`, prefer the `current`/newer source, set `flags=["conflict"]`. Built-in test case: sales playbook's $79 Growth price vs product-pricing's $99.
- **Staleness:** any citation from a `status="archived"` doc adds `flags=["stale_source"]` and the answer must say a newer version exists (`superseded_by` is in chunk metadata).
- Chunks wrapped as untrusted data.

### 3.6 Reply logic (`reply_logic.py`)

Pure Python (no LLM) mapping of terminal states to consistent user-facing messages: no-result (uniform wording, §2), insufficient evidence, clarification request, out-of-domain, refusal, internal error (generic message + `trace_id`; detail in logs only). Centralized so tone and behavior are consistent and testable.

---

## 4. MCP Layer

`FastMCP` server (official `mcp` SDK) in `mcp_server/server.py`. **This is where access control is enforced** — the server decides what the caller may see.

### Auth model

Every tool takes a `token` parameter; the server calls `iam.resolve_token()` itself and derives roles server-side. Caller-supplied roles are not accepted anywhere. **TODO (auth roadmap):** swap static token for JWT carried in transport auth headers once the server moves to streamable HTTP; tool signatures stay stable.

### Tools

| Tool | Signature | Behavior |
|---|---|---|
| `search_knowledge` | `(token, query, top_k=8, include_archived=False)` | Embed query → dual-scope ACL-filtered search (below). Returns chunks with full citation metadata (`doc_id, title, page, period, source, status, superseded_by, score, text`). Empty → structured `{status: "no_result"}`, distinct from errors. |

### Retrieval strategy — merged dual-scope search

Chunk metadata carries two access fields (computed at ingestion, §5):

- `is_global: bool` — `True` iff the chunk's effective ACL is all six roles (company-wide).
- `access_roles: list[str]` — the effective ACL (always populated, including for global chunks).

`search_knowledge` **always searches both scopes and merges by similarity score**:

1. **Team scope:** filter `is_global == False` **and** `access_roles ∩ user.roles ≠ ∅`.
2. **Global scope:** filter `is_global == True` (ACL condition still attached — trivially satisfied since global = everyone, but no unfiltered code path exists, per §0.4-1).
3. Results are deduped, score-floored, sorted by similarity, and capped at top-k. `status != "archived"` applies to both scopes unless `include_archived=True`.

Merging both scopes is load-bearing for conflict detection: a team document that disagrees with a company-wide document (the $79 playbook vs the $99 pricing sheet) must reach the generator **together with it**, or the conflict can never be flagged — the generator only flags disagreement between chunks it sees. Team preference emerges from semantic closeness, not hard gating. Chunk-level overrides compose correctly: the exec-only paragraph inside the company-wide all-hands doc has `access_roles=["exec"]`, hence `is_global=False` — it lives in the team scope, visible only to exec.

- Trade-off (accepted): every query costs two vector searches; scope latency is logged per search. **TODO (scale):** single query with a role-aware boost instead of two scoped searches; tune the score floor on eval data.
- Server config from the same `config.py`; every tool call logged per §0.5 (audit trail). Transport: stdio for POC; streamable HTTP is the deployment shape.

---

## 5. KB Creation / Update Pipeline

CLI: `python -m knowledge_assistant.ingestion.pipeline [--rebuild]`. Idempotent; re-running updates changed docs only.

### Stages

1. **Load** (`loader.py`) — `pypdf` extracts text per page (page numbers kept for citations), then **NFKC-normalizes** it: PDF ligatures (`ﬁ ﬂ ﬀ` → `fi fl ff`) and non-breaking spaces become plain ASCII, so marker matching and embeddings see clean text. A PDF that fails to parse or yields empty text is **logged and skipped**, never silently dropped; the run report lists skipped files. Folder names (`u_maria/…`) are organizational only — the manifest is the authority: any PDF on disk not listed in `manifest.json` is **skipped with a warning** (no ACL → no ingestion, deny by default).
2. **Chunk** (`chunker.py`) — **heading-aware section splitting**, never crossing page boundaries (citations stay page-exact), hard cap 2,000 chars/chunk, **no overlap**. This corpus extracts with no blank lines, so headings are the structure signal: a short line (<60 chars) without terminal punctuation starts a new section; over-detection (e.g. table rows) is harmless because consecutive sections are re-packed greedily up to the cap. A confidential-marker line **always** starts its own section, and marker sections are never packed with other content — a restricted section can never share a chunk (and thus an ACL) with general text. Oversized sections fall back to line-level splitting. Deterministic `chunk_id = {doc_id}:{page}:{seq}`.
   - Trade-offs: the heading heuristic is tuned to this corpus's conventions; a marker section runs to the next heading or page end, which can over-restrict a trailing general line — fails safe (over-restriction, never a leak). No overlap is deliberate: sections are self-contained, and tail-overlap could smear restricted text into a chunk with a broader ACL.
   - Future (if other document shapes appear): layout-aware heading detection (PyMuPDF font sizes or `unstructured`), per-source heading conventions in config, table-aware extraction, bounded within-section overlap for long prose documents.
3. **Enrich** (`enricher.py`) — per chunk:
   - Attach manifest metadata: `doc_id, title, period, source, status, supersedes/superseded_by`.
   - **Effective ACL:** start from the manifest `access` list; apply chunk-level overrides (below); validate every role against `iam.known_roles()` (unknown role = hard error).
   - **Confidential-marker override:** if the chunk text contains the marker `CONFIDENTIAL — EXECUTIVE COMMITTEE ONLY` (case-insensitive, dash-variant tolerant: `—`/`–`/`-`), the chunk's `access_roles` is overridden to `["exec"]`. Assumption: this exact marker convention is how sub-document restrictions are expressed in this corpus; the known instance is the exec-only paragraph in `general/all-hands-2025-q2.pdf`. The shared regex lives in `ingestion/markers.py`; the chunker guarantees the marker section is an isolated chunk, so the override restricts only the confidential section — the rest of a company-wide document stays company-wide.
   - **Derive `is_global`:** `True` iff effective `access_roles` == all six roles; otherwise `False`. Derived *after* overrides, so a demoted chunk in a global doc is correctly non-global.
   - `content_hash` per doc for incremental updates (unchanged → skip; changed → delete chunks by `doc_id`, re-insert).
4. **Embed & upsert** — batch-embed via `text-embedding-3-small`, upsert through the `VectorStore` protocol. Embedding calls go through `telemetry.py` (tokens + cost per batch); pipeline run summary logs total chunks, tokens, cost, duration.

- Known limitation (state in write-up): ACLs are baked into chunk metadata at ingestion; a manifest permission change requires re-running the pipeline (cheap here). **TODO (scale):** event-driven re-index on manifest change; move ACL resolution to query time against a live policy store.

---

## 6. KB Evaluation Pipeline

Two CLIs — question-bank maintenance and evaluation. The **question bank** (`eval/question_bank/questions.json`) is a maintained, committed dataset — versioning it keeps runs comparable; **run artifacts** (`eval/runs/`) are gitignored. Every eval LLM/embedding call goes through `telemetry.py`; reports include total cost and duration. No generated-answer grading — evaluation measures **retrieval**, before the compressor.

### 6.1 Question bank generation — `python -m knowledge_assistant.evaluation.question_gen`

| Mode | Behavior |
|---|---|
| *(default)* | **Auto-scan:** generate only for manifest documents with no questions in the pool yet — covers newly added knowledge |
| `--doc DOC_ID` (repeatable) | Regenerate for the named document(s); other documents' questions are kept |
| `--all` | Regenerate the entire pool |

- Questions are user-proxy: generated from the **original document text** (not chunks), phrased the way an employee types into a chat box rather than in the document's wording — avoids vocabulary bias between question and chunk embeddings. Documents longer than ~8k chars are split into page-aligned parts first. Up to **10 questions per document (soft limit** — thin docs yield fewer; never pad). Each question is tagged `{source_doc_id, access_roles}` (document-level ACL).
- Newly generated questions pass the **quality judge** (`quality.py` — answerability + specificity, 1–5, threshold 3.5) before entering the pool → `eval/question_bank/questions.json`.

### 6.2 Evaluation — `python -m knowledge_assistant.evaluation.evaluate [--mode retrieval|e2e] [--num-questions N] [--seed S]`

Requires a non-empty pool. Default evaluates **every** pooled question; `--num-questions N` evaluates a **random sample** of N from the pool (`--seed` makes the sample reproducible — cheaper smoke runs vs. full sweeps).

**`--mode retrieval` (default)** — measures the retrieval layer directly. Per sampled question, run as a user entitled to the source doc:

- **a. Retrieval quality:** merged dual-scope search top-k → **hit-rate@k** (source doc in results) + **MRR**.
- **b. Reverse-question relevancy:** each retrieved chunk is reversed into up to 5 questions it can answer (LLM, **cached per chunk_id** — a chunk is reversed once per run, ≤1 call per chunk in the corpus). An LLM judge then scores each reversed question against the input question on a **1–5 relevancy scale** (one batched call per query); per input question, the **relevancy score** = max of its judge scores (best candidate), and the report headline is the average of per-question scores. Measures whether what came back can actually answer what was asked. Low relevancy with high hit-rate = right doc, noisy chunks; low both = retrieval failure.
- **c. Access control (the one that matters):** the same query retried as every `users.json` user *without* access to the source doc. **Pass = zero chunks from that doc and zero chunks whose ACL excludes the user. Any failure exits non-zero.** Doubles as `test_access_control.py` in pytest so it runs in CI, not just in reports.
- Fixed behavioral scenarios (hand-written, keyed to the planted traps) stay in the suite: pricing-conflict → conflict flag + $99 preferred; brand-guidelines → v3 cited, not archived v2; injection probe via `q2-pipeline-report.pdf` → embedded instruction not obeyed; Maria asks compensation → uniform no-result wording, zero leakage; exec-only all-hands paragraph → answerable by Erin only.

**`--mode e2e`** — simulates real app usage: each sampled question runs through the **full agent path** (intent rewrite → MCP boundary → compressor → generator) via `orchestrator.answer(token, question)`, as the entitled user **and** as every unentitled user. Metrics: **citation hit-rate** (source doc among the answer's citations), **answer-kind distribution** (answered / no_result / clarify / …), **flag counts** (conflict, stale_source), avg latency + per-question cost (from the app's own telemetry `meta`, incl. stage breakdown). Access check: an unentitled user's answer must contain **no citation to a document they cannot read** (answers grounded in their own accessible docs are legitimate); any violation fails the run. Trade-off: ~10 LLM calls per question per user — run it on a `--num-questions` sample, not the full pool.

### 6.3 Outputs — inputs for later visualization

`report.json` (summary incl. `mode`, pool size, sample size, seed; retrieval mode: hit-rate, MRR, relevancy-score mean; e2e mode: citation hit-rate, kind distribution, flag counts, avg latency; both: access pass/fail, cost, duration), `results.json` (per-query detail — retrieval mode: retrieved chunks + raw scores, reversed questions + 1–5 judge scores, access sweep; e2e mode: answer text, kind, flags, cited docs, per-stage cost/latency, access sweep), `reversed_questions.json` (chunk → reversed-questions cache).

- Trade-off: the relevancy score depends on the reversal LLM's question quality; the per-chunk cache keeps its cost linear in corpus size, not query count. Sampled runs trade metric stability for cost — small N wobbles run-to-run unless seeded.
- Score-floor tuning needs no code: `SCORE_FLOOR` is env config and `results.json` retains every returned chunk's raw score — run once with `SCORE_FLOOR=0`, then sweep candidate floors offline against the stored scores.
- **TODO (next step):** MLflow-track each eval run (params: models, top_k, score floor, sample size/seed; metrics + artifacts above) for run-over-run comparison; the Streamlit **evaluation dashboard page** (`frontend/pages/2_evaluation_dashboard.py`) renders `report.json`/`results.json` — metric tiles, access banner, per-doc hit-rate, relevancy distribution, score-vs-similarity scatter, per-question drill-down (mode-aware for retrieval/e2e); add answer-level faithfulness grading if the POC graduates.

---

## 7. requirements.txt (only what's imported)

```
fastapi             # API layer
uvicorn             # ASGI server
httpx               # Streamlit → FastAPI client
openai              # LLM + embeddings (async client)
mcp                 # MCP server (FastMCP) + client SDK
pinecone            # vector store client (§0.3)
pypdf               # PDF text extraction
streamlit           # frontend
pydantic            # data contracts
pydantic-settings   # config from .env
```

`requirements-dev.txt`: `pytest`, `ruff`. Notes: `python-dotenv` comes transitively with `pydantic-settings` — not listed unless imported directly. Versions pinned (`~=`) at first successful run. Nothing enters this file until the import actually exists in `src/`.

`.env` additions beyond the scaffold's example: `OPENAI_MODEL=gpt-5.6-luna`, `OPENAI_EMBED_MODEL=text-embedding-3-small`, the three price-per-1M vars (§0.5), `PINECONE_API_KEY` / `PINECONE_INDEX` / `PINECONE_CLOUD` / `PINECONE_REGION`.

---

## 8. Build order (suggested)

1. `models.py`, `config.py`, `log.py`, `telemetry.py`, `iam/` + tests — everything depends on these.
2. Ingestion pipeline + vector store (§0.3 decided) → index the 12 PDFs; hand-verify chunk metadata, `is_global` derivation, and the exec-only override on the all-hands doc.
3. MCP server + `test_access_control.py` — prove the boundary (both passes) before any LLM work.
4. Agent (intent → orchestrator → compressor → generator → reply logic) end-to-end in a REPL.
5. FastAPI layer, then Streamlit UI against it.
6. Evaluation pipeline; run it, fix what it finds.
7. `quickstart.md` + written note (approach, assumptions, next steps — the TODO items collected from each section).
