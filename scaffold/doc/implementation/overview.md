# Overview — Internal Knowledge Assistant (POC)

## Approach

Access-aware RAG exposed over MCP: Streamlit UI → FastAPI → agent (intent → retrieve → compress → generate) → Knowledge MCP server → Pinecone.

- **Ingestion**: manifest-listed PDFs are loaded page by page, chunked along headings (never across pages), enriched with manifest metadata + ACL, embedded, and upserted. Content hashes make re-ingestion incremental and idempotent.
- **Retrieval**: one shared function searches team-specific and company-wide scopes, merges by similarity, and applies a score floor. The ACL filter (`access_roles $in` the user's roles) is attached inside the vector query itself — unauthorized chunks are never retrieved, so they can never reach the prompt.
- **MCP boundary**: the MCP server (`search_knowledge`, `get_document`, `list_sources`) is where access is enforced. Every tool resolves the raw token to roles itself; caller-supplied roles are accepted nowhere.
- **Traceability**: chunk IDs are `{doc_id}:{page}:{seq}`; every claim in an answer carries a citation (title, page, period, source, status, verbatim quote). Citations are validated in code — a citation pointing outside the retrieved set is dropped.
- **Injection safety**: retrieved text is always wrapped in `<untrusted_document_content>` tags and treated as data, never instructions.

### Trade-offs

- **PDF-only ingestion**: the loader focuses on PDFs; other formats (Excel, images, docx…) need their own loaders/OCR. The pipeline seams (loader → chunker → enricher) are format-agnostic, so adding one is localized work.
- **Both scopes always searched** (team + global): slightly more retrieval work, but conflicts between a team doc and a global doc (e.g. the outdated $79 vs current $99 price) stay visible so the generator can flag them.
- **MCP as stdio subprocess per request**: simple and correct at POC scale; a persistent session or streamable-HTTP server is the scaling step.
- **In-query ACL filter over post-filtering**: post-filtering top-k results could return fewer than k after removal and risks leaking via reranking; filtering inside the query keeps recall and safety.

### Fallback flows

Every "cannot answer" path returns a deliberate, distinct reply instead of a guess:

- **Unclear question** → one clarification round max; a second consecutive unclear settles instead of looping.
- **Greeting / out-of-domain** → canned reply, no retrieval spend.
- **Manipulation attempt** → refused at the intent gate and logged.
- **Invalid token** → auth error; the presented token value is never logged.
- **No accessible chunks** → "no accessible information" — wording identical for unauthorized and nonexistent content, so nothing leaks by existence.
- **Zero valid citations or unsupported answer** → degraded to insufficient-evidence in code, independent of the model.
- **Conflicting sources** → answer states both values with period/status plus a `conflict` flag; **archived source cited** → `stale_source` flag appended by code.
- **Unexpected error** → generic reply carrying only a trace ID for log correlation.

## Assumptions

- **Identity**: a request presents a static bearer token; `users.json` resolves it to a user with a set of roles. Tokens are demo credentials — real validation (JWT signature, expiry, scopes) is out of scope by design, and `iam/service.py` is the single seam to swap it in.
- **Permissions**: the manifest is the ACL authority, not folder names. Each document's `access` list names the roles that may read it; a PDF on disk that is not in the manifest is not ingested (no ACL → deny by default). Access is role-based and flat — no role hierarchy or inheritance.
- **Sub-document restrictions** are expressed by an in-text marker convention ("Confidential — Executive Committee Only"): a marked section becomes its own chunk and its ACL is demoted to `exec`, restricting only that section.
- **Corpus conventions**: `status: archived` marks stale versions (excluded from retrieval by default); `period` + `status` are sufficient to prefer current sources when facts conflict.

## Skipped for now

- Real token validation (JWT) and transport-level auth for a remote MCP server.
- Non-PDF ingestion (Excel, images/OCR, docx).
- Persistent MCP session / streamable-HTTP transport.
- MLflow experiment tracking (structured JSONL telemetry with per-call cost/latency is in place instead).
- Reranking and hybrid (keyword + vector) search.

## Potential next steps

- **JWT tokens**: swap `resolve_token` for signed-JWT validation (signature, expiry, scopes) — module already isolated for this.
- **Fine-tune KB quality from evaluation results**: use the eval harness (hit-rate@k, MRR, reversed-relevancy, access-leak checks) to iterate on chunk size, score floor, and top-k.
- Add loaders for more formats and a reranker once the corpus grows.
- Move the MCP server to streamable HTTP with auth headers; tool signatures stay stable.
