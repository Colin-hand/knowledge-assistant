"""Knowledge MCP server — THE access-control boundary.

Every tool takes the raw token and resolves it here via iam.service;
caller-supplied roles are accepted nowhere. Unauthorized and nonexistent
documents return the identical not-found shape (no existence leak).

Transport: stdio (spawned by the agent orchestrator). TODO (auth roadmap):
streamable HTTP with JWT in transport auth headers; tool signatures stable.
"""

import time

from mcp.server.fastmcp import FastMCP

from knowledge_assistant import retrieval, telemetry
from knowledge_assistant.iam.service import AuthenticationError, resolve_token
from knowledge_assistant.log import get_logger
from knowledge_assistant.models import SearchResponse
from knowledge_assistant.vectorstore.pinecone_store import get_store

logger = get_logger(__name__)
mcp = FastMCP("knowledge-mcp")

_AUTH_ERROR = {"status": "error", "error": "invalid_token"}
_NOT_FOUND = {"status": "not_found"}


def _audit(tool: str, user_id: str, t0: float, **fields) -> None:
    logger.info(
        "mcp_tool_call",
        extra={"tool": tool, "user_id": user_id, "latency_ms": round((time.perf_counter() - t0) * 1000, 1), **fields},
    )


@mcp.tool()
async def search_knowledge(
    token: str, query: str, top_k: int = 8, include_archived: bool = False
) -> dict:
    """Permission-scoped semantic search over the knowledge base.

    Returns chunks (with citation metadata) the calling user is entitled to
    see. Team-specific and company-wide scopes are always both searched and
    merged by similarity.
    """
    t0 = time.perf_counter()
    try:
        user = resolve_token(token)
    except AuthenticationError:
        return _AUTH_ERROR
    vec = (await telemetry.embed("query_embed", [query]))[0]
    chunks, scope = retrieval.search_accessible(
        get_store(), user.roles, vec, top_k=top_k, include_archived=include_archived
    )
    _audit("search_knowledge", user.id, t0, scope=scope, n_results=len(chunks),
           include_archived=include_archived)
    if not chunks:
        return SearchResponse(status="no_result", scope="none").dump()
    return SearchResponse(status="ok", scope=scope, chunks=chunks).dump()


@mcp.tool()
async def get_document(token: str, doc_id: str) -> dict:
    """Full metadata and chunks for one document, iff the caller may read it."""
    t0 = time.perf_counter()
    try:
        user = resolve_token(token)
    except AuthenticationError:
        return _AUTH_ERROR
    from knowledge_assistant.ingestion.pipeline import load_manifest

    doc = next((d for d in load_manifest().documents if d.doc_id == doc_id), None)
    if doc is None or not set(doc.access) & set(user.roles):
        _audit("get_document", user.id, t0, doc_id=doc_id, granted=False)
        return _NOT_FOUND
    # Chunk-level ACL still applies (e.g. exec-only paragraphs in global docs).
    chunks = [
        c for c in get_store().get_by_doc(doc_id) if set(c.access_roles) & set(user.roles)
    ]
    _audit("get_document", user.id, t0, doc_id=doc_id, granted=True, n_chunks=len(chunks))
    return {
        "status": "ok",
        "document": doc.model_dump(mode="json"),
        "chunks": [c.model_dump(mode="json") for c in chunks],
    }


@mcp.tool()
async def list_sources(token: str) -> dict:
    """Documents the calling user may read (title, period, source, status)."""
    t0 = time.perf_counter()
    try:
        user = resolve_token(token)
    except AuthenticationError:
        return _AUTH_ERROR
    from knowledge_assistant.ingestion.pipeline import load_manifest

    docs = [
        {"doc_id": d.doc_id, "title": d.title, "period": d.period,
         "source": d.source, "status": d.status}
        for d in load_manifest().documents
        if set(d.access) & set(user.roles)
    ]
    _audit("list_sources", user.id, t0, n_docs=len(docs))
    return {"status": "ok", "sources": docs}


if __name__ == "__main__":
    mcp.run()
