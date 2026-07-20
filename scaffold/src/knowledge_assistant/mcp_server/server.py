"""Access-control boundary: resolves the raw token to roles itself.

Stdio subprocess; KA_TRACE_ID carries the caller's trace id.
"""

import os
import time

from mcp.server.fastmcp import FastMCP

from knowledge_assistant import retrieval, telemetry
from knowledge_assistant.iam.service import AuthenticationError, resolve_token
from knowledge_assistant.log import get_logger, set_trace_id
from knowledge_assistant.models import SearchResponse
from knowledge_assistant.vectorstore.pinecone_store import get_store

logger = get_logger(__name__)
mcp = FastMCP("knowledge-mcp")

_AUTH_ERROR = {"status": "error", "error": "invalid_token"}


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


if __name__ == "__main__":
    if tid := os.environ.get("KA_TRACE_ID"):
        set_trace_id(tid)
    mcp.run()
