"""ACL-scoped retrieval over team-specific and company-wide knowledge.

Both scopes are ALWAYS searched and merged by similarity score, so
company-wide documents that disagree with team documents (e.g. an outdated
team playbook price vs the current global pricing sheet) reach the generator
together and conflicts can be flagged — the generator can only flag
disagreement between chunks it sees.

The ACL condition is attached by the store on both queries; no unfiltered
code path exists (security invariant §0.4-1).
"""

import time
from typing import Literal

from knowledge_assistant.config import get_settings
from knowledge_assistant.log import get_logger
from knowledge_assistant.models import Chunk
from knowledge_assistant.vectorstore.base import VectorStore

logger = get_logger(__name__)

Scope = Literal["team", "global", "both", "none"]


def search_accessible(
    store: VectorStore,
    roles: list[str],
    query_vector: list[float],
    top_k: int | None = None,
    include_archived: bool = False,
    score_floor: float | None = None,
) -> tuple[list[Chunk], Scope]:
    s = get_settings()
    top_k = top_k or s.top_k
    floor = s.score_floor if score_floor is None else score_floor

    def scope_filter(is_global: bool) -> dict:
        flt: dict = {"is_global": {"$eq": is_global}}
        if not include_archived:
            flt["status"] = {"$ne": "archived"}
        return flt

    merged: dict[str, Chunk] = {}
    for scope_name, is_global in (("team", False), ("global", True)):
        t0 = time.perf_counter()
        results = store.search(
            query_vector=query_vector,
            acl_roles=roles,
            extra_filter=scope_filter(is_global),
            top_k=top_k,
        )
        logger.info(
            "vector_search",
            extra={
                "scope": scope_name,
                "roles": roles,
                "n_raw": len(results),
                "latency_ms": round((time.perf_counter() - t0) * 1000, 1),
            },
        )
        for chunk in results:
            merged.setdefault(chunk.chunk_id, chunk)

    kept = sorted(
        (c for c in merged.values() if c.score is None or c.score >= floor),
        key=lambda c: c.score or 0.0,
        reverse=True,
    )[:top_k]
    if not kept:
        return [], "none"
    has_team = any(not c.is_global for c in kept)
    has_global = any(c.is_global for c in kept)
    scope: Scope = "both" if has_team and has_global else ("team" if has_team else "global")
    return kept, scope
