"""VectorStore protocol; acl_roles is required — no unfiltered search."""

from typing import Protocol

from knowledge_assistant.models import Chunk


class VectorStore(Protocol):
    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None: ...

    def search(
        self,
        query_vector: list[float],
        acl_roles: list[str],
        extra_filter: dict | None,
        top_k: int,
    ) -> list[Chunk]: ...

    def get_by_doc(self, doc_id: str) -> list[Chunk]: ...

    def delete_by_doc(self, doc_id: str) -> None: ...


def chunk_metadata(chunk: Chunk) -> dict:
    """Flat metadata stored alongside each vector."""
    meta = {
        "doc_id": chunk.doc_id,
        "title": chunk.title,
        "page": chunk.page,
        "seq": chunk.seq,
        "text": chunk.text,
        "access_roles": chunk.access_roles,
        "is_global": chunk.is_global,
        "period": chunk.period,
        "source": chunk.source,
        "status": chunk.status,
    }
    if chunk.superseded_by:
        meta["superseded_by"] = chunk.superseded_by
    return meta


def chunk_from_metadata(chunk_id: str, meta: dict, score: float | None = None) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=meta["doc_id"],
        title=meta["title"],
        page=int(meta["page"]),
        seq=int(meta["seq"]),
        text=meta["text"],
        access_roles=list(meta["access_roles"]),
        is_global=bool(meta["is_global"]),
        period=meta["period"],
        source=meta["source"],
        status=meta["status"],
        superseded_by=meta.get("superseded_by"),
        score=score,
    )
