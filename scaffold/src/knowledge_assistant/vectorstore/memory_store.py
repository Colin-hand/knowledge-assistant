"""In-memory VectorStore mirroring Pinecone filter semantics (tests)."""

import math

from knowledge_assistant.models import Chunk
from knowledge_assistant.vectorstore.base import chunk_from_metadata, chunk_metadata


def _matches(meta: dict, flt: dict) -> bool:
    for key, cond in flt.items():
        if key == "$and":
            if not all(_matches(meta, sub) for sub in cond):
                return False
            continue
        value = meta.get(key)
        if not isinstance(cond, dict):
            cond = {"$eq": cond}
        for op, operand in cond.items():
            if op == "$eq":
                if value != operand:
                    return False
            elif op == "$ne":
                if value == operand:
                    return False
            elif op == "$in":
                if isinstance(value, list):
                    if not set(value) & set(operand):
                        return False
                elif value not in operand:
                    return False
            else:
                raise ValueError(f"unsupported filter op: {op}")
    return True


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class InMemoryVectorStore:
    def __init__(self) -> None:
        self._rows: dict[str, tuple[list[float], dict]] = {}

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        for chunk, vec in zip(chunks, vectors, strict=True):
            self._rows[chunk.chunk_id] = (vec, chunk_metadata(chunk))

    def search(
        self,
        query_vector: list[float],
        acl_roles: list[str],
        extra_filter: dict | None,
        top_k: int,
    ) -> list[Chunk]:
        flt: dict = {"access_roles": {"$in": list(acl_roles)}}
        if extra_filter:
            flt = {"$and": [flt, extra_filter]}
        scored = [
            (chunk_id, _cosine(query_vector, vec), meta)
            for chunk_id, (vec, meta) in self._rows.items()
            if _matches(meta, flt)
        ]
        scored.sort(key=lambda t: t[1], reverse=True)
        return [chunk_from_metadata(cid, meta, score) for cid, score, meta in scored[:top_k]]

    def get_by_doc(self, doc_id: str) -> list[Chunk]:
        rows = [
            chunk_from_metadata(cid, meta)
            for cid, (_, meta) in self._rows.items()
            if meta["doc_id"] == doc_id
        ]
        return sorted(rows, key=lambda c: (c.page, c.seq))

    def delete_by_doc(self, doc_id: str) -> None:
        for cid in [c for c, (_, m) in self._rows.items() if m["doc_id"] == doc_id]:
            del self._rows[cid]
