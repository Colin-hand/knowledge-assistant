"""Pinecone serverless VectorStore impl.

Serverless indexes don't support delete-by-metadata-filter, so incremental
re-ingestion deletes by id prefix — chunk ids are `{doc_id}:{page}:{seq}`,
making `doc_id` the prefix by design.
"""

from functools import lru_cache

from pinecone import Pinecone, ServerlessSpec

from knowledge_assistant.config import get_settings
from knowledge_assistant.log import get_logger
from knowledge_assistant.models import Chunk
from knowledge_assistant.vectorstore.base import chunk_from_metadata, chunk_metadata

logger = get_logger(__name__)


class PineconeStore:
    def __init__(self) -> None:
        s = get_settings()
        if not s.pinecone_api_key:
            raise RuntimeError("PINECONE_API_KEY is not set — fill in .env first")
        self._pc = Pinecone(api_key=s.pinecone_api_key)
        if s.pinecone_index not in self._pc.list_indexes().names():
            logger.info("creating_pinecone_index", extra={"index": s.pinecone_index})
            self._pc.create_index(
                name=s.pinecone_index,
                dimension=s.embed_dimension,
                metric="cosine",
                spec=ServerlessSpec(cloud=s.pinecone_cloud, region=s.pinecone_region),
            )
        self._index = self._pc.Index(s.pinecone_index)

    def upsert(self, chunks: list[Chunk], vectors: list[list[float]]) -> None:
        rows = [
            {"id": c.chunk_id, "values": v, "metadata": chunk_metadata(c)}
            for c, v in zip(chunks, vectors, strict=True)
        ]
        for i in range(0, len(rows), 100):
            self._index.upsert(vectors=rows[i : i + 100])

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
        res = self._index.query(
            vector=query_vector, top_k=top_k, filter=flt, include_metadata=True
        )
        return [
            chunk_from_metadata(m["id"], dict(m["metadata"]), m.get("score"))
            for m in res.get("matches", [])
        ]

    def _ids_for_doc(self, doc_id: str) -> list[str]:
        # list() yields pages; page items are ListItem objects (v9) or bare id strings.
        ids: list[str] = []
        for page in self._index.list(prefix=f"{doc_id}:"):
            ids.extend(item.id if hasattr(item, "id") else str(item) for item in page)
        return ids

    def get_by_doc(self, doc_id: str) -> list[Chunk]:
        ids = self._ids_for_doc(doc_id)
        if not ids:
            return []
        fetched = self._index.fetch(ids=ids)
        chunks = [
            chunk_from_metadata(cid, dict(vec.metadata))
            for cid, vec in fetched.vectors.items()
        ]
        return sorted(chunks, key=lambda c: (c.page, c.seq))

    def delete_by_doc(self, doc_id: str) -> None:
        ids = self._ids_for_doc(doc_id)
        if ids:
            self._index.delete(ids=ids)


@lru_cache
def get_store() -> PineconeStore:
    return PineconeStore()
