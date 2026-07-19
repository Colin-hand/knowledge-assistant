import json

import pytest

from knowledge_assistant.config import get_settings
from knowledge_assistant.models import Chunk, Manifest, UsersFile
from knowledge_assistant.vectorstore.memory_store import InMemoryVectorStore

DIM = 8


@pytest.fixture(scope="session")
def users_file() -> UsersFile:
    return UsersFile.model_validate(json.loads(get_settings().users_file.read_text()))


@pytest.fixture(scope="session")
def manifest() -> Manifest:
    return Manifest.model_validate(json.loads(get_settings().manifest_file.read_text()))


def make_chunk(doc, page: int = 1, seq: int = 0, access: list[str] | None = None,
               all_roles: frozenset[str] | None = None, text: str = "lorem ipsum") -> Chunk:
    from knowledge_assistant.iam.service import known_roles

    roles = access if access is not None else list(doc.access)
    return Chunk(
        chunk_id=f"{doc.path}:{page}:{seq}",
        doc_id=doc.path,
        title=doc.title,
        page=page,
        seq=seq,
        text=text,
        access_roles=roles,
        is_global=set(roles) == set(all_roles or known_roles()),
        period=doc.period,
        source=doc.source,
        status=doc.status,
        superseded_by=doc.superseded_by,
    )


@pytest.fixture()
def populated_store(manifest) -> InMemoryVectorStore:
    """One chunk per manifest doc (identical vectors → filtering is all that
    differs), plus the exec-only-override chunk inside the global all-hands doc."""
    store = InMemoryVectorStore()
    vec = [1.0] * DIM
    chunks, vectors = [], []
    for doc in manifest.documents:
        chunks.append(make_chunk(doc))
        vectors.append(vec)
        if doc.path == "general/all-hands-2025-q2.pdf":
            chunks.append(make_chunk(doc, page=2, seq=0, access=["exec"],
                                     text="CONFIDENTIAL — EXECUTIVE COMMITTEE ONLY: secret"))
            vectors.append(vec)
    store.upsert(chunks, vectors)
    return store
