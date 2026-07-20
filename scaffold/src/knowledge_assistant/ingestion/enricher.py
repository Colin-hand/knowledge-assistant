"""Attach manifest metadata; marker chunks demoted to exec-only."""

from knowledge_assistant.iam.service import known_roles
from knowledge_assistant.ingestion.chunker import RawChunk
from knowledge_assistant.ingestion.markers import EXEC_MARKER
from knowledge_assistant.log import get_logger
from knowledge_assistant.models import Chunk, DocumentMeta

logger = get_logger(__name__)


def enrich(doc: DocumentMeta, raw_chunks: list[RawChunk]) -> list[Chunk]:
    valid = known_roles()
    bad = set(doc.access) - valid
    if bad:
        raise ValueError(f"{doc.path}: unknown role(s) in manifest access list: {sorted(bad)}")

    all_roles = valid  # is_global iff every known role may read
    chunks: list[Chunk] = []
    for rc in raw_chunks:
        access = list(doc.access)
        if EXEC_MARKER.search(rc.text):
            access = ["exec"]
            logger.info(
                "chunk_acl_override",
                extra={"doc_id": doc.doc_id, "page": rc.page, "seq": rc.seq, "roles": access},
            )
        chunks.append(
            Chunk(
                chunk_id=f"{doc.doc_id}:{rc.page}:{rc.seq}",
                doc_id=doc.doc_id,
                title=doc.title,
                page=rc.page,
                seq=rc.seq,
                text=rc.text,
                access_roles=access,
                is_global=set(access) == all_roles,
                period=doc.period,
                source=doc.source,
                status=doc.status,
                superseded_by=doc.superseded_by,
            )
        )
    return chunks
