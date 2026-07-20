"""KB ingestion CLI: python -m knowledge_assistant.ingestion.pipeline [--rebuild]

Idempotent by content hash; unlisted PDFs are never ingested.
"""

import argparse
import asyncio
import hashlib
import json
import time

from knowledge_assistant import telemetry
from knowledge_assistant.config import get_settings
from knowledge_assistant.ingestion.chunker import chunk_pages
from knowledge_assistant.ingestion.enricher import enrich
from knowledge_assistant.ingestion.loader import PdfLoadError, load_pdf
from knowledge_assistant.log import get_logger, new_trace_id
from knowledge_assistant.models import Manifest
from knowledge_assistant.vectorstore.base import VectorStore

logger = get_logger(__name__)


def load_manifest() -> Manifest:
    raw = json.loads(get_settings().manifest_file.read_text())
    return Manifest.model_validate(raw)


def _load_state() -> dict[str, str]:
    f = get_settings().ingest_state_file
    return json.loads(f.read_text()) if f.exists() else {}


def _save_state(state: dict[str, str]) -> None:
    get_settings().ingest_state_file.write_text(json.dumps(state, indent=2))


async def run(store: VectorStore, rebuild: bool = False) -> dict:
    s = get_settings()
    new_trace_id()
    telemetry.start_request()
    t0 = time.perf_counter()
    manifest = load_manifest()
    state = {} if rebuild else _load_state()
    report = {"ingested": [], "skipped_unchanged": [], "failed": [], "unlisted": []}

    listed = {d.path for d in manifest.documents}
    for pdf in sorted(s.pdf_dir.rglob("*.pdf")):
        rel = str(pdf.relative_to(s.pdf_dir))
        if rel not in listed:
            logger.warning("pdf_not_in_manifest_skipped", extra={"path": rel})
            report["unlisted"].append(rel)

    total_chunks = 0
    for doc in manifest.documents:
        path = s.pdf_dir / doc.path
        if not path.exists():
            logger.error("manifest_doc_missing_on_disk", extra={"doc_id": doc.doc_id})
            report["failed"].append(doc.path)
            continue
        content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
        if state.get(doc.doc_id) == content_hash:
            report["skipped_unchanged"].append(doc.path)
            continue
        try:
            pages = load_pdf(path)
        except PdfLoadError as exc:
            logger.error("pdf_load_failed", extra={"doc_id": doc.doc_id, "error": str(exc)})
            report["failed"].append(doc.path)
            continue

        chunks = enrich(doc, chunk_pages(pages))
        vectors = await telemetry.embed("ingest_embed", [c.text for c in chunks])
        store.delete_by_doc(doc.doc_id)
        store.upsert(chunks, vectors)
        state[doc.doc_id] = content_hash
        total_chunks += len(chunks)
        report["ingested"].append(doc.path)
        logger.info("doc_ingested", extra={"doc_id": doc.doc_id, "n_chunks": len(chunks)})

    _save_state(state)
    meta = telemetry.summary((time.perf_counter() - t0) * 1000)
    report["total_chunks"] = total_chunks
    report["cost_usd"] = meta.cost_usd
    report["duration_ms"] = meta.latency_ms
    logger.info("ingestion_complete", extra=report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest manifest-listed PDFs into Pinecone")
    parser.add_argument("--rebuild", action="store_true", help="re-ingest everything")
    args = parser.parse_args()

    from knowledge_assistant.vectorstore.pinecone_store import get_store

    report = asyncio.run(run(get_store(), rebuild=args.rebuild))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
