"""Generate user-proxy eval questions from ORIGINAL document text (not chunks).

CLI: python -m knowledge_assistant.evaluation.question_gen [--all | --doc ID ...]

Modes:
  (default)   auto-scan: generate only for manifest documents that have no
              questions in the pool yet (covers newly added knowledge)
  --doc ID    regenerate for the named document(s); other docs' questions kept
  --all       regenerate the entire pool

Newly generated questions pass the quality judge (quality.py) before entering
the pool at eval/question_bank/questions.json. Documents longer than MAX_DOC_CHARS
are split into page-aligned parts first (char-split as a last resort for a
single oversized page); ≤ MAX_QUESTIONS_PER_DOC per document, doc-level ACL
tagged so the access-control suite knows who is entitled to ask.
"""

import argparse
import asyncio
import json
import time

from pydantic import BaseModel, Field

from knowledge_assistant import telemetry
from knowledge_assistant.agent.prompts import untrusted_block
from knowledge_assistant.config import get_settings
from knowledge_assistant.evaluation.prompts import MAX_QUESTIONS_PER_DOC, QUESTION_GEN_SYSTEM
from knowledge_assistant.ingestion.loader import PdfLoadError, load_pdf
from knowledge_assistant.ingestion.pipeline import load_manifest
from knowledge_assistant.log import get_logger, new_trace_id
from knowledge_assistant.models import DocumentMeta

logger = get_logger(__name__)

MAX_DOC_CHARS = 8000


class QuestionsOutput(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        description=(
            "Questions an employee would plausibly ask that this document can answer, "
            "phrased in the user's own words rather than the document's wording."
        ),
    )


class EvalQuestion(BaseModel):
    question: str
    source_doc_id: str
    access_roles: list[str]


def load_pool() -> list[EvalQuestion]:
    path = get_settings().question_bank_dir / "questions.json"
    if not path.exists():
        return []
    return [EvalQuestion.model_validate(q) for q in json.loads(path.read_text())]


def select_targets(
    docs: list[DocumentMeta],
    existing: list[EvalQuestion],
    regen_all: bool,
    doc_ids: list[str] | None,
) -> list[DocumentMeta]:
    if regen_all:
        return list(docs)
    if doc_ids:
        by_id = {d.doc_id: d for d in docs}
        unknown = set(doc_ids) - by_id.keys()
        if unknown:
            raise ValueError(f"unknown doc id(s), not in manifest: {sorted(unknown)}")
        return [by_id[i] for i in doc_ids]
    covered = {q.source_doc_id for q in existing}
    return [d for d in docs if d.doc_id not in covered]


def _doc_parts(doc_text_pages: list[str]) -> list[str]:
    """Page-aligned parts of at most MAX_DOC_CHARS; char-split oversized pages."""
    parts: list[str] = []
    current: list[str] = []
    size = 0
    for page_text in doc_text_pages:
        while len(page_text) > MAX_DOC_CHARS:
            if current:
                parts.append("\n".join(current))
                current, size = [], 0
            parts.append(page_text[:MAX_DOC_CHARS])
            page_text = page_text[MAX_DOC_CHARS:]
        if current and size + len(page_text) > MAX_DOC_CHARS:
            parts.append("\n".join(current))
            current, size = [], 0
        current.append(page_text)
        size += len(page_text) + 1
    if current:
        parts.append("\n".join(current))
    return parts


async def generate_for_doc(doc: DocumentMeta, doc_text_pages: list[str]) -> list[EvalQuestion]:
    questions: list[str] = []
    for part in _doc_parts(doc_text_pages):
        messages = [
            {"role": "system", "content": QUESTION_GEN_SYSTEM},
            {"role": "user", "content": f'Document: "{doc.title}"\n\n{untrusted_block(part)}'},
        ]
        out = await telemetry.chat_parse("eval_question_gen", messages, QuestionsOutput)
        questions.extend(out.questions)
        if len(questions) >= MAX_QUESTIONS_PER_DOC:
            break
    return [
        EvalQuestion(question=q, source_doc_id=doc.doc_id, access_roles=list(doc.access))
        for q in questions[:MAX_QUESTIONS_PER_DOC]
    ]


async def run(regen_all: bool = False, doc_ids: list[str] | None = None) -> dict:
    s = get_settings()
    new_trace_id()
    telemetry.start_request()
    t0 = time.perf_counter()

    docs = load_manifest().documents
    existing = load_pool()
    targets = select_targets(docs, existing, regen_all, doc_ids)
    report: dict = {"generated": {}, "skipped": [], "pool_size": len(existing)}
    if not targets:
        logger.info("question_gen_nothing_to_do", extra={"pool_size": len(existing)})
        report["note"] = "all documents already have questions; use --all or --doc to regenerate"
        return report

    new_questions: list[EvalQuestion] = []
    for doc in targets:
        try:
            pages = load_pdf(s.pdf_dir / doc.path)
        except (PdfLoadError, FileNotFoundError) as exc:
            logger.warning("eval_doc_skipped", extra={"doc_id": doc.doc_id, "error": str(exc)})
            report["skipped"].append(doc.path)
            continue
        qs = await generate_for_doc(doc, [p.text for p in pages])
        logger.info("eval_questions_generated", extra={"doc_id": doc.doc_id, "n": len(qs)})
        new_questions.extend(qs)

    # Quality-gate only the newly generated questions before they enter the pool.
    from knowledge_assistant.evaluation.quality import filter_questions

    kept = await filter_questions(new_questions)
    target_ids = {d.doc_id for d in targets}
    pool = [q for q in existing if q.source_doc_id not in target_ids] + kept

    s.question_bank_dir.mkdir(parents=True, exist_ok=True)
    (s.question_bank_dir / "questions.json").write_text(
        json.dumps([q.model_dump() for q in pool], indent=2)
    )
    meta = telemetry.summary((time.perf_counter() - t0) * 1000)
    for doc in targets:
        report["generated"][doc.doc_id] = sum(1 for q in kept if q.source_doc_id == doc.doc_id)
    report.update(pool_size=len(pool), cost_usd=meta.cost_usd, duration_ms=meta.latency_ms)
    logger.info("question_gen_complete", extra=report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate/refresh the eval question pool")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--all", action="store_true", help="regenerate the entire pool")
    mode.add_argument(
        "--doc",
        action="append",
        metavar="DOC_ID",
        help="regenerate for a specific document (repeatable), e.g. u_maria/brand-guidelines-v3.pdf",
    )
    args = parser.parse_args()
    report = asyncio.run(run(regen_all=args.all, doc_ids=args.doc))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
