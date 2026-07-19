"""KB evaluation pipeline. CLI:

    python -m knowledge_assistant.evaluation.evaluate [--mode retrieval|e2e]
                                                      [--num-questions N] [--seed S]

Requires a question pool (run question_gen first). By default every pooled
question is evaluated; --num-questions N tests a random sample of N from the
pool (--seed makes the sample reproducible).

Modes:
  retrieval (default) — measures the retrieval layer directly (below).
  e2e — simulates a real user: each question runs through the FULL app path
        (intent rewrite → MCP boundary → compressor → generator) via
        orchestrator.answer(token, question), as the entitled user and as
        every unentitled user. Metrics: citation hit-rate, answer-kind
        distribution, flag counts, per-question cost/latency; access check =
        no citation to a document the user cannot read. EXPENSIVE: ~10 LLM
        calls per question per user — prefer a small --num-questions sample.

Retrieval-mode flow, per sampled question:
  1. retrieve top-k chunks from the live vector store as an entitled user
     → hit-rate@k + MRR against the question's source document
  2. reverse each retrieved chunk into up to 5 questions it can answer
     (LLM, cached per chunk_id — a chunk is reversed once per run)
  3. compare the input question against each reversed question by embedding
     cosine → per-chunk max similarity, per-query "reversed relevancy"
     (mean of per-chunk max) — measures whether what came back can answer
     what was asked, free of chunk-vocabulary bias
  4. access control: retry the same query as every unentitled user; any chunk
     from the source doc, or any chunk whose ACL excludes them, fails the run
     (non-zero exit)

Outputs in eval/runs/ (inputs for later visualization):
  report.json              — summary metrics + cost/duration
  results.json             — per-query detail (retrieved chunks, scores,
                             reversed questions, similarities, access sweep)
  reversed_questions.json  — chunk_id → reversed questions cache
TODO: MLflow run tracking; a dashboard can read results.json as-is.
"""

import argparse
import asyncio
import json
import math
import random
import sys
import time
from collections import Counter

from pydantic import BaseModel, Field

from knowledge_assistant import retrieval, telemetry
from knowledge_assistant.agent.prompts import untrusted_block
from knowledge_assistant.config import get_settings
from knowledge_assistant.evaluation.prompts import REVERSE_QUESTIONS_SYSTEM
from knowledge_assistant.evaluation.question_gen import load_pool
from knowledge_assistant.iam.service import all_users
from knowledge_assistant.log import get_logger, new_trace_id
from knowledge_assistant.models import Chunk, User

logger = get_logger(__name__)


class ReversedQuestions(BaseModel):
    questions: list[str] = Field(
        default_factory=list,
        description="Questions this excerpt can directly and fully answer.",
    )


def _entitled_user(roles: list[str]) -> User | None:
    return next((u for u in all_users() if set(u.roles) & set(roles)), None)


def _unentitled_users(roles: list[str]) -> list[User]:
    return [u for u in all_users() if not set(u.roles) & set(roles)]


def _cosine(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    na, nb = math.sqrt(sum(x * x for x in a)), math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0


class _ReverseCache:
    """chunk_id → (reversed questions, their embeddings); one LLM call per chunk per run."""

    def __init__(self) -> None:
        self.questions: dict[str, list[str]] = {}
        self.vectors: dict[str, list[list[float]]] = {}

    async def get(self, chunk: Chunk) -> tuple[list[str], list[list[float]]]:
        if chunk.chunk_id not in self.questions:
            messages = [
                {"role": "system", "content": REVERSE_QUESTIONS_SYSTEM},
                {
                    "role": "user",
                    "content": f'Excerpt from "{chunk.title}" (page {chunk.page}):\n'
                    f"{untrusted_block(chunk.text)}",
                },
            ]
            out = await telemetry.chat_parse("eval_reverse", messages, ReversedQuestions)
            self.questions[chunk.chunk_id] = out.questions
            self.vectors[chunk.chunk_id] = (
                await telemetry.embed("eval_embed", out.questions) if out.questions else []
            )
        return self.questions[chunk.chunk_id], self.vectors[chunk.chunk_id]


async def _run_e2e(questions: list, pool_size: int, seed: int | None) -> dict:
    """Simulate real app usage: full agent path per question, per user."""
    from knowledge_assistant.agent import orchestrator
    from knowledge_assistant.ingestion.pipeline import load_manifest

    s = get_settings()
    access_by_doc = {d.doc_id: set(d.access) for d in load_manifest().documents}
    results: list[dict] = []
    hits: list[bool] = []
    kinds: Counter = Counter()
    flag_counts: Counter = Counter()
    latencies: list[float] = []
    total_cost = 0.0
    access_failures: list[dict] = []
    t_start = time.perf_counter()

    for q in questions:
        user = _entitled_user(q.access_roles)
        if user is None:
            logger.warning("no_entitled_user", extra={"question": q.question})
            continue

        answer = await orchestrator.answer(user.token, q.question)
        cited_docs = sorted({c.doc_id for c in answer.citations})
        hit = answer.kind == "answered" and q.source_doc_id in cited_docs
        hits.append(hit)
        kinds[answer.kind] += 1
        flag_counts.update(answer.flags)
        if answer.meta:
            total_cost += answer.meta.cost_usd
            latencies.append(answer.meta.latency_ms)

        # Access check: an unentitled user must never receive a citation to a
        # document they cannot read (answers from their own accessible docs are fine).
        leaks = []
        for outsider in _unentitled_users(q.access_roles):
            o_answer = await orchestrator.answer(outsider.token, q.question)
            if o_answer.meta:
                total_cost += o_answer.meta.cost_usd
            bad = sorted(
                {
                    c.doc_id
                    for c in o_answer.citations
                    if not access_by_doc.get(c.doc_id, set()) & set(outsider.roles)
                }
            )
            if bad:
                leaks.append({"user": outsider.id, "cited_restricted_docs": bad,
                              "kind": o_answer.kind})
        if leaks:
            access_failures.append({"question": q.question, "leaks": leaks})

        results.append(
            {
                "question": q.question,
                "source_doc_id": q.source_doc_id,
                "entitled_user": user.id,
                "kind": answer.kind,
                "flags": answer.flags,
                "answer": answer.text,
                "cited_docs": cited_docs,
                "citation_hit": hit,
                "cost_usd": answer.meta.cost_usd if answer.meta else None,
                "latency_ms": answer.meta.latency_ms if answer.meta else None,
                "stage_breakdown": answer.meta.stage_breakdown if answer.meta else None,
                "access_leaks": leaks,
            }
        )

    report = {
        "mode": "e2e",
        "pool_size": pool_size,
        "n_questions": len(results),
        "seed": seed,
        "citation_hit_rate": round(sum(hits) / len(hits), 3) if hits else None,
        "kind_distribution": dict(kinds),
        "flag_counts": dict(flag_counts),
        "avg_latency_ms": round(sum(latencies) / len(latencies), 1) if latencies else None,
        "access_control": {"passed": not access_failures, "failures": access_failures},
        "cost_usd": round(total_cost, 6),
        "duration_ms": round((time.perf_counter() - t_start) * 1000, 1),
    }
    s.eval_runs_dir.mkdir(parents=True, exist_ok=True)
    (s.eval_runs_dir / "report.json").write_text(json.dumps(report, indent=2))
    (s.eval_runs_dir / "results.json").write_text(json.dumps(results, indent=2))
    logger.info("evaluation_complete", extra={"report": report})
    return report


async def run(
    num_questions: int | None = None, seed: int | None = None, mode: str = "retrieval"
) -> dict:
    s = get_settings()
    new_trace_id()
    telemetry.start_request()
    t_start = time.perf_counter()

    pool = load_pool()
    if not pool:
        raise RuntimeError(
            "question pool is empty — run `python -m knowledge_assistant.evaluation.question_gen` first"
        )
    if num_questions is not None and num_questions < len(pool):
        questions = random.Random(seed).sample(pool, num_questions)
    else:
        questions = pool

    if mode == "e2e":
        return await _run_e2e(questions, pool_size=len(pool), seed=seed)

    from knowledge_assistant.vectorstore.pinecone_store import get_store

    store = get_store()

    q_vectors = await telemetry.embed("eval_embed", [q.question for q in questions])
    cache = _ReverseCache()
    results: list[dict] = []
    hits, rr, relevancies = [], [], []
    access_failures: list[dict] = []

    for q, q_vec in zip(questions, q_vectors):
        user = _entitled_user(q.access_roles)
        if user is None:
            logger.warning("no_entitled_user", extra={"question": q.question})
            continue

        # 1. retrieval as an entitled user
        chunks, scope = retrieval.search_accessible(store, user.roles, q_vec)
        returned_docs = [c.doc_id for c in chunks]
        hit = q.source_doc_id in returned_docs
        hits.append(hit)
        rank = returned_docs.index(q.source_doc_id) + 1 if hit else None
        rr.append(1.0 / rank if rank else 0.0)

        # 2 + 3. reverse retrieved chunks into questions, compare by cosine
        chunk_results = []
        for chunk in chunks:
            reversed_qs, reversed_vecs = await cache.get(chunk)
            sims = [round(_cosine(q_vec, rv), 4) for rv in reversed_vecs]
            chunk_results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "retrieval_score": chunk.score,
                    "reversed_questions": reversed_qs,
                    "similarities": sims,
                    "max_similarity": max(sims) if sims else 0.0,
                }
            )
        relevancy = (
            sum(c["max_similarity"] for c in chunk_results) / len(chunk_results)
            if chunk_results
            else 0.0
        )
        relevancies.append(relevancy)

        # 4. access control sweep
        leaks = []
        for outsider in _unentitled_users(q.access_roles):
            leaked_chunks, _ = retrieval.search_accessible(store, outsider.roles, q_vec)
            leaked = [c.chunk_id for c in leaked_chunks if c.doc_id == q.source_doc_id]
            acl_violations = [
                c.chunk_id
                for c in leaked_chunks
                if not set(c.access_roles) & set(outsider.roles)
            ]
            if leaked or acl_violations:
                leaks.append(
                    {"user": outsider.id, "leaked": leaked, "acl_violations": acl_violations}
                )
        if leaks:
            access_failures.append({"question": q.question, "leaks": leaks})

        results.append(
            {
                "question": q.question,
                "source_doc_id": q.source_doc_id,
                "access_roles": q.access_roles,
                "entitled_user": user.id,
                "scope": scope,
                "hit": hit,
                "rank": rank,
                "reversed_relevancy": round(relevancy, 4),
                "chunks": chunk_results,
                "access_leaks": leaks,
            }
        )

    meta = telemetry.summary((time.perf_counter() - t_start) * 1000)
    report = {
        "mode": "retrieval",
        "pool_size": len(pool),
        "n_questions": len(results),
        "seed": seed,
        "retrieval": {
            "hit_rate_at_k": round(sum(hits) / len(hits), 3) if hits else None,
            "mrr": round(sum(rr) / len(rr), 3) if rr else None,
        },
        "reversed_relevancy_mean": (
            round(sum(relevancies) / len(relevancies), 3) if relevancies else None
        ),
        "access_control": {"passed": not access_failures, "failures": access_failures},
        "cost_usd": meta.cost_usd,
        "duration_ms": meta.latency_ms,
    }
    s.eval_runs_dir.mkdir(parents=True, exist_ok=True)
    (s.eval_runs_dir / "report.json").write_text(json.dumps(report, indent=2))
    (s.eval_runs_dir / "results.json").write_text(json.dumps(results, indent=2))
    (s.eval_runs_dir / "reversed_questions.json").write_text(json.dumps(cache.questions, indent=2))
    logger.info("evaluation_complete", extra={"report": report})
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the KB evaluation pipeline")
    parser.add_argument(
        "--mode",
        choices=["retrieval", "e2e"],
        default="retrieval",
        help="retrieval: measure the retrieval layer directly (default); "
        "e2e: simulate real users through the full agent path (expensive — "
        "use with --num-questions)",
    )
    parser.add_argument(
        "--num-questions",
        type=int,
        default=None,
        metavar="N",
        help="evaluate a random sample of N questions from the pool (default: all)",
    )
    parser.add_argument(
        "--seed", type=int, default=None, help="random seed for reproducible sampling"
    )
    args = parser.parse_args()
    report = asyncio.run(run(num_questions=args.num_questions, seed=args.seed, mode=args.mode))
    print(json.dumps(report, indent=2))
    if not report["access_control"]["passed"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
