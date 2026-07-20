"""KB evaluation CLI: python -m knowledge_assistant.evaluation.evaluate
[--mode retrieval|e2e] [--num-questions N] [--seed S]

retrieval: hit-rate@k, MRR, judged relevancy score, access sweep.
e2e: full agent path per question and user.
Writes report.json / results.json; exits non-zero on access leaks.
"""

import argparse
import asyncio
import json
import random
import sys
import time
from collections import Counter

from pydantic import BaseModel, Field

from knowledge_assistant import retrieval, telemetry
from knowledge_assistant.agent.prompts import untrusted_block
from knowledge_assistant.config import get_settings
from knowledge_assistant.evaluation.prompts import RELEVANCY_JUDGE_SYSTEM, REVERSE_QUESTIONS_SYSTEM
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


class RelevancyScores(BaseModel):
    scores: list[int] = Field(
        default_factory=list,
        description="One 1-5 relevancy score per candidate question, in input order.",
    )


class _ReverseCache:
    """chunk_id → reversed questions; one call per chunk."""

    def __init__(self) -> None:
        self.questions: dict[str, list[str]] = {}

    async def get(self, chunk: Chunk) -> list[str]:
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
        return self.questions[chunk.chunk_id]


async def _judge_relevancy(question: str, candidates: list[str]) -> list[int]:
    """One 1-5 score per candidate, single batched call."""
    if not candidates:
        return []
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    messages = [
        {"role": "system", "content": RELEVANCY_JUDGE_SYSTEM},
        {"role": "user", "content": f"User question: {question}\n\nCandidates:\n{numbered}"},
    ]
    out = await telemetry.chat_parse("eval_judge", messages, RelevancyScores)
    scores = [min(5, max(1, s)) for s in out.scores[: len(candidates)]]
    if len(scores) < len(candidates):  # model miscounted — score missing ones lowest
        logger.warning(
            "judge_score_count_mismatch",
            extra={"expected": len(candidates), "got": len(scores)},
        )
        scores += [1] * (len(candidates) - len(scores))
    return scores


async def _run_e2e(questions: list, pool_size: int, seed: int | None) -> dict:
    """Full agent path per question, per user."""
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

        # Unentitled users must never cite restricted documents.
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

        # 2+3. reverse chunks, judge each reversed question (1-5)
        reversed_by_chunk = [(chunk, await cache.get(chunk)) for chunk in chunks]
        flat = [rq for _, rqs in reversed_by_chunk for rq in rqs]
        flat_scores = await _judge_relevancy(q.question, flat)
        chunk_results = []
        all_scores: list[int] = []
        idx = 0
        for chunk, rqs in reversed_by_chunk:
            scores = flat_scores[idx : idx + len(rqs)]
            idx += len(rqs)
            all_scores.extend(scores)
            chunk_results.append(
                {
                    "chunk_id": chunk.chunk_id,
                    "doc_id": chunk.doc_id,
                    "retrieval_score": chunk.score,
                    "reversed_questions": rqs,
                    "judge_scores": scores,
                    "max_judge": max(scores) if scores else 0,
                }
            )
        # Relevancy score = max judge score for this question.
        relevancy = float(max(all_scores)) if all_scores else 0.0
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
                "relevancy_score": round(relevancy, 3),
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
        # Average of per-question relevancy scores.
        "relevancy_score_mean": (
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
