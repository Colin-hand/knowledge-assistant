"""LLM judge drops generated questions below the quality threshold."""

import asyncio

from pydantic import BaseModel, Field

from knowledge_assistant import telemetry
from knowledge_assistant.evaluation.prompts import QUALITY_SYSTEM
from knowledge_assistant.evaluation.question_gen import EvalQuestion
from knowledge_assistant.log import get_logger

logger = get_logger(__name__)

THRESHOLD = 4


class QualityScore(BaseModel):
    answerable: int = Field(
        description="1–5: could the named source document plausibly contain a concrete answer?"
    )
    specific: int = Field(
        description=(
            "1–5: is the question precise enough that relevant text is identifiable "
            "(5 = one clear fact, 1 = vague/broad)?"
        )
    )


async def judge(question: EvalQuestion) -> float:
    messages = [
        {"role": "system", "content": QUALITY_SYSTEM},
        {
            "role": "user",
            "content": f'Question: "{question.question}"\nSource document: {question.source_doc_id}',
        },
    ]
    score = await telemetry.chat_parse("eval_quality", messages, QualityScore)
    return (score.answerable + score.specific) / 2


async def filter_questions(
    questions: list[EvalQuestion], concurrency: int = 8
) -> list[EvalQuestion]:
    sem = asyncio.Semaphore(concurrency)

    async def scored(q: EvalQuestion) -> tuple[EvalQuestion, float]:
        async with sem:
            return q, await judge(q)

    results = await asyncio.gather(*(scored(q) for q in questions))
    kept = [q for q, sc in results if sc >= THRESHOLD]
    logger.info("eval_quality_filtered", extra={"n_in": len(questions), "n_kept": len(kept)})
    return kept
