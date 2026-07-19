"""Per-chunk relevance filtering, invoked in parallel.

One LLM call per chunk extracts verbatim-relevant sentences; irrelevant
chunks are dropped. Results map back to full chunk metadata by chunk_id;
`text` becomes the verbatim extract.

TODO (scale): batch chunks per call or swap in a cross-encoder reranker.
"""

import asyncio

from pydantic import BaseModel, Field

from knowledge_assistant import telemetry
from knowledge_assistant.agent.prompts import COMPRESSOR_SYSTEM, untrusted_block
from knowledge_assistant.config import get_settings
from knowledge_assistant.log import get_logger
from knowledge_assistant.models import Chunk

logger = get_logger(__name__)


class CompressResult(BaseModel):
    has_relevant_content: bool = Field(
        description="False if nothing in the excerpt bears on the question."
    )
    extract: str = Field(
        description=(
            "The relevant sentences copied word-for-word from the excerpt. "
            "Empty when has_relevant_content is false."
        )
    )


async def _compress_one(query: str, chunk: Chunk, sem: asyncio.Semaphore) -> Chunk | None:
    async with sem:
        messages = [
            {"role": "system", "content": COMPRESSOR_SYSTEM},
            {
                "role": "user",
                "content": f"Question: {query}\n\nExcerpt from \"{chunk.title}\" "
                f"(page {chunk.page}):\n{untrusted_block(chunk.text)}",
            },
        ]
        try:
            result = await telemetry.chat_parse("compressor", messages, CompressResult)
        except Exception as exc:
            # Fail open per chunk: keep the original text rather than drop evidence.
            logger.warning("compressor_chunk_failed", extra={"chunk_id": chunk.chunk_id, "error": str(exc)})
            return chunk
    if not result.has_relevant_content or not result.extract.strip():
        return None
    return chunk.model_copy(update={"text": result.extract.strip()})


async def compress(query: str, chunks: list[Chunk]) -> list[Chunk]:
    sem = asyncio.Semaphore(get_settings().compressor_concurrency)
    results = await asyncio.gather(*(_compress_one(query, c, sem) for c in chunks))
    kept = [c for c in results if c is not None]
    logger.info("compressor_done", extra={"n_in": len(chunks), "n_out": len(kept)})
    return kept
