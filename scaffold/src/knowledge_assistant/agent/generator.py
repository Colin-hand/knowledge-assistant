import re

from pydantic import BaseModel, Field

from knowledge_assistant import telemetry
from knowledge_assistant.config import get_settings
from knowledge_assistant.agent.prompts import (
    DEFAULT_TONE,
    GENERATOR_SYSTEM,
    tone_section,
    untrusted_block,
)
from knowledge_assistant.log import get_logger
from knowledge_assistant.models import Chunk, Citation

logger = get_logger(__name__)

# Strip bracketed chunk-id references the model may leak into answer text.
_INLINE_REF = re.compile(r"\s*\[[^\[\]]*\.pdf:\d+:\d+[^\[\]]*\]")


class RawCitation(BaseModel):
    chunk_id: str = Field(description="The chunk_id of a provided evidence chunk.")
    quote: str = Field(description="A short verbatim supporting snippet from that chunk.")


class GeneratorOutput(BaseModel):
    answer: str = Field(
        description="The grounded answer. Empty when insufficient_evidence is true."
    )
    citations: list[RawCitation] = Field(
        default_factory=list,
        description="At least one citation per claim made in the answer.",
    )
    flags: list[str] = Field(
        default_factory=list,
        description=(
            "'conflict' when different documents disagree on a fact stated in the "
            "answer; 'inconsistent_source' when figures or statements within a "
            "single document do not reconcile; 'stale_source' when any cited "
            "chunk has status archived."
        ),
    )
    insufficient_evidence: bool = Field(
        default=False,
        description="True when the provided chunks do not support an answer.",
    )


def _render_chunk(chunk: Chunk) -> str:
    return (
        f"[chunk_id: {chunk.chunk_id}] \"{chunk.title}\" — page {chunk.page}, "
        f"period {chunk.period}, status {chunk.status}"
        + (f", superseded_by {chunk.superseded_by}" if chunk.superseded_by else "")
        + f"\n{untrusted_block(chunk.text)}"
    )


def validate_citations(output: GeneratorOutput, chunks: list[Chunk]) -> tuple[list[Citation], bool]:
    """Drop unknown citations; zero valid ones → insufficient evidence."""
    by_id = {c.chunk_id: c for c in chunks}
    valid: list[Citation] = []
    for raw in output.citations:
        chunk = by_id.get(raw.chunk_id)
        if chunk is None:
            logger.warning("citation_dropped_unknown_chunk", extra={"chunk_id": raw.chunk_id})
            continue
        valid.append(
            Citation(
                chunk_id=chunk.chunk_id,
                doc_id=chunk.doc_id,
                title=chunk.title,
                page=chunk.page,
                period=chunk.period,
                source=chunk.source,
                status=chunk.status,
                superseded_by=chunk.superseded_by,
                quote=raw.quote,
            )
        )
    grounded = bool(valid) and not output.insufficient_evidence
    return valid, grounded


async def generate(
    query: str, chunks: list[Chunk], tone: str = DEFAULT_TONE
) -> tuple[GeneratorOutput, list[Citation], bool]:
    evidence = "\n\n".join(_render_chunk(c) for c in chunks)
    messages = [
        {"role": "system", "content": GENERATOR_SYSTEM + tone_section(tone)},
        {"role": "user", "content": f"Question: {query}\n\nEvidence chunks:\n\n{evidence}"},
    ]
    output = await telemetry.chat_parse(
        "generator",
        messages,
        GeneratorOutput,
        reasoning_effort=get_settings().openai_reasoning_effort_generator,
    )
    output.answer = _INLINE_REF.sub("", output.answer).strip()
    citations, grounded = validate_citations(output, chunks)
    # Force stale_source whenever an archived chunk is cited.
    if any(c.status == "archived" for c in citations) and "stale_source" not in output.flags:
        output.flags.append("stale_source")
    return output, citations, grounded
