"""Heading-aware chunking; never crosses pages; marker sections isolated."""

from dataclasses import dataclass

import tiktoken

from knowledge_assistant.config import get_settings
from knowledge_assistant.ingestion.loader import PageText
from knowledge_assistant.ingestion.markers import EXEC_MARKER

# Hard cap and sibling overlap in tokens; bound at import time.
MAX_TOKENS = get_settings().chunk_max_tokens
OVERLAP_TOKENS = get_settings().chunk_overlap_tokens
_ENC = tiktoken.get_encoding("cl100k_base")
_TERMINAL_PUNCT = (".", "!", "?", ":", ";", ",")

# Body budget reserves room so overlap never breaks the hard cap.
_BODY_BUDGET = MAX_TOKENS - OVERLAP_TOKENS - 1 if OVERLAP_TOKENS else MAX_TOKENS


def ntokens(text: str) -> int:
    return len(_ENC.encode(text))


def _tail(text: str) -> str:
    ids = _ENC.encode(text)
    return _ENC.decode(ids[-OVERLAP_TOKENS:]) if len(ids) > OVERLAP_TOKENS else text


@dataclass
class RawChunk:
    page: int
    seq: int  # sequence within the page
    text: str


def _is_heading(line: str) -> bool:
    s = line.strip()
    return 0 < len(s) < 60 and not s.endswith(_TERMINAL_PUNCT)


def _sections(text: str) -> list[str]:
    """Split page text at heading-like and marker lines."""
    sections: list[list[str]] = []
    current: list[str] = []
    for line in text.split("\n"):
        if current and (_is_heading(line) or EXEC_MARKER.search(line)):
            sections.append(current)
            current = []
        current.append(line)
    if current:
        sections.append(current)
    return [s for s in ("\n".join(sec).strip() for sec in sections) if s]


def _split_oversize(section: str, budget: int) -> list[str]:
    """Line-level fallback for sections exceeding the budget."""
    if ntokens(section) <= budget:
        return [section]
    pieces: list[str] = []
    current: list[str] = []
    for line in section.split("\n"):
        while ntokens(line) > budget:  # pathological single line
            ids = _ENC.encode(line)
            pieces.append(_ENC.decode(ids[:budget]))
            line = _ENC.decode(ids[budget:])
        if current and ntokens("\n".join([*current, line])) > budget:
            pieces.append("\n".join(current))
            current = [line]
        else:
            current.append(line)
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_pages(pages: list[PageText]) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for page in pages:
        seq = 0
        packed: list[str] = []
        prev_tail: str | None = None

        def flush() -> None:
            nonlocal seq, packed, prev_tail
            if packed:
                body = "\n\n".join(packed)
                text = f"{prev_tail}\n{body}" if prev_tail and OVERLAP_TOKENS else body
                chunks.append(RawChunk(page=page.page, seq=seq, text=text))
                seq += 1
                prev_tail = _tail(body)
                packed = []

        for section in _sections(page.text):
            if EXEC_MARKER.search(section):
                # Marker sections stay whole and alone; no overlap crosses them.
                flush()
                chunks.append(RawChunk(page=page.page, seq=seq, text=section))
                seq += 1
                prev_tail = None
                continue
            for piece in _split_oversize(section, _BODY_BUDGET):
                if packed and ntokens("\n\n".join([*packed, piece])) > _BODY_BUDGET:
                    flush()
                    packed = [piece]
                else:
                    packed.append(piece)
        flush()
    return chunks
