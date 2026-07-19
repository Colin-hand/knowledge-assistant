"""Heading-aware section chunking that never crosses page boundaries.

This corpus extracts with no blank lines, so headings are the structure
signal: a short line without terminal punctuation starts a new section.
Over-detection (e.g. table rows) is harmless — consecutive sections are
re-packed greedily up to MAX_CHARS.

A confidential-marker line ALWAYS starts a section, and marker sections are
never packed with other content, so a restricted section can never share a
chunk (and thus an ACL) with general text. No overlap by design: sections
are self-contained, and tail-overlap could smear restricted text into a
chunk with a broader ACL.
"""

from dataclasses import dataclass

from knowledge_assistant.ingestion.loader import PageText
from knowledge_assistant.ingestion.markers import EXEC_MARKER

MAX_CHARS = 2000  # hard cap per chunk (≈500 tokens)
_TERMINAL_PUNCT = (".", "!", "?", ":", ";", ",")


@dataclass
class RawChunk:
    page: int
    seq: int  # sequence within the page
    text: str


def _is_heading(line: str) -> bool:
    s = line.strip()
    return 0 < len(s) < 60 and not s.endswith(_TERMINAL_PUNCT)


def _sections(text: str) -> list[str]:
    """Split page text into sections at heading-like and marker lines."""
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


def _split_oversize(section: str) -> list[str]:
    """Line-level fallback for sections exceeding the cap."""
    if len(section) <= MAX_CHARS:
        return [section]
    pieces: list[str] = []
    current: list[str] = []
    size = 0
    for line in section.split("\n"):
        while len(line) > MAX_CHARS:  # pathological single line
            pieces.append(line[:MAX_CHARS])
            line = line[MAX_CHARS:]
        if current and size + len(line) > MAX_CHARS:
            pieces.append("\n".join(current))
            current, size = [], 0
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append("\n".join(current))
    return pieces


def chunk_pages(pages: list[PageText]) -> list[RawChunk]:
    chunks: list[RawChunk] = []
    for page in pages:
        seq = 0
        packed: list[str] = []
        size = 0

        def flush() -> None:
            nonlocal seq, packed, size
            if packed:
                chunks.append(RawChunk(page=page.page, seq=seq, text="\n\n".join(packed)))
                seq += 1
                packed, size = [], 0

        for section in _sections(page.text):
            marked = bool(EXEC_MARKER.search(section))
            for piece in _split_oversize(section):
                if marked:
                    # Marker sections are isolated: flushed out alone, never packed.
                    flush()
                    chunks.append(RawChunk(page=page.page, seq=seq, text=piece))
                    seq += 1
                elif packed and size + len(piece) > MAX_CHARS:
                    flush()
                    packed, size = [piece], len(piece)
                else:
                    packed.append(piece)
                    size += len(piece) + 2
        flush()
    return chunks
