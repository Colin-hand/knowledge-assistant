import unicodedata
from dataclasses import dataclass
from pathlib import Path

from pypdf import PdfReader

from knowledge_assistant.log import get_logger

logger = get_logger(__name__)


@dataclass
class PageText:
    page: int  # 1-based
    text: str


class PdfLoadError(Exception):
    pass


def _normalize(text: str) -> str:
    # NFKC folds PDF ligatures (ﬁ ﬂ ﬀ → fi fl ff) and non-breaking spaces to
    # plain ASCII — marker matching and embeddings must see clean text.
    return unicodedata.normalize("NFKC", text)


def load_pdf(path: Path) -> list[PageText]:
    try:
        reader = PdfReader(path)
        pages = [
            PageText(page=i + 1, text=_normalize((p.extract_text() or "").strip()))
            for i, p in enumerate(reader.pages)
        ]
    except Exception as exc:  # pypdf raises a zoo of exception types
        raise PdfLoadError(f"failed to parse {path.name}: {exc}") from exc
    pages = [p for p in pages if p.text]
    if not pages:
        raise PdfLoadError(f"{path.name}: no extractable text")
    return pages
