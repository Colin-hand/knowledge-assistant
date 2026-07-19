from pathlib import Path

import pytest

from knowledge_assistant.config import get_settings
from knowledge_assistant.ingestion.chunker import MAX_CHARS, chunk_pages
from knowledge_assistant.ingestion.enricher import enrich
from knowledge_assistant.ingestion.loader import PageText, load_pdf
from knowledge_assistant.ingestion.markers import EXEC_MARKER
from knowledge_assistant.models import DocumentMeta

ALL_ROLES = ["marketing", "sales", "ops", "people", "finance", "exec"]


def _doc(access=None, path="general/test.pdf") -> DocumentMeta:
    return DocumentMeta(
        path=path, title="Test Doc", access=access or ALL_ROLES,
        period="2025-01", source="notion", status="current",
    )


def test_chunks_never_cross_pages():
    pages = [PageText(1, "Title\nbody text one."), PageText(2, "more body text.")]
    chunks = chunk_pages(pages)
    assert {c.page for c in chunks} == {1, 2}


def test_chunk_size_hard_capped():
    long_text = "\n".join(f"this is content line {i} that ends with a period." * 6 for i in range(40))
    chunks = chunk_pages([PageText(1, long_text)])
    assert len(chunks) > 1
    assert all(len(c.text) <= MAX_CHARS for c in chunks)


def test_heading_oversplit_repacked():
    # Table rows look heading-like; packing must merge them back into one chunk.
    table = "Pricing\n" + "\n".join(f"Plan{i} $49 up to 5 Email" for i in range(10))
    chunks = chunk_pages([PageText(1, table)])
    assert len(chunks) == 1


def test_exec_marker_variants():
    for dash in ("—", "–", "-"):
        assert EXEC_MARKER.search(f"CONFIDENTIAL {dash} EXECUTIVE COMMITTEE ONLY")
    assert EXEC_MARKER.search("confidential — executive committee only")
    assert not EXEC_MARKER.search("confidential marketing plan")


def test_marker_section_isolated_and_demoted():
    page = PageText(1, "\n".join([
        "Quarterly Notes",
        "General update everyone may read.",
        "CONFIDENTIAL — EXECUTIVE COMMITTEE ONLY. Do not distribute.",
        "The secret acquisition detail.",
    ]))
    chunks = enrich(_doc(), chunk_pages([page]))
    marked = [c for c in chunks if "secret acquisition" in c.text]
    unmarked = [c for c in chunks if "General update" in c.text]
    assert marked and unmarked and marked[0].chunk_id != unmarked[0].chunk_id
    assert marked[0].access_roles == ["exec"]
    assert marked[0].is_global is False  # derived AFTER the override
    assert set(unmarked[0].access_roles) == set(ALL_ROLES)
    assert unmarked[0].is_global is True


def test_enrich_rejects_unknown_role():
    with pytest.raises(ValueError, match="unknown role"):
        enrich(_doc(access=["marketing", "wizards"]), chunk_pages([PageText(1, "hi")]))


# --- Regression tests against the real corpus ---

def _real_pdf(rel: str) -> Path:
    return get_settings().pdf_dir / rel


def test_ligatures_normalized_in_real_pdf():
    pages = load_pdf(_real_pdf("general/all-hands-2025-q2.pdf"))
    assert "ﬁ" not in pages[0].text  # ﬁ ligature folded
    assert "Classification" in pages[0].text
    assert "Confluence" in pages[0].text


def test_all_hands_exec_paragraph_isolated():
    """Non-exec staff must keep the company-wide sections of the all-hands doc;
    only the confidential paragraph may be demoted to exec."""
    doc = _doc(path="general/all-hands-2025-q2.pdf")
    chunks = enrich(doc, chunk_pages(load_pdf(_real_pdf(doc.path))))
    exec_chunks = [c for c in chunks if c.access_roles == ["exec"]]
    global_chunks = [c for c in chunks if c.is_global]
    assert len(exec_chunks) == 1
    assert "Project Atlas" in exec_chunks[0].text
    assert global_chunks, "general sections must remain company-wide"
    general_text = " ".join(c.text for c in global_chunks)
    assert "Q3 priorities" in general_text
    assert "Project Atlas" not in general_text
