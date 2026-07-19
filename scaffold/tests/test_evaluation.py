import pytest

from knowledge_assistant.evaluation.question_gen import EvalQuestion, select_targets
from knowledge_assistant.models import DocumentMeta

ALL_ROLES = ["marketing", "sales", "ops", "people", "finance", "exec"]


def _doc(path: str) -> DocumentMeta:
    return DocumentMeta(
        path=path, title=path, access=ALL_ROLES,
        period="2025-01", source="notion", status="current",
    )


def _q(doc_id: str) -> EvalQuestion:
    return EvalQuestion(question="q?", source_doc_id=doc_id, access_roles=ALL_ROLES)


DOCS = [_doc("a.pdf"), _doc("b.pdf"), _doc("c.pdf")]


def test_default_mode_targets_only_uncovered_docs():
    targets = select_targets(DOCS, [_q("a.pdf")], regen_all=False, doc_ids=None)
    assert [d.doc_id for d in targets] == ["b.pdf", "c.pdf"]


def test_default_mode_nothing_when_all_covered():
    existing = [_q("a.pdf"), _q("b.pdf"), _q("c.pdf")]
    assert select_targets(DOCS, existing, regen_all=False, doc_ids=None) == []


def test_all_mode_targets_everything():
    targets = select_targets(DOCS, [_q("a.pdf")], regen_all=True, doc_ids=None)
    assert [d.doc_id for d in targets] == ["a.pdf", "b.pdf", "c.pdf"]


def test_doc_mode_targets_named_docs():
    targets = select_targets(DOCS, [], regen_all=False, doc_ids=["b.pdf"])
    assert [d.doc_id for d in targets] == ["b.pdf"]


def test_doc_mode_rejects_unknown_id():
    with pytest.raises(ValueError, match="unknown doc id"):
        select_targets(DOCS, [], regen_all=False, doc_ids=["nope.pdf"])
