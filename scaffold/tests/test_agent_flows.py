from knowledge_assistant.agent import reply_logic
from knowledge_assistant.agent.generator import GeneratorOutput, RawCitation, validate_citations
from knowledge_assistant.models import Chunk


def _chunk(chunk_id: str, status: str = "current") -> Chunk:
    return Chunk(
        chunk_id=chunk_id, doc_id="d", title="T", page=1, seq=0, text="evidence",
        access_roles=["marketing"], is_global=False, period="2025-01",
        source="notion", status=status,
    )


def test_no_result_and_insufficient_evidence_wording_identical():
    # Uniform message: "doesn't exist" and "not entitled" must be indistinguishable.
    assert reply_logic.no_result().text == reply_logic.insufficient_evidence().text


def test_unknown_citation_dropped():
    out = GeneratorOutput(
        answer="x", citations=[RawCitation(chunk_id="d:1:0", quote="q"),
                               RawCitation(chunk_id="forged:9:9", quote="q")],
    )
    citations, grounded = validate_citations(out, [_chunk("d:1:0")])
    assert [c.chunk_id for c in citations] == ["d:1:0"]
    assert grounded


def test_zero_valid_citations_degrades():
    out = GeneratorOutput(answer="x", citations=[RawCitation(chunk_id="forged:1:0", quote="q")])
    citations, grounded = validate_citations(out, [_chunk("d:1:0")])
    assert citations == [] and grounded is False


def test_insufficient_evidence_not_grounded():
    out = GeneratorOutput(answer="", citations=[RawCitation(chunk_id="d:1:0", quote="q")],
                          insufficient_evidence=True)
    _, grounded = validate_citations(out, [_chunk("d:1:0")])
    assert grounded is False


def test_error_reply_carries_trace_ref_only():
    ans = reply_logic.internal_error("abc123")
    assert "abc123" in ans.text
    assert "Traceback" not in ans.text


def test_last_assistant_kind_reads_most_recent():
    from knowledge_assistant.agent.orchestrator import _last_assistant_kind

    history = [
        {"role": "user", "content": "i need you age"},
        {"role": "assistant", "content": "Whose age?", "kind": "clarify"},
        {"role": "user", "content": "sam"},
    ]
    assert _last_assistant_kind(history) == "clarify"
    assert _last_assistant_kind([]) is None
    assert _last_assistant_kind(None) is None


def test_clarify_exhausted_is_settled_not_clarify():
    ans = reply_logic.clarify_exhausted()
    assert ans.kind == "out_of_domain"
    assert "internal company knowledge" in ans.text
