"""The non-negotiable suite: no user may ever receive a chunk whose ACL
excludes all of their roles — across both retrieval passes, for every
user × question combination, exec-only chunk and archived handling included."""

from knowledge_assistant.retrieval import search_accessible
from tests.conftest import DIM

QUERY = [1.0] * DIM


def _search(store, roles, **kw):
    chunks, scope = search_accessible(store, roles, QUERY, top_k=50, **kw)
    return chunks, scope


def test_no_unauthorized_chunk_for_any_user(populated_store, users_file):
    for user in users_file.users:
        chunks, _ = _search(populated_store, user.roles)
        for c in chunks:
            assert set(c.access_roles) & set(user.roles), (
                f"{user.id} received unauthorized chunk {c.chunk_id} (acl={c.access_roles})"
            )


def test_restricted_docs_invisible_to_outsiders(populated_store, users_file, manifest):
    for user in users_file.users:
        chunks, _ = _search(populated_store, user.roles)
        returned_docs = {c.doc_id for c in chunks}
        for doc in manifest.documents:
            if not set(doc.access) & set(user.roles):
                assert doc.path not in returned_docs, (
                    f"{user.id} can see restricted doc {doc.path}"
                )


def test_exec_only_chunk_in_global_doc(populated_store, users_file):
    exec_chunk = "general/all-hands-2025-q2.pdf:2:0"
    for user in users_file.users:
        chunks, _ = _search(populated_store, user.roles)
        ids = {c.chunk_id for c in chunks}
        if "exec" in user.roles:
            assert exec_chunk in ids
        else:
            assert exec_chunk not in ids, f"{user.id} sees the exec-only paragraph"


def test_archived_excluded_by_default(populated_store):
    chunks, _ = _search(populated_store, ["marketing"])
    assert all(c.status != "archived" for c in chunks)
    chunks, _ = _search(populated_store, ["marketing"], include_archived=True)
    assert any(c.status == "archived" for c in chunks)  # brand-guidelines v2


def test_merged_scope_returns_team_and_global(populated_store):
    chunks, scope = _search(populated_store, ["marketing"])
    assert scope == "both"
    assert any(not c.is_global for c in chunks) and any(c.is_global for c in chunks)


def test_conflicting_team_and_global_docs_both_retrieved(populated_store):
    # The conflict-flag prerequisite: a team doc and a disagreeing global doc
    # must BOTH reach the generator (Sam: $79 playbook vs $99 pricing sheet).
    chunks, _ = _search(populated_store, ["sales", "marketing"])
    docs = {c.doc_id for c in chunks}
    assert "u_sam/sales-playbook.pdf" in docs
    assert "general/product-pricing.pdf" in docs


def test_global_only_role_still_acl_filtered(populated_store):
    # A role with no team-specific docs in this corpus receives only global
    # chunks (never someone else's team docs).
    chunks, scope = _search(populated_store, ["ops"])
    assert scope == "global"
    assert chunks, "ops should still see company-wide docs"
    assert all(c.is_global for c in chunks)


def test_acl_filter_is_required():
    """search() cannot be called without acl_roles — the unfiltered path must not exist."""
    import inspect

    from knowledge_assistant.vectorstore.base import VectorStore

    sig = inspect.signature(VectorStore.search)
    assert sig.parameters["acl_roles"].default is inspect.Parameter.empty
