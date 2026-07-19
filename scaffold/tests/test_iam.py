import pytest

from knowledge_assistant.iam.service import AuthenticationError, known_roles, resolve_token


def test_resolves_known_token():
    user = resolve_token("tok_marketing_demo")
    assert user.id == "u_maria"
    assert user.roles == ["marketing"]


def test_multi_role_user():
    assert set(resolve_token("tok_sales_demo").roles) == {"sales", "marketing"}


def test_unknown_token_rejected():
    with pytest.raises(AuthenticationError):
        resolve_token("tok_forged")


def test_blank_token_rejected():
    with pytest.raises(AuthenticationError):
        resolve_token("  ")


def test_known_roles_from_file():
    assert known_roles() == {"marketing", "sales", "ops", "people", "finance", "exec"}
