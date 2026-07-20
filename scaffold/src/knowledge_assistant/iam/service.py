"""Token → user resolution backed by data/users.json."""

import json
from functools import lru_cache

from knowledge_assistant.config import get_settings
from knowledge_assistant.log import get_logger
from knowledge_assistant.models import User, UsersFile

logger = get_logger(__name__)


class AuthenticationError(Exception):
    pass


@lru_cache
def _load_users() -> UsersFile:
    raw = json.loads(get_settings().users_file.read_text())
    return UsersFile.model_validate(raw)


def known_roles() -> frozenset[str]:
    return frozenset(_load_users().roles)


def resolve_token(token: str) -> User:
    if not token or not token.strip():
        raise AuthenticationError("missing token")
    for user in _load_users().users:
        if user.token == token:
            logger.info("token_resolved", extra={"user_id": user.id, "roles": user.roles})
            return user
    # Never log the presented token value.
    logger.warning("token_rejected")
    raise AuthenticationError("unknown token")


def all_users() -> list[User]:
    return list(_load_users().users)
