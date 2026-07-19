"""Bearer-token extraction. The API layer passes the token through UNRESOLVED —
role resolution happens only at the MCP boundary. Here we just reject a
missing/blank header early with 401."""

from fastapi import Depends, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

_bearer = HTTPBearer(auto_error=False)


def get_token(creds: HTTPAuthorizationCredentials | None = Depends(_bearer)) -> str:
    if creds is None or not creds.credentials.strip():
        raise HTTPException(status_code=401, detail="missing bearer token")
    return creds.credentials
