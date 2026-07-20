from typing import Any, Literal

from pydantic import BaseModel, Field


class User(BaseModel):
    id: str
    name: str
    token: str
    roles: list[str]


class UsersFile(BaseModel):
    roles: list[str]
    users: list[User]


class DocumentMeta(BaseModel):
    path: str
    title: str
    access: list[str]
    period: str
    source: str
    status: Literal["current", "archived"]
    note: str | None = None
    supersedes: str | None = None
    superseded_by: str | None = None

    @property
    def doc_id(self) -> str:
        return self.path


class Manifest(BaseModel):
    documents: list[DocumentMeta]


class Chunk(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    page: int
    seq: int
    text: str
    access_roles: list[str]
    is_global: bool
    period: str
    source: str
    status: str
    superseded_by: str | None = None
    score: float | None = None


class Citation(BaseModel):
    chunk_id: str
    doc_id: str
    title: str
    page: int
    period: str
    source: str
    status: str
    superseded_by: str | None = None
    quote: str


AnswerKind = Literal[
    "answered", "clarify", "no_result", "refused", "greeting", "out_of_domain", "error"
]


class RequestMeta(BaseModel):
    trace_id: str = "-"
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    latency_ms: float = 0.0
    stage_breakdown: dict[str, float] = Field(default_factory=dict)


class AgentAnswer(BaseModel):
    kind: AnswerKind
    text: str
    citations: list[Citation] = Field(default_factory=list)
    flags: list[str] = Field(default_factory=list)
    meta: RequestMeta | None = None


class SearchResponse(BaseModel):
    """Payload of the search_knowledge MCP tool."""

    status: Literal["ok", "no_result", "error"]
    scope: Literal["team", "global", "both", "none"] = "none"
    chunks: list[Chunk] = Field(default_factory=list)
    error: str | None = None

    def dump(self) -> dict[str, Any]:
        return self.model_dump(mode="json")
