from typing import Literal

from pydantic import BaseModel, Field

from knowledge_assistant.models import AgentAnswer

# Closed set (not free text): the tone value reaches the generator system
# prompt, so it must never be a caller-controlled injection channel.
Tone = Literal["professional", "friendly", "concise"]


class ChatTurn(BaseModel):
    role: str
    content: str
    kind: str | None = None  # assistant turns: the AgentAnswer kind (e.g. "clarify")


class ChatRequest(BaseModel):
    query: str
    tone: Tone = "professional"
    history: list[ChatTurn] = Field(default_factory=list)


# The response IS the AgentAnswer (kind, text, citations, flags, meta).
ChatResponse = AgentAnswer
