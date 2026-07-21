"""Templated terminal replies — no LLM involved."""

from knowledge_assistant.models import AgentAnswer

# Same wording for missing vs unauthorized — no existence leak.
NO_INFO = (
    "I couldn't find anything on that in the documents available to you. "
    "Feel free to rephrase, or ask me about something else — happy to keep looking!"
)
_INSUFFICIENT_EVIDENCE = (
    "Sorry, I couldn't find any document related to that. "
    "Try rephrasing or narrowing the question — happy to take another look!"
)

_OUT_OF_DOMAIN = (
    "That one's a bit outside my wheelhouse — I can only help with our internal "
    "company knowledge. Happy to help with things like pricing, brand guidelines, "
    "or company policies though!"
)
_REFUSED = "Sorry, that's not something I can help with — but I'm glad to answer questions about our internal documents."
_ERROR = "Sorry — something went wrong on my side. Please try again in a moment."
_INVALID_TOKEN = "It looks like your session is no longer valid. Please sign in again and I'll be right here."


def out_of_domain() -> AgentAnswer:
    return AgentAnswer(kind="out_of_domain", text=_OUT_OF_DOMAIN)


def refused() -> AgentAnswer:
    return AgentAnswer(kind="refused", text=_REFUSED)


def no_result() -> AgentAnswer:
    return AgentAnswer(kind="no_result", text=NO_INFO)


def insufficient_evidence() -> AgentAnswer:
    return AgentAnswer(kind="insufficient_evidence", text=_INSUFFICIENT_EVIDENCE)


def invalid_token() -> AgentAnswer:
    return AgentAnswer(kind="error", text=_INVALID_TOKEN)


def internal_error(trace_id: str) -> AgentAnswer:
    return AgentAnswer(kind="error", text=f"{_ERROR} (ref: {trace_id})")
