"""Wraps every OpenAI call with token, cost, and latency accounting.

All LLM/embedding traffic in the codebase goes through `chat_parse` / `embed`
so that per-request summaries (returned in the API `meta` field) and the
JSONL log always agree. Prices come from env config — never hardcoded.
"""

import time
from contextvars import ContextVar
from dataclasses import dataclass, field

from openai import AsyncOpenAI
from pydantic import BaseModel

from knowledge_assistant.config import get_settings
from knowledge_assistant.log import current_trace_id, get_logger
from knowledge_assistant.models import RequestMeta

logger = get_logger(__name__)
_client: AsyncOpenAI | None = None


def client() -> AsyncOpenAI:
    global _client
    if _client is None:
        _client = AsyncOpenAI(api_key=get_settings().openai_api_key)
    return _client


@dataclass
class _Accumulator:
    llm_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    stages: dict[str, float] = field(default_factory=dict)


_acc: ContextVar[_Accumulator | None] = ContextVar("telemetry_acc", default=None)


def start_request() -> None:
    _acc.set(_Accumulator())


def record_stage(stage: str, latency_ms: float) -> None:
    acc = _acc.get()
    if acc is not None:
        acc.stages[stage] = acc.stages.get(stage, 0.0) + latency_ms


def summary(total_latency_ms: float) -> RequestMeta:
    acc = _acc.get() or _Accumulator()
    return RequestMeta(
        trace_id=current_trace_id(),
        llm_calls=acc.llm_calls,
        prompt_tokens=acc.prompt_tokens,
        completion_tokens=acc.completion_tokens,
        cost_usd=round(acc.cost_usd, 6),
        latency_ms=round(total_latency_ms, 1),
        stage_breakdown={k: round(v, 1) for k, v in acc.stages.items()},
    )


def _record(component: str, model: str, latency_ms: float, pt: int, ct: int, cost: float) -> None:
    acc = _acc.get()
    if acc is not None:
        acc.llm_calls += 1
        acc.prompt_tokens += pt
        acc.completion_tokens += ct
        acc.cost_usd += cost
    logger.info(
        "llm_call",
        extra={
            "component": component,
            "model": model,
            "latency_ms": round(latency_ms, 1),
            "prompt_tokens": pt,
            "completion_tokens": ct,
            "cost_usd": round(cost, 6),
        },
    )


async def chat_parse[T: BaseModel](component: str, messages: list[dict], schema: type[T]) -> T:
    """Structured-output chat call; returns a validated instance of `schema`."""
    s = get_settings()
    t0 = time.perf_counter()
    completion = await client().beta.chat.completions.parse(
        model=s.openai_model, messages=messages, response_format=schema
    )
    latency = (time.perf_counter() - t0) * 1000
    usage = completion.usage
    pt = usage.prompt_tokens if usage else 0
    ct = usage.completion_tokens if usage else 0
    cost = (
        pt * s.openai_model_input_price_per_1m + ct * s.openai_model_output_price_per_1m
    ) / 1_000_000
    _record(component, s.openai_model, latency, pt, ct, cost)
    parsed = completion.choices[0].message.parsed
    if parsed is None:
        raise ValueError(f"{component}: model returned no parsable output")
    return parsed


async def embed(component: str, texts: list[str]) -> list[list[float]]:
    s = get_settings()
    t0 = time.perf_counter()
    resp = await client().embeddings.create(model=s.openai_embed_model, input=texts)
    latency = (time.perf_counter() - t0) * 1000
    tokens = resp.usage.total_tokens if resp.usage else 0
    cost = tokens * s.openai_embed_price_per_1m / 1_000_000
    _record(component, s.openai_embed_model, latency, tokens, 0, cost)
    return [d.embedding for d in resp.data]
