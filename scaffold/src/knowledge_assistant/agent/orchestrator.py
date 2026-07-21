"""Agent orchestrator: intent → MCP search → compress → generate → reply."""

import json
import sys
import time

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import get_default_environment, stdio_client

from knowledge_assistant import progress, telemetry
from knowledge_assistant.agent import compressor, generator, greeter, intent, reply_logic
from knowledge_assistant.agent.prompts import DEFAULT_TONE
from knowledge_assistant.iam.service import AuthenticationError, resolve_token
from knowledge_assistant.log import current_trace_id, get_logger
from knowledge_assistant.models import AgentAnswer, SearchResponse

logger = get_logger(__name__)

def _server_params() -> StdioServerParameters:
    return StdioServerParameters(
        command=sys.executable,
        args=["-m", "knowledge_assistant.mcp_server.server"],
        env={**get_default_environment(), "KA_TRACE_ID": current_trace_id()},
    )


def _last_assistant_kind(history: list[dict] | None) -> str | None:
    for turn in reversed(history or []):
        if turn.get("role") == "assistant":
            return turn.get("kind")
    return None


async def _call_search(token: str, query: str) -> SearchResponse:
    async with stdio_client(_server_params()) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(
                "search_knowledge", {"token": token, "query": query}
            )
    payload = result.structuredContent
    if payload is None:
        payload = json.loads(result.content[0].text)
    if "result" in payload and "status" not in payload:  # FastMCP wraps bare dicts
        payload = payload["result"]
    if payload.get("status") == "error":
        if payload.get("error") == "invalid_token":
            raise PermissionError("invalid_token")
        raise RuntimeError(f"mcp error: {payload.get('error')}")
    return SearchResponse.model_validate(payload)


async def answer(
    token: str,
    query: str,
    history: list[dict] | None = None,
    tone: str = DEFAULT_TONE,
    progress_id: str | None = None,
) -> AgentAnswer:
    telemetry.start_request()
    t_start = time.perf_counter()

    def _finish(ans: AgentAnswer) -> AgentAnswer:
        progress.set_stage(progress_id, "done")
        ans.meta = telemetry.summary((time.perf_counter() - t_start) * 1000)
        logger.info(
            "request_summary",
            extra={"kind": ans.kind, "flags": ans.flags, **ans.meta.model_dump(exclude={"trace_id"})},
        )
        return ans

    try:
        # 1. Intent gate
        progress.set_stage(progress_id, "intent")
        t0 = time.perf_counter()
        gate = await intent.gate(query, history)
        telemetry.record_stage("intent_ms", (time.perf_counter() - t0) * 1000)
        match gate.category:
            case "greeting":
                try:
                    user = resolve_token(token)
                except AuthenticationError:
                    return _finish(reply_logic.invalid_token())
                t0 = time.perf_counter()
                ans = await greeter.greeting(user.name.split(" ")[0], tone)
                telemetry.record_stage("greeting_ms", (time.perf_counter() - t0) * 1000)
                return _finish(ans)
            case "out_of_domain":
                return _finish(reply_logic.out_of_domain())
            case "manipulation":
                logger.warning("manipulation_blocked", extra={"reason": gate.reason})
                return _finish(reply_logic.refused())
            case "unclear":
                # One clarification round max.
                if _last_assistant_kind(history) == "clarify":
                    return _finish(reply_logic.clarify_exhausted())
                return _finish(reply_logic.clarify_request(gate.reason))
        rewritten = gate.rewritten_query or query

        # 2. Permission-scoped retrieval via MCP
        progress.set_stage(progress_id, "retrieval")
        t0 = time.perf_counter()
        try:
            search = await _call_search(token, rewritten)
        except PermissionError:
            return _finish(reply_logic.invalid_token())
        telemetry.record_stage("retrieval_ms", (time.perf_counter() - t0) * 1000)
        if search.status == "no_result" or not search.chunks:
            return _finish(reply_logic.no_result())

        # 3. Per-chunk relevance compression
        progress.set_stage(progress_id, "compress")
        t0 = time.perf_counter()
        compressed = await compressor.compress(rewritten, search.chunks)
        telemetry.record_stage("compress_ms", (time.perf_counter() - t0) * 1000)
        if not compressed:
            return _finish(reply_logic.insufficient_evidence())

        # 4. Grounded, cited generation
        progress.set_stage(progress_id, "generate")
        t0 = time.perf_counter()
        output, citations, grounded = await generator.generate(rewritten, compressed, tone)
        telemetry.record_stage("generate_ms", (time.perf_counter() - t0) * 1000)
        if not grounded:
            return _finish(reply_logic.insufficient_evidence())
        return _finish(
            AgentAnswer(kind="answered", text=output.answer, citations=citations, flags=output.flags)
        )
    except Exception:
        logger.exception("agent_request_failed")
        return _finish(reply_logic.internal_error(current_trace_id()))
