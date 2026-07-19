from fastapi import APIRouter, Depends

from knowledge_assistant.agent import orchestrator
from knowledge_assistant.api.deps import get_token
from knowledge_assistant.api.schemas import ChatRequest, ChatResponse
from knowledge_assistant.config import get_settings

router = APIRouter()


@router.post("/chat", response_model=ChatResponse)
async def chat(req: ChatRequest, token: str = Depends(get_token)) -> ChatResponse:
    return await orchestrator.answer(
        token=token,
        query=req.query,
        history=[t.model_dump() for t in req.history],
        tone=req.tone,
    )


@router.get("/sources")
async def sources(token: str = Depends(get_token)) -> dict:
    # Proxy to the MCP list_sources tool so the ACL decision stays at the boundary.
    import json
    import sys

    from mcp import ClientSession, StdioServerParameters
    from mcp.client.stdio import stdio_client

    params = StdioServerParameters(
        command=sys.executable, args=["-m", "knowledge_assistant.mcp_server.server"]
    )
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool("list_sources", {"token": token})
    payload = result.structuredContent or json.loads(result.content[0].text)
    if "result" in payload and "status" not in payload:
        payload = payload["result"]
    return payload


@router.get("/healthz")
async def healthz() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "pinecone_configured": bool(s.pinecone_api_key),
        "openai_configured": bool(s.openai_api_key),
    }
