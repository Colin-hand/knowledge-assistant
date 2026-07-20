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


@router.get("/healthz")
async def healthz() -> dict:
    s = get_settings()
    return {
        "status": "ok",
        "pinecone_configured": bool(s.pinecone_api_key),
        "openai_configured": bool(s.openai_api_key),
    }
