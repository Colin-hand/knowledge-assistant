"""LLM-generated greeting — personalizes small talk with the user's name."""

from pydantic import BaseModel, Field

from knowledge_assistant import telemetry
from knowledge_assistant.agent.prompts import GREETING_SYSTEM, tone_section
from knowledge_assistant.models import AgentAnswer


class GreetingOutput(BaseModel):
    message: str = Field(description="A short, warm greeting addressed to the user by name.")


async def greeting(name: str, tone: str) -> AgentAnswer:
    messages = [
        {"role": "system", "content": GREETING_SYSTEM + tone_section(tone)},
        {"role": "user", "content": f"The user's first name is {name}."},
    ]
    output = await telemetry.chat_parse("greeting", messages, GreetingOutput)
    return AgentAnswer(kind="greeting", text=output.message)
