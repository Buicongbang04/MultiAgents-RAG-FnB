import asyncio
import json
from typing import AsyncGenerator

from app.core.config import get_settings
from app.core.constants import StreamEventType
from app.streaming.clause_splitter import ClauseSplitter


def sse_event(event_type: StreamEventType, data: dict) -> str:
    payload = {
        "type": event_type.value,
        "data": data,
    }
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


async def stream_text_as_sse(
    text: str,
) -> AsyncGenerator[str, None]:
    """
    MVP token streaming.

    Chưa gọi LLM thật nên giả lập token bằng từng từ.
    Sau thay token bằng SGLang/vLLM streaming response.
    """

    settings = get_settings()
    splitter = ClauseSplitter()

    words = text.split(" ")

    for idx, word in enumerate(words):
        token = word
        if idx < len(words) - 1:
            token += " "

        yield sse_event(
            StreamEventType.TOKEN,
            {
                "token": token,
            },
        )

        clauses = splitter.push(token)

        for clause in clauses:
            yield sse_event(
                StreamEventType.CLAUSE,
                {
                    "clause": clause,
                },
            )

        await asyncio.sleep(settings.stream_token_delay_seconds)

    for clause in splitter.flush():
        yield sse_event(
            StreamEventType.CLAUSE,
            {
                "clause": clause,
            },
        )

    yield "data: [DONE]\n\n"