import json

from app.core.constants import StreamEventType


def sse_event(event_type: StreamEventType, data: dict) -> str:
    payload = {"type": event_type.value, "data": data}
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"
