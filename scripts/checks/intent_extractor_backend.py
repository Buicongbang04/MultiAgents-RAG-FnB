from __future__ import annotations

import asyncio
import json

from app.cache.intent_extractor import IntentExtractionInput, get_intent_extractor
from app.core.constants import Intent, Language


CASES = [
    ("phase27-test", "Wifi quán là gì?", Intent.FAQ),
    ("phase27-test", "Cho em xin mật khẩu wifi với nha", Intent.FAQ),
    ("phase27-test", "Em gợi ý món nào giá mềm mà dễ uống?", Intent.CONSULTANT),
    ("phase27-test", "Cho anh 2 bạc xỉu đá", Intent.ORDER),
    ("phase27-test", "hello em", Intent.IGNORE),
]


async def main() -> None:
    extractor = get_intent_extractor()

    for session_id, text, intent in CASES:
        output = await extractor.extract(
            IntentExtractionInput(
                session_id=session_id,
                text=text,
                intent=intent,
                language=Language.VI,
                history=[],
            )
        )
        print("=" * 80)
        print(text)
        print(json.dumps(output.model_dump(), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    asyncio.run(main())