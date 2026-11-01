from __future__ import annotations

import asyncio
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from app.agents.router_agent import router_agent
from app.core.schemas import RouterInput


TEST_CASES = [
    "Cho anh một ly bạc xỉu đá",
    "Có gì ngon rẻ không?",
    "Wifi quán là gì?",
    "hello em",
    "Can I order one latte?",
    "What time do you open?",
    "Any good blended drink?",
    "hmm not sure",
]


async def main() -> None:
    for idx, text in enumerate(TEST_CASES, start=1):
        output = await router_agent.classify(
            RouterInput(
                session_id=f"router-test-{idx}",
                text=text,
            )
        )

        print("=" * 80)
        print(f"INPUT: {text}")
        print(f"INTENT: {output.action.value}")
        print(f"LANGUAGE: {output.language.value}")
        print(f"JSON: {output.to_required_json()}")
        print(f"METADATA: {output.metadata}")


if __name__ == "__main__":
    asyncio.run(main())