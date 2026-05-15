# import asyncio

# from app.core.schemas import ChatRequest
# from app.services.chat_service import chat_service


# async def run_case(text: str, session_id=None):
#     result = await chat_service.chat(
#         ChatRequest(
#             text=text,
#             session_id=session_id,
#         )
#     )

#     print("\n" + "=" * 80)
#     print("SESSION:", result.session_id)
#     print("INTENT:", result.intent.value)
#     print("AGENT:", result.agent.value)
#     print("LATENCY:", round(result.latency_ms, 2), "ms")
#     print("ANSWER:\n", result.answer)
#     print("SOURCES:", len(result.sources))

#     return result.session_id


# async def main():

#     session_id = None

#     session_id = await run_case(
#         "Cho anh một ly bạc xỉu đá",
#         session_id,
#     )

#     session_id = await run_case(
#         "Wifi tên gì vậy?",
#         session_id,
#     )

#     session_id = await run_case(
#         "Có gì ngon ít ngọt không em?",
#         session_id,
#     )

#     session_id = await run_case(
#         "haha",
#         session_id,
#     )


# asyncio.run(main())


import asyncio

from app.core.schemas import ChatRequest
from app.services.chat_service import chat_service


async def main():
    response = await chat_service.chat(
        ChatRequest(
            text="Cho anh một ly bạc xỉu đá",
            session_id="test-session",
        )
    )
    print(response)


asyncio.run(main())
