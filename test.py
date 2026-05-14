import asyncio
from app.agents.router_agent import router_agent
from app.core.schemas import RouterInput


async def main():
    samples = [
        "Cho anh một ly bạc xỉu đá",
        "Tính tiền giúp anh",
        "Có gì ngon rẻ không em?",
        "Bạc xỉu ngon không?",
        "Wifi tên gì vậy?",
        "Mấy giờ đóng cửa?",
        "haha",
        "Trời mưa quá",
        "Cho em một ly bạc xỉu, wifi là gì vậy?",
    ]

    for i, text in enumerate(samples):
        out = await router_agent.classify(
            RouterInput(
                session_id=f"sess_test_{i}",
                text=text,
            )
        )
        print(text, "=>", out.to_required_json(), out.confidence, out.metadata)


asyncio.run(main())