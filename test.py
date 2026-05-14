import asyncio

from app.agents.consultant_agent import consultant_agent
from app.agents.faq_agent import faq_agent
from app.agents.ignore_handler import ignore_handler
from app.agents.order_agent import order_agent
from app.core.constants import Intent
from app.core.schemas import AgentInput, RAGQuery


async def run_case(text: str, intent: Intent):
    agent_map = {
        Intent.ORDER: order_agent,
        Intent.CONSULTANT: consultant_agent,
        Intent.FAQ: faq_agent,
        Intent.IGNORE: ignore_handler,
    }

    agent = agent_map[intent]

    output = await agent.run(
        AgentInput(
            session_id="sess_test",
            text=text,
            intent=intent,
            metadata={
                "rag_query": RAGQuery(
                    query=text,
                    intent=intent,
                )
            },
        )
    )

    print("\n" + "=" * 80)
    print("INPUT:", text)
    print("INTENT:", intent.value)
    print("AGENT:", output.agent.value)
    print("ANSWER:\n", output.answer)
    print("SOURCES:", len(output.sources))


async def main():
    await run_case("Cho anh một ly bạc xỉu đá", Intent.ORDER)
    await run_case("Có gì ngon ít ngọt không em?", Intent.CONSULTANT)
    await run_case("Wifi tên gì vậy?", Intent.FAQ)
    await run_case("haha", Intent.IGNORE)


asyncio.run(main())