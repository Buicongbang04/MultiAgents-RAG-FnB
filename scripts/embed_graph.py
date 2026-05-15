import asyncio

from app.rag.vector_index import embed_all_graph_nodes


async def main():
    result = await embed_all_graph_nodes()

    print("=" * 80)
    print("GRAPH EMBEDDING DONE")
    print("=" * 80)
    print(result)


if __name__ == "__main__":
    asyncio.run(main())