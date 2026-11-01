"""One command to (re)build everything: migrate -> seed -> indexes -> project -> embed.

    uv run python -m scripts.bootstrap [--reset] [--skip-embed]
"""

import argparse
import asyncio
import subprocess
import sys
from pathlib import Path

from app.catalog.seed import seed_catalog
from app.db.session import get_engine, get_sessionmaker
from app.knowledge.graph_writer import project_catalog
from app.knowledge.indexes import ensure_indexes
from app.rag.neo4j_client import neo4j_client

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "mock"


async def run(reset: bool, skip_embed: bool) -> dict:
    subprocess.run([sys.executable, "-m", "alembic", "upgrade", "head"], cwd=ROOT, check=True)

    if not neo4j_client.verify_connection():
        raise SystemExit("Neo4j connection failed")

    async with get_sessionmaker()() as db:
        seeded = await seed_catalog(db, DATA_DIR, reset=reset)
        await db.commit()
        ensure_indexes(neo4j_client)
        projected = await project_catalog(db, neo4j_client)

    result = {
        "products": seeded.products,
        "variants": seeded.variants,
        "faqs": seeded.faqs,
        "chunks": seeded.chunks,
        "graph": projected,
    }
    if not skip_embed:
        from scripts.embed_graph import embed_graph

        result["embedded"] = await embed_graph()

    await get_engine().dispose()
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reset", action="store_true", help="truncate catalog tables before seeding")
    parser.add_argument("--skip-embed", action="store_true", help="skip BGE-M3 embedding step")
    args = parser.parse_args()

    result = asyncio.run(run(args.reset, args.skip_embed))
    print("=" * 60)
    print(
        f"products {result['products']} / variants {result['variants']} / "
        f"faqs {result['faqs']} / chunks {result['chunks']}"
    )
    print(f"graph: {result['graph']}")
    if "embedded" in result:
        print(f"embedded: {result['embedded']}")
    print("Bootstrap completed")


if __name__ == "__main__":
    main()
