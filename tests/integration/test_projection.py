import os
from pathlib import Path

import pytest
import pytest_asyncio

from app.catalog.seed import seed_catalog
from app.core.config import get_settings
from app.db.models.catalog import Product
from app.knowledge.graph_writer import project_catalog
from app.knowledge.indexes import ensure_indexes
from app.rag import menu as menu_module
from app.rag.menu import get_menu_names, invalidate_menu_caches
from app.rag.neo4j_client import Neo4jClient

pytestmark = pytest.mark.integration

DATA = Path(__file__).resolve().parents[2] / "data" / "mock"
TEST_NEO4J_URI = os.getenv("TEST_NEO4J_URI", "bolt://localhost:7688")


@pytest.fixture(scope="module")
def graph():
    """Client bound to the TEST Neo4j instance only (never the dev graph)."""
    client = Neo4jClient(uri=TEST_NEO4J_URI)
    assert client.verify_connection(), f"test Neo4j not reachable at {TEST_NEO4J_URI}"
    yield client
    client.close()


@pytest_asyncio.fixture
async def seeded(db_session):
    await seed_catalog(db_session, DATA, reset=True)
    return db_session


def _count(graph, label: str) -> int:
    return graph.execute_query(f"MATCH (n:{label}) RETURN count(n) AS c")[0]["c"]


def _counts(graph) -> dict[str, int]:
    return {k: _count(graph, k) for k in ("MenuItem", "FAQ", "Chunk", "Category", "Entity")}


async def test_projection_counts(seeded, graph):
    await project_catalog(seeded, graph)
    c = _counts(graph)
    assert (c["MenuItem"], c["FAQ"], c["Chunk"], c["Category"]) == (55, 30, 20, 4)
    row = graph.execute_query(
        "MATCH (m:MenuItem {id:'menu_0001'}) RETURN m.product_id AS p, m.available AS a"
    )[0]
    assert row["p"] and row["a"] is True


async def test_projection_idempotent(seeded, graph):
    await project_catalog(seeded, graph)
    first = _counts(graph)
    rels = graph.execute_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"]
    await project_catalog(seeded, graph)
    assert _counts(graph) == first
    assert graph.execute_query("MATCH ()-[r]->() RETURN count(r) AS c")[0]["c"] == rels


async def test_sold_out_not_available(seeded, graph, monkeypatch):
    seeded_product = await seeded.get(Product, "bac-xiu")
    seeded_product.status = "sold_out"
    await seeded.flush()
    await project_catalog(seeded, graph)
    rows = graph.execute_query(
        "MATCH (m:MenuItem {product_id:'bac-xiu'}) RETURN m.available AS a"
    )
    assert len(rows) == 3 and all(r["a"] is False for r in rows)

    monkeypatch.setattr(menu_module, "neo4j_client", graph)
    invalidate_menu_caches()
    try:
        assert "Bạc xỉu" not in get_menu_names()
    finally:
        invalidate_menu_caches()


def test_vector_indexes_online(graph):
    ensure_indexes(graph)
    ensure_indexes(graph)  # idempotent
    names = {r["name"] for r in graph.execute_query("SHOW VECTOR INDEXES YIELD name RETURN name")}
    assert {"menu_embedding", "faq_embedding", "chunk_embedding"} <= names
