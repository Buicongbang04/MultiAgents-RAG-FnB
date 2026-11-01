from app.rag.neo4j_client import Neo4jClient

CONSTRAINTS = [
    "CREATE CONSTRAINT menu_id_unique IF NOT EXISTS FOR (m:MenuItem) REQUIRE m.id IS UNIQUE",
    "CREATE CONSTRAINT faq_id_unique IF NOT EXISTS FOR (f:FAQ) REQUIRE f.id IS UNIQUE",
    "CREATE CONSTRAINT chunk_id_unique IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_key_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.key IS UNIQUE",
    "CREATE CONSTRAINT category_name_unique IF NOT EXISTS FOR (c:Category) REQUIRE c.name IS UNIQUE",
]

# (index name, label) — 1024-dim cosine, matching BGE-M3 and app/rag/vector_index.py.
VECTOR_INDEXES = [
    ("menu_embedding", "MenuItem"),
    ("faq_embedding", "FAQ"),
    ("chunk_embedding", "Chunk"),
]


def ensure_constraints(graph: Neo4jClient) -> None:
    for query in CONSTRAINTS:
        graph.execute_query(query)


def ensure_vector_indexes(graph: Neo4jClient) -> None:
    for name, label in VECTOR_INDEXES:
        graph.execute_query(
            f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
            f"FOR (n:{label}) ON n.embedding "
            "OPTIONS {indexConfig: {`vector.dimensions`: 1024, "
            "`vector.similarity_function`: 'cosine'}}"
        )


def ensure_indexes(graph: Neo4jClient) -> None:
    """Constraints + vector indexes, all IF NOT EXISTS."""
    ensure_constraints(graph)
    ensure_vector_indexes(graph)
