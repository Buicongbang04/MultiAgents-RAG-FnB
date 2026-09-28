import re
from functools import lru_cache

from app.rag.neo4j_client import neo4j_client

# Thứ tự hiển thị size. Size lạ (nếu có) xếp sau.
SIZE_ORDER = {"S": 0, "M": 1, "L": 2}


def get_menu_variants(names: list[str]) -> dict[str, list[dict]]:
    """
    Mọi size/giá của các món theo tên (mỗi size là một MenuItem riêng).

    Retrieval chỉ trả vài node top-k, có thể thiếu size → hỏi size phải dựa trên
    danh sách đầy đủ này, không dựa trên nguồn RAG.
    """
    if not names:
        return {}

    rows = neo4j_client.execute_query(
        """
        UNWIND $names AS name
        MATCH (m:MenuItem)
        WHERE toLower(m.name_vi) = toLower(name) AND coalesce(m.available, true)
        RETURN name, m.id AS id, m.name_vi AS name_vi, m.size AS size,
               m.price AS price, m.category AS category, m.description AS description
        """,
        {"names": names},
    )

    variants: dict[str, list[dict]] = {name: [] for name in names}
    for row in rows:
        variants[row["name"]].append(
            {
                "id": row["id"],
                "name": row["name_vi"],
                "size": row["size"],
                "price": row["price"],
                "category": row["category"],
                "description": row["description"],
            }
        )

    for items in variants.values():
        items.sort(key=lambda v: (SIZE_ORDER.get(v["size"], 99), v["price"] or 0))
    return variants


@lru_cache(maxsize=1)
def get_menu_vocabulary() -> frozenset[str]:
    """
    Mọi từ xuất hiện trong menu (tên, mô tả, danh mục, tag, nguyên liệu).
    Cache theo process: sau khi ingest lại menu cần restart backend.
    """
    rows = neo4j_client.execute_query(
        """
        MATCH (m:MenuItem)
        OPTIONAL MATCH (m)-[:HAS_TAG|HAS_INGREDIENT]->(e:Entity)
        RETURN m.name_vi AS name_vi, m.name_en AS name_en, m.description AS description,
               m.category AS category, collect(e.name) AS entities
        """
    )
    words: set[str] = set()
    for row in rows:
        parts = [row["name_vi"], row["name_en"], row["description"], row["category"], *row["entities"]]
        for part in parts:
            words.update(re.findall(r"\w+", (part or "").lower()))
    return frozenset(words)


@lru_cache(maxsize=1)
def get_menu_names() -> tuple[str, ...]:
    """Tên các món đang bán (mỗi tên một lần). Cache theo process như vocabulary."""
    rows = neo4j_client.execute_query(
        """
        MATCH (m:MenuItem) WHERE coalesce(m.available, true)
        RETURN DISTINCT m.name_vi AS name
        """
    )
    return tuple(row["name"] for row in rows if row["name"])
