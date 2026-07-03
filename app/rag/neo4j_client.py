from neo4j import GraphDatabase, NotificationDisabledCategory

from app.core.config import get_settings
from app.core.logging import get_logger

logger = get_logger(__name__)


class Neo4jClient:
    """
    Thin Neo4j wrapper cho MVP.

    Chỉ quản lý:
    - connect
    - execute query
    - health check
    - close

    Retrieval logic sẽ ở phase sau.
    """

    def __init__(self) -> None:
        settings = get_settings()

        self.uri = settings.neo4j_uri
        self.user = settings.neo4j_user
        self.password = settings.neo4j_password
        self.database = settings.neo4j_database

        self.driver = GraphDatabase.driver(
            self.uri,
            auth=(self.user, self.password),
            notifications_disabled_categories=[
                NotificationDisabledCategory.UNRECOGNIZED,
            ],
        )

    def verify_connection(self) -> bool:
        try:
            self.driver.verify_connectivity()
            logger.info("Neo4j connected")
            return True
        except Exception:
            logger.exception("Neo4j connection failed")
            return False

    def close(self) -> None:
        self.driver.close()

    def ensure_vector_indexes(self, dim: int | None = None) -> dict[str, str]:
        """
        Tạo (idempotent) các vector index cần cho retrieval. Gọi lúc startup / sau
        khi embed. An toàn khi chạy lại: dùng IF NOT EXISTS.

        `dim` mặc định lấy từ settings.embedding_dim để khớp model đang dùng
        (bge-m3 = 1024). Trả về map index_name -> trạng thái.
        """
        if dim is None:
            dim = get_settings().embedding_dim
        dim = int(dim)  # inline vào Cypher nên ép int để tránh injection

        specs = {
            "menu_embedding": ("MenuItem", "m"),
            "faq_embedding": ("FAQ", "f"),
            "chunk_embedding": ("Chunk", "c"),
        }

        results: dict[str, str] = {}
        for name, (label, var) in specs.items():
            cypher = (
                f"CREATE VECTOR INDEX {name} IF NOT EXISTS "
                f"FOR ({var}:{label}) ON {var}.embedding "
                f"OPTIONS {{indexConfig: {{"
                f"`vector.dimensions`: {dim}, "
                f"`vector.similarity_function`: 'cosine'}}}}"
            )
            try:
                self.execute_query(cypher)
                results[name] = "ok"
            except Exception as exc:
                logger.warning("Failed to ensure vector index %s: %s", name, exc)
                results[name] = f"error: {exc}"

        return results

    def execute_query(
        self,
        query: str,
        parameters: dict | None = None,
    ) -> list[dict]:
        parameters = parameters or {}

        with self.driver.session(database=self.database) as session:
            result = session.run(query, parameters)

            records = []
            for row in result:
                records.append(dict(row))

            return records


neo4j_client = Neo4jClient()