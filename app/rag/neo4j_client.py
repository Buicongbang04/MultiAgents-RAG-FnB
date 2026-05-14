from neo4j import GraphDatabase

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