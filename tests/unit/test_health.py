from fastapi.testclient import TestClient

from app.api import routes
from app.main import app


def test_health_degraded_when_postgres_down(monkeypatch):
    async def down():
        return False

    monkeypatch.setattr(routes, "ping_database", down)
    monkeypatch.setattr(routes.neo4j_client, "verify_connection", lambda: True)
    client = TestClient(app)  # no lifespan: avoid model warmup
    resp = client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["postgres"] is False
    assert body["status"] == "degraded"
