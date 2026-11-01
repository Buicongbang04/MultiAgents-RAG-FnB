import asyncio

from app import main


class _BoomEmbedding:
    async def embed_text(self, text):
        raise RuntimeError("boom")


def test_warmup_failure_does_not_block_lifespan(monkeypatch):
    monkeypatch.setattr(main, "get_embedding_client", lambda: _BoomEmbedding())

    class _Llm:
        async def aclose(self):
            pass

    class _Store:
        async def start_background_cleanup(self):
            pass

        async def close(self):
            pass

    monkeypatch.setattr(main, "get_llm_client", lambda: _Llm())
    monkeypatch.setattr(main, "session_store", _Store())

    async def run():
        entered = False
        async with main.lifespan(main.app):
            entered = True
        return entered

    assert asyncio.run(run()) is True
