import asyncio
import threading

import numpy as np

from app.rag.embedding_client import SentenceTransformersEmbeddingClient
from app.rag.reranker import BGEReranker
from app.core.schemas import RetrievedSource


class _FakeModel:
    def __init__(self):
        self.threads = []

    def encode(self, texts, **kwargs):
        self.threads.append(threading.get_ident())
        return np.ones((len(texts), 4), dtype=np.float32)

    def predict(self, pairs):
        self.threads.append(threading.get_ident())
        return [0.5 for _ in pairs]


def _client(model):
    client = object.__new__(SentenceTransformersEmbeddingClient)
    client.model = model
    client.batch_size = 2
    return client


def test_encode_runs_in_thread():
    model = _FakeModel()
    client = _client(model)
    loop_thread = threading.get_ident()

    async def run():
        await client.embed_text("a")
        await client.embed_batch(["a", "b"])

    asyncio.run(run())
    assert len(model.threads) == 2
    assert all(t != loop_thread for t in model.threads)


def test_arerank_runs_predict_in_thread():
    model = _FakeModel()
    reranker = BGEReranker(model_name="x", device="cpu", dtype="float32")
    reranker._model = model
    sources = [RetrievedSource(source_id="1", text="t", score=0.1, source_type="faq")]
    loop_thread = threading.get_ident()

    out = asyncio.run(reranker.arerank("q", sources, top_k=1, threshold=0.0))
    assert len(out) == 1
    assert model.threads and model.threads[0] != loop_thread


def test_concurrent_arerank_loads_model_once(monkeypatch):
    import time

    loads = []
    reranker = BGEReranker(model_name="x", device="cpu", dtype="float32")

    def slow_load():
        loads.append(1)
        time.sleep(0.2)
        reranker._model = _FakeModel()

    monkeypatch.setattr(reranker, "_load_unlocked", slow_load)
    sources = [RetrievedSource(source_id="1", text="t", score=0.1, source_type="faq")]

    async def run():
        await asyncio.gather(*[reranker.arerank("q", list(sources), top_k=1) for _ in range(8)])

    asyncio.run(run())
    assert len(loads) == 1
