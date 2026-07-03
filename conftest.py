"""Pytest bootstrap: force lightweight backends so importing `app.*` in tests
never loads GPU models or hits external services."""
import os

os.environ.setdefault("LLM_BACKEND", "mock")
os.environ.setdefault("EMBEDDING_BACKEND", "mock")
os.environ.setdefault("RERANKER_BACKEND", "null")
os.environ.setdefault("ROUTER_BACKEND", "rule_based")
os.environ.setdefault("INTENT_EXTRACTOR_BACKEND", "rule_based")
os.environ.setdefault("EMBEDDING_DEVICE", "cpu")
os.environ.setdefault("RERANKER_DEVICE", "cpu")
