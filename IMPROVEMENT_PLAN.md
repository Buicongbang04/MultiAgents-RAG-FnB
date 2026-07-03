# Phương án cải thiện — MultiAgents RAG FnB

> Tài liệu này tổng hợp kết quả rà soát toàn bộ codebase và đề xuất roadmap cải thiện theo thứ tự ưu tiên.
> Mỗi mục kèm `file:line` để dễ giao việc. Xóa file này bất cứ lúc nào nếu không cần.

## ✅ Đã hoàn thành (session 2026-07-03) — 11/11 unit test PASSED

Correctness & quick wins đã làm + verify bằng `pytest` (env conda `work`):
- **Reranker sigmoid** (`app/rag/reranker.py`) — logits → xác suất [0,1], threshold có nghĩa; giữ điểm cũ ở `metadata.pre_rerank_score`.
- **Word-boundary keyword** (`app/agents/intent_rules.py`) — "add" không còn khớp "address"; regex compile-cache.
- **Language default** (`app/agents/base.py`) — UNKNOWN → `vi` thay vì trả "unknown".
- **Semantic-cache alias floor** (`app/cache/semantic_cache.py`) — thêm `alias_similarity_floor=0.75`; `print` → `logger.debug`.
- **SGLang generate raise** (`app/llm/sglang.py`) — lỗi transport propagate để retry hoạt động; guard `choices` rỗng; giữ guardrail empty-context.
- **Vector index** — thêm `Neo4jClient.ensure_vector_indexes()` (idempotent, dùng `embedding_dim`), gọi lúc startup (`app/main.py`); 3 block `except: pass` → log-once; **fix FAQ dùng đúng `faq_embedding`** (bỏ bug lọc-label-sau-topk).
- **Đóng Neo4j driver** khi shutdown (`app/main.py`).
- **Bỏ giá trị bịa** trong few-shot (`app/prompts/{faq,order,consultant}.py`) → placeholder `[...]`.
- **Hạ tầng**: `.env.example`, `Dockerfile`, `docker-compose.yml`, `.dockerignore`, `pytest.ini`, `conftest.py`.
- **Test**: `tests/test_intent_rules.py`, `test_reranker.py`, `test_semantic_cache.py`.

⏳ **Còn lại của Phase 0** (chưa làm, cần cẩn trọng vì đụng async): Neo4j async / `to_thread` offloading, summarization ra ngoài lock, intent-extraction sau cache lookup, per-attempt timeout+jitter, cache-key over-collapse. CI (`.github/workflows`) cũng chưa thêm.

---

## Đánh giá nhanh

Kiến trúc thiết kế tốt (multi-agent + graph RAG + multi-layer cache + streaming, module rõ ràng, config qua `.env`).
Nhưng có **1 vấn đề hệ thống lớn** làm vô hiệu phần lớn thiết kế, cộng nhiều **bug correctness khiến bot trả lời sai**, và thiếu toàn bộ lớp **hạ tầng production**.

---

## 🔴 Vấn đề #1 — "Async giả": mọi tác vụ nặng block event loop

Server FastAPI async nhưng gần như mọi tác vụ nặng chạy blocking trên 1 thread loop. Kết hợp `generator_max_concurrency=1` (`app/core/config.py:39`), hệ thống thực chất xử lý **tuần tự**.

| Vị trí | Vấn đề |
|---|---|
| `app/rag/neo4j_client.py:30,50` | Driver Neo4j **đồng bộ** trong path async; `retrieve()` không có `await` nào |
| `app/agents/router_hf_lora.py:150` | `model.generate()` blocking trong `async def classify` |
| `app/cache/intent_extractor_hf.py:145` | `model.generate()` blocking trong `async def extract` |
| `app/rag/embedding_client.py:57` | `model.encode()` blocking |
| `app/rag/reranker.py:56` | `CrossEncoder.predict()` blocking |
| `app/rag/retriever.py:507` | Cosine thuần Python trên 500 rows × 1024 chiều trên loop |
| `app/session/session_store.py:79-138` | Gọi LLM tóm tắt **trong lúc giữ global lock** → serialize mọi session |

**Hướng sửa:** chuyển Neo4j sang `AsyncGraphDatabase`; bọc mọi `model.*` blocking bằng `asyncio.to_thread` + `Semaphore` (config `embedding/reranker_max_concurrency` đã có nhưng chưa được dùng); đưa summarization ra ngoài lock.

---

## 🔴 Nhóm bug correctness (bot trả lời SAI)

1. **Vector index chưa từng tạo trong code** → fast-path `db.index.vector.queryNodes` luôn ném lỗi, bị `except Exception: pass` nuốt (`app/rag/retriever.py:503,573,623`), rơi về quét 500 row. "10x faster" là code chết.
2. **Lọc label sau top-k** (`app/rag/retriever.py:552,603`): lấy top-20 toàn cục *rồi mới* `WHERE f:FAQ` → FAQ bị Chunk lấn át, mất recall.
3. **Reranker ghi đè điểm fusion + threshold sai** (`app/rag/reranker.py:69-72`): so `reranker_threshold=0.7` với **raw logits** không bounded → thiếu `sigmoid`; đồng thời ghi đè toàn bộ điểm fusion/boost đã tính.
4. **Semantic cache "domain-alias" trả lời sai dưới ngưỡng** (`app/cache/semantic_cache.py:229`): chỉ trùng 1 tag (vd "wifi") là hit bất kể cosine ~0.
5. **Cache-key gộp quá đà** (`app/cache/intent_extractor.py:198`): mọi câu tư vấn gộp về `"gợi ý món ngon"` → cà phê/trà/bánh đụng chung 1 câu cache.
6. **Substring match, không word-boundary** (`app/agents/intent_rules.py:52,89`): `"add"` trong `"address"` → "địa chỉ quán?" route sang ORDER.
7. **`language or Language.VI` là nhánh chết** (`app/agents/base.py:40`): `Language.UNKNOWN` truthy → trả `"unknown"` thay vì mặc định `vi`.
8. **Tóm tắt hội thoại no-op trên SGLang** (`app/session/session_store.py:174` + `app/llm/sglang.py:54`): guardrail short-circuit → summary chỉ là 200 ký tự đầu.
9. **Cơ chế retry là code chết** (`app/llm/sglang.py:98`): `generate()` bắt mọi exception rồi return fallback thay vì raise → queue không bao giờ retry.
10. **Metadata `rag_query` hard-index** (`app/agents/order_agent.py:15`, `consultant_agent.py:15`, `faq_agent.py:16`): `KeyError` nếu thiếu; nên dùng field typed `AgentInput.rag_result` (đang bỏ trống, `schemas.py:229`).

---

## 🟠 Cache tự phá chính nó

- **Intent extractor (có thể là LLM HF) chạy TRƯỚC cache lookup** (`app/services/chat_service.py:56` trước `:72`) → mỗi request kể cả cache-hit vẫn trả full `model.generate`. Lợi ích latency của cache bị triệt tiêu.
- **Embedding tính 2-3 lần/request** (get_semantic `chat_service`/`cache_service.py:83`, set `:147`, backfill `:248`) thay vì tính 1 lần rồi tái dùng.
- Semantic cache eviction FIFO (`app/cache/semantic_cache.py:307`) vs exact cache LRU — không nhất quán, hot entry bị evict.

---

## 🟠 Thiếu hạ tầng production

- ❌ Không có `Dockerfile` / `docker-compose.yml` (dù cần Neo4j + SGLang + backend + UI)
- ❌ Không CI (`.github/workflows`), không `.env.example`, không `pyproject.toml`
- ❌ `tests/` chỉ là script chạy tay, không có `test_*.py` thật, không `pytest.ini`/`conftest.py`
- ❌ Rate limiter in-memory per-process (`app/middleware/rate_limit.py:17`): không scale ngang, rò rỉ RAM theo IP (`:27`), không đọc `X-Forwarded-For` (`:23`)
- ❌ `httpx.AsyncClient` tạo mới mỗi request (`app/llm/sglang.py:73,126`, `chainlit_app.py:53`) — không pool/keep-alive
- ❌ Neo4j driver không đóng khi shutdown (`app/main.py:12-18` chỉ đóng session_store)

---

## 🟡 Duplication & maintainability

- Bảng keyword domain **lặp 3 nơi**: `app/cache/semantic_cache.py:71`, `app/cache/intent_extractor.py:141`, `app/cache/paraphrase.py:58`
- **3 hàm detect language**: `intent_rules.py:22`, `llm_router.py:146`, `router_hf_lora.py:60`
- **2 parser label**: `llm_router.py:36` (set — nondeterministic), `router_hf_lora.py:50` (dict)
- **2 bộ model LLM trùng**: `app/llm/base.py:7-26` vs `app/core/schemas.py:246-264`
- **2 serializer SSE** khác format: `app/streaming/sse.py:8` (json) vs `schemas.py:280` (orjson + `created_at`)
- Magic number rải khắp `retriever.py` (`0.65/0.35`, `0.95/0.85/0.70`, `LIMIT 500`) và `intent_rules.py` (`0.70/0.65/0.75`)

---

## 🟡 Security / robustness

- **Prompt injection**: nội dung retrieve nối thẳng với block `*_RULES:` (`app/agents/context_builders.py:45-58`), không tách "CONTEXT là data không phải lệnh"
- **Không giới hạn độ dài input** trước tokenize/LLM (`app/agents/router_hf_lora.py:155`, thiếu `truncation=True`) → vector DoS
- **Few-shot bịa giá trị thật** (`app/prompts/faq.py:11`: wifi password `highlands123` + giờ mở cửa; `app/prompts/order.py:12`, `consultant.py:14`: giá tiền) → model đọc thuộc, hallucination dù CONTEXT không có

---

# Roadmap theo phase

## Phase 0 — Correctness & Async (ưu tiên cao nhất)
- [ ] Neo4j → `AsyncGraphDatabase` (`neo4j_client.py`, `retriever.py`)
- [ ] Bọc `model.generate/encode/predict` bằng `asyncio.to_thread` + `Semaphore` (router HF, intent extractor HF, embedding, reranker)
- [ ] Đưa summarization LLM ra ngoài global session lock (`session_store.py`)
- [ ] Tạo vector index tự động (startup/script) + bỏ `except: pass`, thêm log warning
- [ ] Sửa lọc-label-sau-topk (over-fetch hoặc index riêng theo label)
- [ ] Reranker: thêm `sigmoid`, blend điểm thay vì ghi đè (`reranker.py`)
- [ ] Semantic-cache domain-alias: thêm sàn similarity (`semantic_cache.py`)
- [ ] Sửa cache-key over-collapse theo subject (`intent_extractor.py`)
- [ ] Word-boundary cho keyword match (`intent_rules.py`)
- [ ] Sửa `language or Language.VI` → default `vi` khi UNKNOWN (`base.py`)
- [ ] `SGLangClient.generate` raise lỗi transport để retry hoạt động; per-attempt timeout + jitter (`sglang.py`, `request_queue.py`)
- [ ] Chuyển intent-extraction sau/song song cache lookup; tính embedding 1 lần/request (`chat_service.py`, `cache_service.py`)

## Phase 1 — Hạ tầng production
- [ ] `Dockerfile` + `docker-compose.yml` (Neo4j + backend + UI) + `.dockerignore`
- [ ] `.env.example` đầy đủ biến
- [ ] Shared `httpx.AsyncClient` (startup/shutdown); đóng Neo4j driver trong lifespan (`main.py`)
- [ ] Rate limiter: `X-Forwarded-For`, eviction bucket idle, interface cho Redis (`rate_limit.py`)
- [ ] `pyproject.toml`, `pytest.ini`, `.github/workflows/ci.yml` (lint + test)

## Phase 2 — Test & chống hồi quy
- [ ] Unit test logic thuần: `classify_by_rules`, parser label, detect language, context builders, cache key, cosine
- [ ] Integration test với `LLM_BACKEND=mock` + Neo4j test container

## Phase 3 — Dọn dẹp & bảo mật
- [ ] Gộp 3 bảng keyword → 1 nguồn duy nhất
- [ ] Gộp language detector, label parser, model LLM, SSE serializer
- [ ] Prompt hardening (tách CONTEXT khỏi instruction) + giới hạn độ dài input
- [ ] Bỏ giá trị bịa trong few-shot prompt
- [ ] Đưa magic number vào config

## Phase 4 — Nâng cấp (tùy chọn)
- [ ] i18n thật cho prompt/fallback (hiện chỉ tiếng Việt dù pipeline detect EN)
- [ ] ANN index cho semantic cache thay linear scan
- [ ] Batch ingest (`UNWIND` + `embed_batch`) thay N+1 write (`vector_index.py`)

---

## Quick wins (ROI cao, rủi ro thấp — nên làm trước)
- Reranker sigmoid (`reranker.py`)
- `language or Language.VI` (`base.py`)
- Word-boundary keyword (`intent_rules.py`)
- Bỏ giá trị bịa trong few-shot prompts (`app/prompts/*.py`)
- `.env.example` + `Dockerfile` + `docker-compose.yml`
