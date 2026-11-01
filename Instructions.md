# MultiAgents RAG FnB

Chatbot tư vấn F&B (Highlands Coffee) sử dụng kiến trúc **Multi-Agent + Graph RAG** chạy Local theo mặc định (SGLang/mock); backend LLM Cloud (Claude) là tuỳ chọn.

Hệ thống xử lý 4 luồng: **Đặt hàng**, **Tư vấn menu**, **FAQ**, **Ignore/Noise**.

---

## Kiến trúc tổng thể

```
User Message
      │
      ▼
 LLM Router (Qwen2.5-7B)        ← hiểu ngôn ngữ tự nhiên, không hardcode
      │  fallback: rule_based
      ├──[order]──────► Order Agent
      ├──[consultant]─► Consultant Agent
      ├──[faq]────────► FAQ Agent
      └──[ignore]─────► Ignore Handler
                              │
                     Graph RAG (Neo4j)
               Hybrid Search + Graph Expansion
                              │
                  Reranker (BGE-reranker-v2-m3)
                              │
                  Generator (Qwen2.5-7B-Instruct-AWQ)
                              │
                    Multi-layer Cache
               Exact Cache + Semantic Cache (≥ 0.92)
                              │
                     Chainlit UI (animate)
```

### Các thành phần chính

| Thành phần | Chi tiết |
|---|---|
| **Router** | LLM-based (Qwen2.5-7B via SGLang), fallback rule_based |
| **Generator** | Qwen2.5-7B-Instruct-AWQ qua SGLang |
| **Graph DB** | Neo4j 5 — MenuItem, FAQ, Chunk, Entity, Category |
| **Retrieval** | Hybrid (keyword + vector) + Graph Expansion + BGE Reranker |
| **Cache** | Exact cache + Semantic cache (cosine ≥ 0.92) |
| **Session** | In-memory, TTL 30 phút, auto-summarization |
| **UI** | Chainlit — word-by-word animation, intent badge, latency |

---

## Dữ liệu (Highlands Coffee)

| Loại | Số lượng | Nội dung |
|---|---|---|
| Menu | 55 món | 8 cà phê × 3 size, 5 trà × 3 size, 4 freeze × 3 size, 4 bánh |
| FAQ | 30 cặp Q&A | 21 chủ đề (wifi, giờ, thanh toán, giao hàng...), Vi + En |
| Policy | 20 chunks | Hoàn trả, tùy chỉnh, combo, chính sách quán |

---

## Yêu cầu phần cứng

| | Tối thiểu | Khuyến nghị |
|---|---|---|
| GPU | 8GB VRAM | RTX 3060 12GB |
| RAM | 16GB | 32GB |
| OS | Ubuntu 20.04+ | Ubuntu 22.04 |
| Python | 3.11 | 3.11 |

---

## Cài đặt

### 1. Clone repo

```bash
git clone https://github.com/Buicongbang04/MultiAgents-RAG-FnB.git
cd MultiAgents-RAG-FnB
```

### 2. Cài uv

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

### 3. Cài dependencies

```bash
uv sync --extra ml --extra ui          # Core API + HuggingFace/training + Chainlit
```

SGLang **không** nằm trong extras của dự án (phụ thuộc vllm/torch xung đột với lockfile). Chạy SGLang từ venv riêng:

```bash
uv venv .venv-sglang --python 3.11
uv pip install --python .venv-sglang/bin/python "sglang[all]"
# Khi khởi động thêm cờ đã chạy được trên máy phát triển:
bash scripts/SGLang_Server.sh -- --attention-backend triton --sampling-backend pytorch
```

Hoặc dùng Docker: `docker compose --profile local-llm up -d sglang` (cần GPU).

### 4. Cấu hình

Mọi tham số nằm trong thư mục [configs/](configs/):

| File | Nội dung |
|---|---|
| `configs/app.yaml` | Tham số runtime: model, backend, Neo4j, ngưỡng, trọng số RAG, cache, timeout, concurrency, SGLang server |
| `configs/lexicon.yaml` | Từ khoá router, stopword, alias, regex của intent extractor / semantic cache |
| `configs/training.yaml` | Siêu tham số QLoRA + `TrainingArguments` cho router và intent extractor |

File `.env` chỉ chứa **secret** và giá trị **override theo máy**. Key lồng nhau viết bằng `__`
(ưu tiên: biến môi trường > `.env` > `app.yaml`):

```env
# Bắt buộc
NEO4J__PASSWORD=your_password

# Override tuỳ máy (ví dụ)
LLM__BACKEND=sglang            # mock nếu không có GPU
EMBEDDING__DEVICE=cuda         # cpu nếu không có GPU
RERANKER__DEVICE=cuda
```

### Backend Claude

`LLM__BACKEND=anthropic` gọi thẳng `api.anthropic.com` với model `claude-opus-5-5` (mặc định trong `app.yaml`). Khóa **chỉ** đọc từ biến môi trường `ANTHROPIC_API_KEY` (đặt trong `.env`, không đặt trong YAML; `ANTHROPIC_AUTH_TOKEN` và profile `ant auth login` bị bỏ qua có chủ đích).

Nếu dùng gateway tương thích Anthropic riêng, override theo máy trong `.env` (không commit):

```env
LLM__ANTHROPIC__BASE_URL=https://your-gateway.example.com   # gốc domain, không có /v1
LLM__ANTHROPIC__MODEL=your-gateway-model-id
```

Lưu ý: khi đặt `BASE_URL`, `ANTHROPIC_API_KEY` sẽ được gửi tới host đó.

Đổi cả bộ cấu hình (vd. cho staging): `APP_CONFIG_DIR=/path/to/configs python run.py`.

### 5. Khởi động hạ tầng (Docker Compose)

```bash
cp .env.example .env                   # rồi sửa NEO4J__PASSWORD, ...
docker compose up -d postgres neo4j    # Postgres :5433, Neo4j :7474/:7687
docker compose ps                      # chờ cả hai ở trạng thái healthy
```

Các profile tuỳ chọn:

```bash
docker compose --profile test up -d postgres-test neo4j-test  # DB riêng cho test tích hợp
docker compose --profile local-llm up -d sglang               # SGLang trong Docker (cần GPU)
docker compose up -d --build backend                          # backend trong Docker (port 8001)
docker compose -f docker-compose.yml -f docker-compose.gpu.yml up -d --build backend  # backend + GPU
```

Nếu còn container Neo4j chạy tay (`neo4j-fnb`) đang giữ port 7474/7687, hãy `docker stop neo4j-fnb` trước.

### 6. Nạp dữ liệu vào Graph

```bash
uv run python -m scripts.bootstrap            # migrate Postgres → seed → tạo index → chiếu sang Neo4j → embed
uv run python -m scripts.bootstrap --reset    # xóa catalog Postgres rồi nạp lại
uv run python -m scripts.bootstrap --skip-embed   # bỏ bước embedding
```

Lệnh trên đã tự tạo constraint và 3 vector index (`menu_embedding`, `faq_embedding`, `chunk_embedding`), không cần chạy Cypher thủ công.

---

## Chạy hệ thống

### Terminal 1 — SGLang Server

```bash
bash scripts/SGLang_Server.sh                          # tham số lấy từ llm.model + llm.server
uv run python -m scripts.launch_sglang --dry-run              # chỉ in câu lệnh sẽ chạy
bash scripts/SGLang_Server.sh -- --disable-flashinfer  # thêm cờ SGLang tuỳ ý
```

Kiểm tra:

```bash
curl http://localhost:30000/v1/models
```

### Terminal 2 — Backend + UI

```bash

uv run python run.py                          # Backend only (port 8001)
uv run python run.py --with-ui                # Backend + Chainlit UI (port 8501)
uv run python run.py --with-ui --ui-port 8080 # Đổi UI port
uv run python run.py --reload                 # Hot-reload (dev)
```

Sau khi start:
- **API docs:** http://localhost:8001/docs
- **Chat UI:** http://localhost:8501

Kiểm tra hệ thống:

```bash
curl http://localhost:8001/health
```

---

## Chạy test

```bash
uv run pytest -q                                              # unit test, không cần hạ tầng
docker compose --profile test up -d postgres-test neo4j-test  # hạ tầng cho integration test
uv run pytest -m integration -q
```

---

## API

### POST `/chat`

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Cho mình 1 bạc xỉu size M", "session_id": "sess-001"}'
```

Response:

```json
{
  "session_id": "sess-001",
  "intent": "order",
  "agent": "order_agent",
  "answer": "Dạ, em tìm được Bạc xỉu size M giá 45.000đ. Anh/chị xác nhận giúp em nhé?",
  "latency_ms": 280.5,
  "sources": [...],
  "metadata": {
    "router": {"router_type": "llm", ...},
    "cache": {"hit": false, ...}
  }
}
```

### POST `/chat/stream`

SSE stream — trả về từng token:

```bash
curl -X POST http://localhost:8001/chat/stream \
  -H "Content-Type: application/json" \
  -d '{"text": "Có gì ngon rẻ không?"}' \
  --no-buffer
```

### POST `/cache/invalidate`

```bash
# Xóa cache FAQ khi dữ liệu thay đổi
curl -X POST "http://localhost:8001/cache/invalidate?intent=faq"

# Xóa toàn bộ cache
curl -X POST http://localhost:8001/cache/invalidate
```

---

## Câu hỏi demo

| Intent | Ví dụ |
|--------|-------|
| **Đặt hàng** | "Cho anh 1 bạc xỉu đá size L" · "1 cà phê đen đá" · "cà phê sữa size M" |
| **Tư vấn** | "Có gì ngon rẻ không?" · "Trời nóng uống gì?" · "Gợi ý món ít ngọt" |
| **FAQ** | "Wifi quán là gì?" · "Mấy giờ đóng cửa?" · "Có ship không?" |
| **Ignore** | "Hello" · "Hôm nay thời tiết thế nào?" |

---

## Benchmark

```bash
python scripts/benchmark_chat_api.py              # Latency & throughput
python scripts/benchmark_intelligent_cache_500.py # Semantic cache hit rate
python scripts/benchmark_router_latency.py        # Router latency
```

---

## Cấu trúc thư mục

```
app/
├── agents/        # LLMRouter, RuleRouter, Order, Consultant, FAQ, Ignore
├── api/           # FastAPI routes (/chat, /cache/invalidate, /health)
├── cache/         # Exact cache + Semantic cache
├── core/          # Config, schemas, constants
├── llm/           # SGLang / Claude / mock client
├── middleware/    # Rate limiting
├── prompts/       # System prompts cho từng agent
├── queueing/      # Concurrency control (Semaphore + retry)
├── rag/           # Neo4j retriever, embedding, reranker
├── services/      # ChatService (orchestration)
├── session/       # Session store + auto-summarization
└── streaming/     # SSE helpers

data/mock/
├── menu.csv       # 55 món Highlands Coffee (S/M/L, giá thật)
├── faq.csv        # 30 Q&A (21 chủ đề, Vi + En)
└── docs.jsonl     # 20 policy chunks

scripts/           # Ingest, embed, benchmark, training
tests/             # Unit (tests/unit) & integration (tests/integration) test
chainlit_app.py    # Chainlit UI (word-by-word animation)
run.py             # Entry point duy nhất
```

---

## Tác giả

**Bùi Công Bằng** — [github.com/Buicongbang04](https://github.com/Buicongbang04)
