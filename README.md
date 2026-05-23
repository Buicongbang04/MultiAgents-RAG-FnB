# MultiAgents RAG FnB

Robot tư vấn F&B (Highlands Coffee) sử dụng kiến trúc **Multi-Agent + Graph RAG** chạy hoàn toàn Local — không phụ thuộc Cloud API.

Hệ thống xử lý 4 luồng: **Đặt hàng**, **Tư vấn**, **FAQ**, **Ignore/Noise**.

---

## Kiến trúc tổng thể

```
User Query
    │
    ▼
Router Agent (Qwen2.5-1.5B)
    │
    ├──[order]──────► Order Agent ──────┐
    ├──[consultant]─► Consultant Agent ─┤
    ├──[faq]────────► FAQ Agent ────────┤
    └──[ignore]─────► Ignore Handler ───┘
                                        │
                              Graph RAG (Neo4j)
                         Hybrid Search + Graph Expansion
                                        │
                              Generator (Qwen2.5-7B)
                                        │
                              Response + TTS (SSE)
```

### Các thành phần chính

| Thành phần | Chi tiết |
|---|---|
| **Router** | Qwen2.5-1.5B fine-tuned, latency ≤ 200ms |
| **Generator** | Qwen2.5-7B-Instruct-AWQ qua SGLang |
| **Graph DB** | Neo4j 5 — Menu, FAQ, Chunk, Entity |
| **Retrieval** | Hybrid (keyword + vector) + Graph Expansion + BGE Reranker |
| **Cache** | Exact cache + Semantic cache (embedding similarity ≥ 0.95) |
| **Session** | In-memory, TTL 30 phút, auto-summarization |
| **Streaming** | True token streaming qua SSE |

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

### 2. Tạo conda environment

```bash
conda create -n fiai python=3.11 -y
conda activate fiai
```

### 3. Cài dependencies

```bash
# Core API
pip install -r requirements.txt

# ML stack (HuggingFace, training)
pip install -r requirements-ml.txt

# Chainlit demo UI
pip install -r requirements-ui.txt

# SGLang serving (chỉ cài trên inference host có CUDA)
pip install -r requirements-serving.txt
```

### 4. Tạo file `.env`

Các biến quan trọng:

```env
# LLM Backend
LLM_BACKEND=sglang
LLM_BASE_URL=http://localhost:30000/v1
GENERATOR_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# Router
ROUTER_BACKEND=hf_merged
ROUTER_MERGED_MODEL_DIR=models/router-qwen2.5-0.5b-merged

# RAG
RAG_RETRIEVAL_MODE=hybrid_graph
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3

# Reranker (optional, cần GPU)
RERANKER_BACKEND=bge
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
```

### 5. Khởi động Neo4j

```bash
docker run \
  --name neo4j-fnb \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/your_password \
  -d neo4j:5
```

### 6. Nạp dữ liệu vào Graph

```bash
conda activate fiai
python -m scripts.ingest_mock_to_neo4j
python -m scripts.embed_graph
```

Tạo Neo4j vector index (chạy 1 lần trong Neo4j Browser):

```cypher
CREATE VECTOR INDEX menu_embedding IF NOT EXISTS
FOR (m:MenuItem) ON m.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};
```

---

## Chạy hệ thống

### Terminal 1 — SGLang Server

```bash
bash scripts/SGLang_Server.sh
```

Kiểm tra:

```bash
curl http://localhost:30000/v1/models
```

### Terminal 2 — Backend (+ tuỳ chọn UI)

Chạy backend đơn giản:

```bash
conda activate fiai
python run.py
```

Chạy backend **kèm Chainlit UI** cùng lúc:

```bash
python run.py --with-ui
# Backend: http://localhost:8001
# UI:      http://localhost:8501
```

Các tuỳ chọn:

```bash
python run.py --port 8002          # đổi port
python run.py --reload             # hot-reload (dev)
python run.py --with-ui --ui-port 8080
```

Kiểm tra:

```bash
curl http://localhost:8001/health
```

---

## API

### POST `/chat`

```bash
curl -X POST http://localhost:8001/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "Wifi quán là gì?", "session_id": "sess-001"}'
```

Response:

```json
{
  "session_id": "sess-001",
  "intent": "faq",
  "agent": "faq_agent",
  "answer": "Wifi của quán là Highlands_Guest, mật khẩu highlands123.",
  "latency_ms": 320.5
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

Events:

```
data: {"type":"metadata","data":{"ttft_ms":85.3,"intent":"consultant","cache_hit":false}}
data: {"type":"token","data":{"token":"Dạ "}}
data: {"type":"clause","data":{"clause":"Dạ, em gợi ý vài món phù hợp."}}
...
data: [DONE]
```

---

## Câu hỏi demo

| Intent | Câu hỏi |
|--------|---------|
| FAQ | "Wifi quán là gì?" · "Mấy giờ đóng cửa?" · "Có ship không?" |
| Order | "Cho anh 1 bạc xỉu đá size L" · "Cho em latte M" |
| Consultant | "Có gì ngon rẻ không?" · "Gợi ý món ít ngọt" |
| Ignore | "Hôm nay thời tiết thế nào?" · "Hello" |

---

## Benchmark

```bash
# Latency & throughput
python scripts/benchmark_chat_api.py

# Semantic cache
python scripts/benchmark_intelligent_cache_500.py

# Router latency
python scripts/benchmark_router_latency.py
```

---

## Cấu trúc thư mục

```
app/
├── agents/        # Router, Order, Consultant, FAQ, Ignore
├── api/           # FastAPI routes
├── cache/         # Exact cache + Semantic cache
├── core/          # Config, schemas, constants
├── llm/           # SGLang / Mock client
├── middleware/    # Rate limiting
├── prompts/       # System prompts cho từng agent
├── queueing/      # Concurrency control (Semaphore + retry)
├── rag/           # Neo4j retriever, embedding, reranker
├── services/      # ChatService (orchestration)
├── session/       # Session store + auto-summarization
├── streaming/     # SSE helpers
└── utils/         # TTS preprocess

scripts/           # Data generation, training, benchmark
tests/             # Unit & integration tests
```

---

## Tác giả

**Bùi Công Bằng** — [github.com/Buicongbang04](https://github.com/Buicongbang04)
