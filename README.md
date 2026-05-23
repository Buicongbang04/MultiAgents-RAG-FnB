# MultiAgents RAG FnB

Chatbot tư vấn F&B (Highlands Coffee) sử dụng kiến trúc **Multi-Agent + Graph RAG** chạy hoàn toàn Local — không phụ thuộc Cloud API.

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

### 2. Tạo conda environment

```bash
conda create -n fiai python=3.11 -y
conda activate fiai
```

### 3. Cài dependencies

```bash
pip install -r requirements.txt          # Core API
pip install -r requirements-ml.txt       # HuggingFace, training
pip install -r requirements-ui.txt       # Chainlit
pip install -r requirements-serving.txt  # SGLang (cần CUDA)
```

### 4. Tạo file `.env`

```env
# LLM Backend
LLM_BACKEND=sglang
LLM_BASE_URL=http://localhost:30000/v1
GENERATOR_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USER=neo4j
NEO4J_PASSWORD=your_password

# RAG
RAG_RETRIEVAL_MODE=hybrid_graph
EMBEDDING_BACKEND=sentence_transformers
EMBEDDING_MODEL=BAAI/bge-m3

# Reranker
RERANKER_BACKEND=bge
RERANKER_MODEL=BAAI/bge-reranker-v2-m3
RERANKER_THRESHOLD=0.7
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
python -m scripts.ingest_mock_to_neo4j   # Xóa graph cũ + nạp menu/FAQ/docs
python -m scripts.embed_graph            # Tạo embeddings cho vector search
```

Tạo Neo4j vector index (chạy 1 lần trong Neo4j Browser):

```cypher
CREATE VECTOR INDEX menu_embedding IF NOT EXISTS
FOR (m:MenuItem) ON m.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX chunk_embedding IF NOT EXISTS
FOR (c:Chunk) ON c.embedding
OPTIONS {indexConfig: {`vector.dimensions`: 1024, `vector.similarity_function`: 'cosine'}};

CREATE VECTOR INDEX faq_embedding IF NOT EXISTS
FOR (f:FAQ) ON f.embedding
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

### Terminal 2 — Backend + UI

```bash
conda activate fiai

python run.py                          # Backend only (port 8001)
python run.py --with-ui                # Backend + Chainlit UI (port 8501)
python run.py --with-ui --ui-port 8080 # Đổi UI port
python run.py --reload                 # Hot-reload (dev)
```

Sau khi start:
- **API docs:** http://localhost:8001/docs
- **Chat UI:** http://localhost:8501

Kiểm tra hệ thống:

```bash
curl http://localhost:8001/health
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
├── llm/           # SGLang client
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
chainlit_app.py    # Chainlit UI (word-by-word animation)
run.py             # Entry point duy nhất
```

---

## Tác giả

**Bùi Công Bằng** — [github.com/Buicongbang04](https://github.com/Buicongbang04)
