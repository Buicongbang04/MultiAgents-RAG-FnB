# MultiAgents-RAG-FnB

Production-ready Multi-Agent RAG chatbot system for Food & Beverage consultation using:

- Multi-Agent Architecture
- Hybrid Graph RAG
- Neo4j Knowledge Graph
- SGLang Serving
- Qwen2.5 Models
- Intelligent Semantic Cache
- FastAPI API Service

---

# 1. System Architecture

## Components

- **Router Agent**
  - Classifies user intent:
    - `order`
    - `consultant`
    - `faq`
    - `ignore`

- **Consultant Agent**
  - Recommends menu items using Hybrid Graph RAG.

- **FAQ Agent**
  - Handles frequently asked restaurant questions.

- **Ignore Agent**
  - Rejects out-of-domain questions politely.

- **Neo4j Knowledge Graph**
  - Stores:
    - Menu
    - FAQ
    - Documents
    - Relationships

- **SGLang Server**
  - Hosts Generator Model:
    - `Qwen2.5-7B-Instruct-AWQ`

- **Semantic Cache**
  - Exact cache
  - Embedding cache
  - Intent-aware cache policy

---

# 2. System Flow

```text
User
 ↓
FastAPI (/chat)
 ↓
Router Agent
 ↓
Intent Extraction
 ↓
Cache Lookup
 ↓ miss
Specialized Agent
 ↓
Hybrid Graph RAG
 ↓
SGLang Generator
 ↓
Response
```

---

# 3. Prerequisites

## Hardware

Recommended:

| Component | Requirement |
|------------|-------------|
| GPU | RTX 3060 12GB or above |
| RAM | 32GB+ |
| OS | Ubuntu / Pop!_OS Linux |
| Python | 3.11 |

Minimum:

- GPU 8GB VRAM
- CPU inference possible but slower

---

# 4. Clone Repository

```bash
git clone https://github.com/Buicongbang04/MultiAgents-RAG-FnB.git
cd MultiAgents-RAG-FnB
```

---

# 5. Create Python Environment

```bash
conda create -n fiai python=3.11 -y
conda activate fiai
```

Verify:

```bash
python --version
```

Expected:

```text
Python 3.11.x
```

---

# 6. Install Dependencies

```bash
pip install -r requirements.txt
```

If Flash Attention fails:

```bash
pip install ninja packaging wheel
```

---

# 7. Install Neo4j

## Docker (Recommended)

```bash
docker run \
  --name neo4j-fnb \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  -d neo4j:5
```

Open browser:

```text
http://localhost:7474
```

Login:

```text
username: neo4j
password: password
```

---

# 8. Configure Environment Variables

Create `.env`

```env
# App
HOST=0.0.0.0
PORT=8001

# LLM Backend
LLM_BACKEND=sglang
GENERATOR_MODEL=Qwen/Qwen2.5-7B-Instruct-AWQ
LLM_BASE_URL=http://localhost:30000/v1
LLM_API_KEY=EMPTY

# Neo4j
NEO4J_URI=bolt://localhost:7687
NEO4J_USERNAME=neo4j
NEO4J_PASSWORD=password

# Retrieval
RAG_RETRIEVAL_MODE=hybrid_graph

# Embedding
EMBEDDING_MODEL=BAAI/bge-m3
EMBEDDING_DEVICE=cpu

# Router
ROUTER_BACKEND=hf_merged
ROUTER_MODEL_PATH=models/router-qwen2.5-0.5b-merged

# Intelligent Cache
INTENT_EXTRACTOR_BACKEND=rule_based

# Cache
EXACT_CACHE_TTL_SECONDS=1800
SEMANTIC_CACHE_TTL_SECONDS=1800
```

---

# 9. Download Models

## Generator

SGLang auto-downloads:

```text
Qwen/Qwen2.5-7B-Instruct-AWQ
```

## Router

Place merged model:

```text
models/
└── router-qwen2.5-0.5b-merged
```

---

# 10. Start SGLang Server
### Terminal 1
Run:

```bash
bash scripts/SGLang_Server.sh
```

Expected log:

```text
Server started successfully
Listening on port 30000
```

Health check:

```bash
curl http://localhost:30000/v1/models
```

---

# 11. Build Neo4j Knowledge Graph

Run ingestion:

```bash
python -m scripts.ingest_mock_to_neo4j
```

Expected:

```text
FAQ ingested
Menu ingested
Documents ingested
Graph relationships created
```

Verify in Neo4j:

```cypher
MATCH (n)
RETURN count(n)
```

---

# 12. Start Application
### Terminal 2
Run:

```bash
bash scripts/restart_system.sh
```

Expected:

```text
Application startup complete.
Uvicorn running on:
http://0.0.0.0:8001
```

---

# 13. Verify System

## Swagger UI

Open:

```text
http://localhost:8001/docs
```

---

## Health Check

```bash
curl http://localhost:8001/health
```

Expected:

```json
{
  "status": "ok"
}
```

---

# 14. Test Chat API

Example request:

```bash
curl -X POST \
"http://localhost:8001/chat" \
-H "Content-Type: application/json" \
-d '{
  "message": "Wifi quán là gì?",
  "session_id": "test-session"
}'
```

Expected response:

```json
{
  "intent": "faq",
  "agent": "faq_agent",
  "answer":
  "Wifi của quán là Highlands_Guest,
   mật khẩu là highlands123."
}
```

---

# 15. Example Queries

## FAQ

```text
Wifi quán là gì?
Mấy giờ đóng cửa?
Có ship không?
```

---

## Consultant

```text
Có gì ngon rẻ không?
Recommend món dễ uống
Quán có bạc xỉu không?
```

---

## Order

```text
Cho anh một bạc xỉu đá
Cho em một latte size M
```

---

## Ignore

```text
Hôm nay thời tiết thế nào?
Ai là tổng thống Mỹ?
```

---

# 16. Run Benchmarks

## End-to-End Benchmark

```bash
python scripts/benchmark_chat_api.py
```

---

## Intelligent Cache Benchmark

```bash
python scripts/benchmark_intelligent_cache_500.py
```

Expected:

```json
{
  "pass": {
    "hit_rate": true,
    "order_no_cache": true,
    "cache_hit_latency_p95_under_250ms": true
  }
}
```

---

# 17. Troubleshooting

## CUDA Out Of Memory

Reduce concurrency:

```bash
--max-running-requests=1
```

Lower VRAM fraction:

```bash
--mem-fraction-static=0.65
```

---

## Neo4j Connection Failed

Check container:

```bash
docker ps
```

Restart:

```bash
docker restart neo4j-fnb
```

---

## SGLang Not Responding

Check:

```bash
curl http://localhost:30000/v1/models
```

Restart:

```bash
bash scripts/SGLang_Server.sh
```

---

# 18. Author

**Bùi Công Bằng**

GitHub:
https://github.com/Buicongbang04