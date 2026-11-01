# 0009. Đẩy ML blocking ra khỏi event loop, fp16 trên GPU, warmup và health Postgres

Trạng thái: Chấp nhận (2026-11-01)

## Bối cảnh

- `SentenceTransformer.encode` và `CrossEncoder.predict` là lời gọi blocking nhưng được gọi trực tiếp trong code async. Mỗi lần gọi chặn cả event loop nên mọi khách hàng đang chat bị phục vụ tuần tự (README §17).
- Model nạp ở fp32; reranker lazy-load nên request đầu tiên sau khi khởi động chịu độ trễ 35–39 s (đo ở spec §2).
- `/health` không phản ánh Postgres, nguồn sự thật của catalog (ADR 0002).

## Quyết định

- `embed_text` / `embed_batch` chạy `model.encode` qua `asyncio.to_thread`. Reranker có thêm `async arerank(...)` (cùng tham số/kết quả với `rerank`) chạy `predict` trong thread; retriever `await arerank`. `NullReranker` cũng có `arerank`.
- Thêm `embedding.dtype` và `reranker.dtype` (`float16|float32`) trong `configs/app.yaml`. fp16 chỉ áp dụng khi device là `cuda`; device khác bị ép `float32`.
- Thêm `server.warmup_on_startup`. Khi bật, lifespan chạy một `embed_text("khởi động")` và một cặp rerank trước khi nhận request, và log thời lượng. Lifespan vẫn tạo LLM client lúc khởi động và đóng khi tắt. Lifespan cũng gọi `setup_logging()` để log INFO không bị nuốt khi chạy bằng uvicorn.
- Warmup lỗi không làm sập khởi động: bắt exception, log warning kèm lỗi, tiếp tục phục vụ; model nạp lười ở request đầu (degrade thay vì fail-fast, vì thiếu warmup chỉ làm chậm request đầu chứ không sai).
- `BGEReranker._load` dùng `threading.Lock` kiểu double-checked vì `arerank` chạy trong nhiều thread; tránh nạp model hai lần và vọt VRAM khi warmup tắt.
- `/health` bọc `ping_database()` bằng `asyncio.wait_for` với `health.db_timeout_seconds` (2.0); quá hạn thì ghi warning và `postgres=false`.
- `/health` thêm `"postgres": bool` (từ `ping_database()`); `status` là `"ok"` chỉ khi neo4j và postgres đều true, ngược lại `"degraded"`. Không bao giờ raise (lỗi neo4j cũng được bắt).
- `torch` / `sentence_transformers` chỉ import bên trong đường nạp model, không import ở mức module, để unit test với `embedding.backend=mock` không cần torch.

## Lý do

- Thread offload giữ event loop phục vụ các request khác (health, rate limit, session) trong lúc GPU tính toán. `encode`/`predict` nhả GIL trong phần tính toán nặng nên thread có hiệu quả thực.
- fp16 giảm một nửa VRAM trọng số, nên LLM 7B AWQ cùng embedding và reranker có cơ hội nằm vừa 12 GB.
- Warmup chuyển chi phí nạp model và khởi tạo CUDA kernel từ request đầu của khách sang lúc khởi động.
- Health "degraded" thay vì lỗi 500 giúp orchestrator và người vận hành thấy phụ thuộc nào hỏng.

## Phương án đã cân nhắc

- Process pool hoặc service inference riêng: tách biệt tốt hơn nhưng nặng và nhân đôi bộ nhớ model; chưa cần ở quy mô hiện tại.
- Giữ gọi đồng bộ và dựa vào `max_concurrency` của queue: không giải quyết việc loop bị chặn.
- Quantize int8 cho embedding/reranker: tốn công đánh giá chất lượng; fp16 gần như không đổi chất lượng.
- Warmup lười ở request đầu: giữ nguyên độ trễ 35–39 s cho khách đầu tiên.

## Hệ quả

- Khởi động chậm hơn khoảng 15 s khi warmup bật (đổi lấy request đầu nhanh). Có thể tắt bằng `SERVER__WARMUP_ON_STARTUP=false`.
- Hai thread có thể cùng dùng một model; `encode`/`predict` của sentence-transformers dùng được từ nhiều thread nhưng GPU vẫn tuần tự hóa phía driver. Giới hạn đồng thời vẫn do `queue.max_concurrency.embedding/reranker`.
- Reranker mới (`BGEReranker.warmup`) nuốt lỗi nạp như trước và rơi về thứ tự gốc.

## Số đo (2026-11-01, RTX 12 GB dùng chung với các process khác của chủ máy)

Chạy `uvicorn app.main:app` với `LLM__BACKEND=mock`, `EMBEDDING__DEVICE=cuda`, `RERANKER__DEVICE=cuda`, fp16, warmup bật. Chỉ tính process của backend (lọc theo PID bằng `nvidia-smi --query-compute-apps`):

| Chỉ số | Giá trị |
|---|---|
| VRAM của backend (bge-m3 + bge-reranker-v2-m3, fp16, sau warmup và 2 chat) | khoảng 2.3 GB (2336 MiB sau warmup, 2410 MiB sau chat) |
| Thời gian warmup lúc khởi động | 15.2 s |
| Thời gian từ lúc chạy tới khi `/health` trả lời | khoảng 23.4 s |
| `/chat` đầu tiên sau khởi động ("Wifi quán là gì?", intent faq, LLM mock) | 2.19 s |
| `/chat` thứ hai | 0.65 s |

Sau khi dừng backend (kill đúng PID đã chạy), PID đó không còn trong danh sách process dùng GPU.

Chưa đo: đồng tồn tại với LLM 7B cục bộ (SGLang). Lần này không chạy SGLang vì LLM mặc định đã chuyển sang cloud và chủ máy muốn giữ GPU trống. Mục tiêu "tổng < 12 GB cùng SGLang `--mem-fraction-static 0.6`" vì vậy chưa kiểm chứng; chỉ biết phần của embedding + reranker là khoảng 2.3 GB, còn LLM cục bộ sẽ cần đo lại nếu dùng backend local. Con số `/chat` đầu dùng LLM mock nên không gồm thời gian sinh của LLM.
