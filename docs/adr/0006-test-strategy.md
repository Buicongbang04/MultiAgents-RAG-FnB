# 0006. Chiến lược kiểm thử: unit và integration tách biệt, LLM giả trong CI

Trạng thái: Chấp nhận (2026-11-01)

## Bối cảnh

- Dự án phụ thuộc Postgres, Neo4j, model ML nặng (bge-m3, reranker) và LLM trả phí. Chạy tất cả trong CI vừa chậm vừa tốn tiền, nhưng chỉ dùng mock thì không bắt được lỗi SQL, migration hay Cypher.
- Cần một pipeline chạy được trên máy chủ CI không có GPU, không có khóa API.

## Quyết định

- Hai nhóm test. Unit (`tests/unit`) chạy mặc định (`pytest -q`), không cần dịch vụ ngoài. Integration (`tests/integration`, marker `integration`) bị loại khỏi lần chạy mặc định bằng `addopts` và chỉ chạy bằng `pytest -m integration`.
- CI có hai job. `unit` cài `uv sync --frozen` không kèm extra `ml`, đặt `EMBEDDING__BACKEND=mock`, `RERANKER__BACKEND=null`, `LLM__BACKEND=mock`. `integration` dùng service container `postgres:16` (cổng 5434) và `neo4j:5` (cổng 7688), đúng các cổng mặc định của test.
- DB test tách biệt hoàn toàn với DB dev: cổng và database riêng (`fnb_test`, Neo4j test). Schema do fixture chạy `alembic upgrade head`; mỗi test chạy trong transaction ngoài rồi rollback.
- CI không gọi LLM thật: dùng `ScriptedLLMClient` / backend `mock` (ADR 0005). Embedding trong integration dùng `MockEmbeddingClient`, nên không cần torch.
- Secret trong CI chỉ là giá trị giả (`NEO4J__PASSWORD`); không đặt `ANTHROPIC_API_KEY`.
- Các eval dùng LLM thật chạy thủ công, không nằm trong CI.
- `torch`, `peft`, `transformers` chỉ được import lười bên trong đường nạp model (router HF, intent extractor HF, embedding, reranker), để unit test chạy được khi không có extra `ml`.

## Lý do

- Unit nhanh, tất định, không cần hạ tầng, cho phản hồi tức thì trên mỗi commit.
- Integration trên Postgres/Neo4j thật bắt lỗi mà mock không thấy được (ràng buộc, migration, Cypher, vector index).
- LLM giả cho kết quả lặp lại được và không tốn tiền; kiểm thử chất lượng câu trả lời bằng LLM thật vốn không tất định nên không phù hợp làm cổng chặn merge.
- Không cài `ml` giảm thời gian cài đặt và dung lượng (torch kéo theo hàng GB CUDA).

## Phương án đã cân nhắc

- Chạy mọi thứ trong một job: chậm hơn và unit lỗi bị che bởi lỗi hạ tầng.
- Mock Postgres/Neo4j (SQLite, fake driver): khác hành vi thật, bỏ sót lỗi migration và Cypher.
- Gọi LLM thật trong CI: tốn tiền, có thể flake, cần giữ secret.
- Cài `ml` trong CI để test model thật: chậm, cần tải model hàng GB, không có GPU.

## Hệ quả

- Chất lượng thực của embedding, reranker và LLM không được CI kiểm chứng; phải chạy eval thủ công trước khi phát hành.
- Ai thêm import `torch`/`transformers` ở mức module sẽ làm job `unit` lỗi ngay (đây là tín hiệu có chủ đích).
- Integration phụ thuộc vào image `postgres:16` và `neo4j:5` và cổng cố định 5434/7688 trên runner.
