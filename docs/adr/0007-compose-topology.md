# ADR 0007: Docker Compose cho hạ tầng và backend

- Trạng thái: Đã chấp nhận
- Ngày: 2026-11-01

## Bối cảnh

Dự án cần Postgres (đơn hàng), Neo4j (graph RAG), backend FastAPI và tuỳ chọn một SGLang server. Trước đây Neo4j được chạy thủ công bằng `docker run`, Postgres chưa có, và test tích hợp dùng chung graph với môi trường dev nên có thể xoá dữ liệu dev. Mặc định hiện nay là LLM cloud, SGLang chỉ còn cần cho máy có GPU.

## Quyết định

Dùng một `docker-compose.yml` gồm:

- `postgres` (16, host port 5433) và `neo4j` (5, 7474/7687), có volume đặt tên và healthcheck.
- `postgres-test` (5434) và `neo4j-test` (7475/7688) thuộc profile `test`, dùng `tmpfs` nên dữ liệu mất khi dừng.
- `backend` build từ `Dockerfile` (python:3.11-slim, `uv sync --frozen --extra ml --no-dev`), phụ thuộc postgres và neo4j ở trạng thái healthy; `backend-gpu` (profile `gpu`) thêm GPU reservation.
- `sglang` thuộc profile `local-llm`, cần GPU, không tự chạy.

## Lý do

- Một lệnh `docker compose up -d` dựng toàn bộ hạ tầng, giống nhau trên mọi máy.
- Dịch vụ test tách port và dùng tmpfs nên test tích hợp không bao giờ xoá graph/DB dev.
- Port host 5433 tránh xung đột với Postgres sẵn có (15432) và giữ 5432 trống.
- SGLang là profile opt-in vì cloud là mặc định; không bắt mọi người kéo image GPU nặng.

## Phương án đã cân nhắc

- Chỉ compose hạ tầng, backend chạy trên host: nhanh khi dev nhưng khó tái lập khi triển khai.
- Dùng chung một DB cho dev và test với schema/database riêng: dễ nhầm lẫn, Neo4j community không hỗ trợ nhiều database.
- Chạy SGLang luôn bật: tốn VRAM và thời gian khởi động khi không cần.

## Hệ quả

- Cần Docker và `NEO4J__PASSWORD` trong `.env` (compose báo lỗi nếu thiếu).
- Image backend lớn do PyTorch/CUDA wheel.
- Container `neo4j-fnb` cũ chiếm 7474/7687 phải dừng trước khi chạy compose.
