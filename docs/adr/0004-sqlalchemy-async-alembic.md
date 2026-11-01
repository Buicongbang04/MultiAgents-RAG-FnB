# ADR 0004: SQLAlchemy 2 (async) + Alembic cho truy cập và migration Postgres

- Trạng thái: Đã chấp nhận
- Ngày: 2026-11-01

## Bối cảnh

Backend là FastAPI bất đồng bộ và cần lớp truy cập Postgres cùng cơ chế quản lý schema có phiên bản.

## Quyết định

Dùng SQLAlchemy 2.x với `AsyncEngine`/`async_sessionmaker` (driver asyncpg) và Alembic cho migration. Quy ước đặt tên constraint (`MetaData.naming_convention`) giúp tên ổn định giữa các migration. URL kết nối đọc từ cấu hình (`database.url`, mật khẩu qua `DATABASE__URL`) và có thể ghi đè bằng `alembic -x url=...`.

## Lý do

- Mô hình khai báo có kiểu (`Mapped[...]`), hỗ trợ ARRAY/JSONB của Postgres.
- Async khớp với FastAPI, không chặn event loop.
- Alembic là chuẩn de facto, có `alembic check` để phát hiện lệch giữa model và migration.
- Test tích hợp chạy migration thật rồi rollback từng test.

## Phương án đã cân nhắc

- SQLModel: lớp mỏng trên SQLAlchemy nhưng kém ổn định hơn với ARRAY/JSONB và kiểu phức tạp; thêm một tầng phụ thuộc.
- asyncpg thuần + SQL thủ công: không có migration, phải tự viết ánh xạ, dễ sai khi schema lớn dần.

## Hệ quả

- Mọi thay đổi schema phải có migration Alembic.
- Test cần Postgres thật (`fnb_test` ở cổng 5434), đánh dấu `integration`.
