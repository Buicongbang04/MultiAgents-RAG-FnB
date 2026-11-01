# ADR 0010: Neo4j là bản chiếu dùng một lần, xóa và dựng lại từ Postgres

- Trạng thái: Đã chấp nhận
- Ngày: 2026-11-01

## Bối cảnh

Postgres là nguồn sự thật của catalog (ADR 0002); Neo4j chỉ phục vụ retrieval (keyword, vector, mở rộng đồ thị). Cần quyết định cách đồng bộ, cách xử lý món "tạm hết"/ẩn, và cách tạo mô tả theo size cho từng MenuItem khi Postgres chỉ lưu một mô tả cho mỗi món.

## Quyết định

- `project_catalog` xóa toàn bộ graph rồi dựng lại từ Postgres (MenuItem theo từng biến thể, FAQ, Chunk, Entity, Category, liên kết miền). Chạy lại luôn cho cùng kết quả; không có đồng bộ gia tăng.
- Món `sold_out`/`hidden` vẫn được chiếu thành MenuItem với `available = false`. Mọi truy vấn trả về MenuItem (keyword, vector index, fallback cosine, mở rộng cùng danh mục trong `retriever.py`, và `menu.py`) lọc `coalesce(m.available, true)`.
- Mô tả của từng MenuItem được dựng lại từ mô tả món: `"<tên> size S (35.000đ) — ..."`, chèn mệnh đề size/giá ngay sau tên.

## Lý do

- Bật/tắt "tạm hết" chỉ đổi cờ `available`, không cần xóa node hay embed lại, nên rẻ và đảo ngược được.
- Dựng lại toàn bộ đơn giản, idempotent, không bao giờ lệch so với Postgres; dữ liệu nhỏ (55 biến thể) nên chi phí thấp.
- Mô tả có mệnh đề size/giá giữ văn bản retrieval tương đương dữ liệu cũ.

## Phương án đã cân nhắc

- Xóa node của món không bán: bị loại vì khi bán lại phải tạo và embed lại, và dễ lệch trạng thái.
- Đồng bộ gia tăng (upsert theo thay đổi): bị loại vì phức tạp, dễ sót node mồ côi.
- Lưu mô tả riêng cho từng size trong Postgres: bị loại vì thêm dữ liệu cần bảo trì.

## Hệ quả

- Câu chữ riêng theo size ở M/L của dữ liệu mock bị mất (mô tả lấy từ dòng size S); số quan hệ `MENTIONS` từ MenuItem tăng từ 116 lên 123, tổng quan hệ 838.
- Mỗi truy vấn MenuItem mới phải nhớ điều kiện `coalesce(m.available, true)`.
- Dựng lại graph xóa embedding, nên cần chạy lại bước embed (bootstrap làm việc này, trừ khi `--skip-embed`).
