# ADR 0002: PostgreSQL là nguồn sự thật, Neo4j chỉ là bản chiếu (projection)

- Trạng thái: Đã chấp nhận
- Ngày: 2026-11-01

## Bối cảnh

Hiện catalog (món, giá, FAQ, chính sách) được nạp từ file mock thẳng vào Neo4j. Nền tảng đặt hàng cần đơn hàng, giá tiền và chỉnh sửa menu từ trang quản trị (Part 3), nên dữ liệu gốc cần ràng buộc và giao dịch. Xem spec §6.1.

## Quyết định

Catalog nằm trong PostgreSQL (bảng `products`, `product_variants`, `option_groups`, `faqs`, `policy_chunks`, `shop_settings`...). Neo4j được dựng lại từ Postgres bằng bước chiếu (graph writer), giữ nguyên hình dạng đồ thị mà retriever đang dùng.

## Lý do

- Giao dịch ACID cho đơn hàng và chỉnh sửa menu.
- Ràng buộc ở tầng DB: giá > 0, size duy nhất theo món, khoá ngoại, trạng thái hợp lệ.
- Trang quản trị (Part 3) sửa dữ liệu qua SQL thông thường rồi chiếu lại sang Neo4j.
- Neo4j có thể xoá và dựng lại bất kỳ lúc nào mà không mất dữ liệu.

## Phương án đã cân nhắc

- Giữ Neo4j làm nguồn sự thật: bị loại vì không có ràng buộc dạng CHECK/UNIQUE linh hoạt và không có giao dịch phù hợp cho đơn hàng.
- Ghi song song vào cả hai nơi: dễ lệch dữ liệu, khó xác định bên nào đúng.

## Hệ quả

- Cần bước chiếu Postgres sang Neo4j (các task sau trong Part 0).
- Mọi thay đổi catalog phải đi qua Postgres, không sửa trực tiếp Neo4j.
- Id cũ (`menu_0001`, `faq_0001`, `doc_0001`) được giữ làm khoá để đồ thị không đổi.
