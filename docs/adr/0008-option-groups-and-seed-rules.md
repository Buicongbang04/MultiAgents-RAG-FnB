# ADR 0008: Mô hình nhóm tùy chọn (option group) và quy tắc seed catalog

- Trạng thái: Đã chấp nhận
- Ngày: 2026-11-01

## Bối cảnh

Khách cần chọn đường, đá, nóng/đá cho đồ uống. Dữ liệu mock (`menu.csv`) chỉ có 55 dòng theo từng size, không có khái niệm tùy chọn. Cần quyết định mô hình tùy chọn và cách gom dòng mock thành món (xem spec §3, §10).

## Quyết định

- Ba nhóm tùy chọn lưu trong bảng `option_groups`: `sugar` (Đường: không/ít/bình thường/nhiều, mặc định bình thường), `ice` (Đá: không/ít/bình thường, mặc định bình thường), `temperature` (Nóng/Đá: đá/nóng, mặc định đá). Gán cho món qua bảng `product_option_groups`.
- Quy tắc gán: `food` không có nhóm nào; tag `hot` -> `sugar`; tag `iced` -> `sugar, ice`; `freeze` -> `sugar`; `tea` -> `sugar, ice`; `coffee` không có tag `hot` lẫn `iced` (Cappuccino, Latte, Americano) -> `sugar, ice, temperature`.
- Gom 55 dòng thành 21 món theo slug của tên (NFC, bỏ dấu, `đ`->`d`, chữ thường, ký tự không phải chữ/số -> `-`). Mô tả lấy từ dòng size S sau khi xoá mệnh đề size/giá; tags và nguyên liệu là hợp của các size. Id biến thể giữ id cũ `menu_xxxx`.
- Seed idempotent bằng `INSERT ... ON CONFLICT DO UPDATE`; `reset=True` TRUNCATE các bảng catalog trước.
- Tùy chọn miễn phí (không cộng tiền): giá chỉ phụ thuộc size.

## Lý do

- Chỉ cà phê không có tag nóng/đá mới thực sự có thể phục vụ cả hai kiểu, nên chỉ các món đó cần hỏi `temperature`; các món còn lại đã ngầm định nóng hoặc đá.
- Miễn phí giúp giữ giá là số nguyên VND theo size, không cần bảng phụ thu và đơn giản hoá tính tiền (spec §3).
- Tùy chọn là dữ liệu trong DB nên trang quản trị (Part 3) có thể chỉnh mà không sửa code.

## Phương án đã cân nhắc

- Tùy chọn nhúng cứng trong code: bị loại vì admin không chỉnh được.
- Tùy chọn có phụ thu: bị loại vì thêm độ phức tạp chưa cần.
- Hỏi `temperature` cho mọi cà phê: gây hỏi thừa với món đã là "đá" hoặc "nóng".

## Hệ quả

- Thêm món mới cần gán nhóm tùy chọn tương ứng (hoặc dùng quy tắc `option_groups_for`).
- Bản FAQ về giao hàng và đặt trước được viết lại (faq_0010, faq_0011, faq_0028) để khớp thực tế: quán tự giao, phí 15.000đ, cho phép đặt trước tối đa 2 ngày, 7:00-22:00. Các giá trị này nằm ở `shop_settings`.
