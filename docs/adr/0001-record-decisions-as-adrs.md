# ADR 0001: Ghi lại các quyết định kiến trúc bằng ADR

- Trạng thái: Đã chấp nhận
- Ngày: 2026-11-01

## Bối cảnh

Dự án đang chuyển từ chatbot Graph-RAG sang nền tảng đặt hàng F&B, với nhiều quyết định kỹ thuật (LLM backend, cơ sở dữ liệu, môi trường, ...). Chủ dự án yêu cầu mọi quyết định phải được giải thích rõ ràng (spec §11) để có thể đọc lại và kiểm chứng sau này.

## Quyết định

Mỗi quyết định kiến trúc quan trọng được ghi thành một file ADR trong `docs/adr/`, đặt tên `NNNN-tieu-de.md`, viết bằng tiếng Việt theo mẫu trong `0000-template.md`.

## Lý do

- Giữ lại lý do đằng sau quyết định, không chỉ kết quả.
- Nhẹ, nằm cùng repo, được review cùng code.
- Tránh tranh luận lại những vấn đề đã được cân nhắc.

## Phương án đã cân nhắc

- Wiki/Notion ngoài repo: dễ lệch với code, không đi cùng lịch sử git.
- Chỉ ghi trong commit message hoặc README: khó tìm, thiếu cấu trúc.

## Hệ quả

- Mỗi thay đổi kiến trúc đáng kể cần kèm một ADR mới (hoặc cập nhật trạng thái ADR cũ).
- ADR không bị xoá; khi đổi hướng thì đánh dấu "Đã thay thế bởi".
