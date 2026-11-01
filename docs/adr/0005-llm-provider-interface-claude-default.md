# ADR 0005: Một giao diện LLM, ba backend, Claude làm mặc định cho demo

- Trạng thái: Đã chấp nhận
- Ngày: 2026-11-01

## Bối cảnh

Hệ thống cần LLM để sinh câu trả lời, trích xuất ý định (NLU) và tóm tắt hội thoại. Trước đây chỉ có SGLang cục bộ (Qwen 7B AWQ) và mock. Bản demo cần chất lượng tiếng Việt và khả năng trả JSON đúng schema ổn định hơn mô hình 7B. Các agent/service đã gọi `generate`/`stream_generate` và truyền `metadata["fallback_answer"]`, nên giao diện này phải giữ nguyên.

## Quyết định

- Giữ một giao diện `BaseLLMClient` với ba backend: `mock`, `sglang` (cục bộ, giữ lại), `anthropic` (Claude). `vllm` vẫn chưa cài đặt.
- Thêm `generate_structured(request, schema)` (Pydantic) và `aclose()`; lỗi chuẩn hóa: `LLMUnavailableError`, `LLMStructuredOutputError`, `LLMConfigurationError`.
- Backend Claude dùng `claude-opus-5-5` cho mọi tác vụ, điều chỉnh độ sâu bằng `effort` theo từng request (`output_config`), không gửi `temperature`/`top_p`. Structured output dùng `messages.parse(output_format=...)`; system prompt gửi dạng block có `cache_control: ephemeral` để cache phần prompt cố định; `stop_reason == "refusal"` được chuyển thành `LLMUnavailableError` để caller dùng fallback.
- Mặc định trong YAML gọi thẳng Anthropic: `llm.anthropic.base_url` rỗng (endpoint mặc định của SDK), `model: claude-opus-5-5`, khóa chỉ lấy từ env `ANTHROPIC_API_KEY`. Gateway OmniRoute của chủ dự án chỉ là override theo máy trong `.env` (git-ignored): `LLM__ANTHROPIC__BASE_URL` (gốc domain, không có `/v1`) và `LLM__ANTHROPIC__MODEL=auto/claude-opus-5.5`; không commit vào YAML để khóa của người khác không bị gửi tới host bên thứ ba. Với gateway, model thực sự phục vụ do gateway định tuyến nên phải kiểm tra `response.model` (đã xác nhận trả `claude-opus-5-5`).
- Mặc định YAML vẫn là `llm.backend: mock`; Part 1 mới chuyển mặc định demo sang Claude.
- `ScriptedLLMClient` trả lần lượt các phản hồi định sẵn và ghi lại request, dùng cho unit test và eval offline.
- `SGLangClient` dùng một `httpx.AsyncClient` dùng chung, đóng ở `aclose()`; có `generate_structured` qua `response_format={"type":"json_schema"}`.

## Lý do

- Một opus ở `effort` thấp/trung bình đơn giản hơn việc duy trì nhiều model nhỏ và thường đủ nhanh (thử thật: khoảng 2-4 giây, TTFT khoảng 2-3 giây qua gateway); caching theo từng model nên một model giữ nguyên một vùng cache.
- Structured outputs loại bỏ bước parse JSON mong manh của NLU.
- Prompt hệ thống cố định là tiền tố cache lý tưởng.
- Giữ SGLang để chạy offline/không phụ thuộc dịch vụ ngoài.

## Phương án đã cân nhắc

- Dùng model nhỏ hơn (Haiku/Sonnet) cho NLU: rẻ hơn nhưng thêm một cấu hình, mất chung cache, và chưa có số đo chứng minh cần thiết.
- Gọi gateway bằng giao diện kiểu OpenAI: mất các tính năng gốc (effort, structured outputs, cache_control).
- Gọi thẳng api.anthropic.com: cần khóa riêng; gateway đã có sẵn.

## Hệ quả

- Rủi ro: gateway có thể không chuyển tiếp mọi tính năng của Claude (đã kiểm tra `effort`, structured output, streaming; chưa kiểm tra cache_control hiệu quả, nên theo dõi `usage.cache_read_input_tokens`) và có thể đổi model phục vụ phía sau; luôn đối chiếu `response.model`.
- WAF của gateway chặn User-Agent mặc định `Anthropic/Python ...` bằng 403, nên client đặt User-Agent riêng khi dùng `base_url` tùy chỉnh.
- Thiếu khóa khi chọn backend `anthropic` làm ứng dụng không khởi động (fail-fast ở lifespan).
- Không có temperature cho Claude: tính xác định không còn điều chỉnh được; test dùng `ScriptedLLMClient`.
