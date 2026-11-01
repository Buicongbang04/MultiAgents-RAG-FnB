# ADR 0003: Dùng uv + pyproject.toml + lockfile cho môi trường

- Trạng thái: Đã chấp nhận
- Ngày: 2026-11-01

## Bối cảnh

Các file `requirements*.txt` cũ để nhiều dependency ML không ghim phiên bản. Ngày 2026-11-01 việc cài mới bị lỗi:

- `datasets` 2.14 không tương thích với `pyarrow` 26.
- `huggingface_hub` 2.x không tương thích với `transformers` 4.57.

Môi trường không tái lập được giữa các máy và các thời điểm cài.

## Quyết định

Dùng `uv` với `pyproject.toml` và `uv.lock` được commit. Dependency lõi nằm trong `dependencies`; các nhóm tuỳ chọn là extras: `ml`, `ui`, `serving` (SGLang). Cài bằng `uv sync --extra ml --extra ui`. Các file `requirements*.txt` giữ lại nhưng chỉ chứa ghi chú trỏ tới `pyproject.toml`. Ghim đã kiểm chứng: `transformers==4.57.6`, `huggingface_hub==0.36.0`, `datasets>=3,<4.4`.

## Lý do

- Lockfile cho kết quả cài giống nhau trên mọi máy.
- `uv` nhanh và quản lý cả venv, Python và lock.
- Extras tách phần nặng (GPU/CUDA) khỏi phần lõi.

## Phương án đã cân nhắc

- pip-tools: có lock nhưng cần thêm công cụ cho venv, chậm hơn.
- conda-lock: nặng, trộn conda và pip phức tạp cho stack PyTorch/SGLang.
- Poetry: resolver chậm, xử lý wheel CUDA kém thuận tiện hơn.

## Hệ quả

- Thêm hoặc đổi dependency phải chạy `uv lock` và commit `uv.lock`.
- Người dùng cần cài `uv`; `Instructions.md` được cập nhật.
- Phiên bản `torch` do lockfile quyết định (hiện 2.7.1 với CUDA 12.6).
