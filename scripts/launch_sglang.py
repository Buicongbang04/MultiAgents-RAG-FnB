"""
Khởi động SGLang server với tham số từ configs/app.yaml (llm.model, llm.server.*).

    python -m scripts.launch_sglang            # chạy server
    python -m scripts.launch_sglang --dry-run  # chỉ in câu lệnh
    python -m scripts.launch_sglang -- --disable-flashinfer   # thêm cờ tuỳ ý
"""

import os
import shlex
import sys

from app.core.config import get_settings


def build_command(extra_args: list[str]) -> list[str]:
    settings = get_settings()
    server = settings.llm.server

    return [
        sys.executable, "-m", "sglang.launch_server",
        "--model-path", settings.llm.model,
        "--host", server.host,
        "--port", str(server.port),
        "--context-length", str(server.context_length),
        "--mem-fraction-static", str(server.mem_fraction_static),
        "--max-running-requests", str(server.max_running_requests),
        "--chunked-prefill-size", str(server.chunked_prefill_size),
        "--schedule-policy", server.schedule_policy,
        *extra_args,
    ]


def main() -> None:
    args = sys.argv[1:]
    dry_run = "--dry-run" in args
    extra_args = args[args.index("--") + 1:] if "--" in args else []

    command = build_command(extra_args)
    print(shlex.join(command))

    if not dry_run:
        os.execvp(command[0], command)


if __name__ == "__main__":
    main()
