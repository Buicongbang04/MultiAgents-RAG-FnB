#!/usr/bin/env bash
# Tham số server (model, port, context length, mem fraction...) nằm trong
# configs/app.yaml → llm.model và llm.server. Thêm cờ SGLang tuỳ ý sau "--", ví dụ:
#   bash scripts/SGLang_Server.sh -- --disable-flashinfer
cd "$(dirname "$0")/.." && exec python -m scripts.launch_sglang "$@"
