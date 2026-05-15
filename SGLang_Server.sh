python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 30000 \
  --context-length 4096 \
  --mem-fraction-static 0.72 \
  --max-running-requests 2 \
  --chunked-prefill-size 2048 \
  --schedule-policy lpm