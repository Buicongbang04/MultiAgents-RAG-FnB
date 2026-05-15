python -m sglang.launch_server \
  --model-path Qwen/Qwen2.5-7B-Instruct-AWQ \
  --host 0.0.0.0 \
  --port 30000 \
  --context-length 4096 \
  --mem-fraction-static 0.65 \
  --max-running-requests 1 \
  --chunked-prefill-size 1024 \
  --schedule-policy lpm 
  # --disable-flashinfer