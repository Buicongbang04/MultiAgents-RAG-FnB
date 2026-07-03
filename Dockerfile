# Backend API container (CPU). Model serving (SGLang) chạy trên host GPU riêng.
# ML/serving deps (torch, sentence-transformers, sglang) KHÔNG cài ở đây —
# backend boot được với EMBEDDING_BACKEND=mock / RERANKER_BACKEND=null và
# trỏ LLM_BACKEND=sglang tới server ngoài.
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# Cài dependencies trước để tận dụng layer cache.
COPY requirements.txt requirements-ui.txt ./
RUN pip install --upgrade pip \
    && pip install -r requirements.txt \
    && pip install -r requirements-ui.txt

COPY . .

EXPOSE 8001

# Healthcheck gọi endpoint /health của app.
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://localhost:8001/health').status==200 else 1)"

CMD ["python", "run.py", "--host", "0.0.0.0", "--port", "8001"]
