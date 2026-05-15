set -e

echo "=================================================="
echo "MultiAgents-RAG-FnB Restart Script"
echo "=================================================="

PROJECT_DIR="$HOME/Documents/MultiAgents-RAG-FnB"
CONDA_ENV="fiai"
NEO4J_CONTAINER="neo4j-fnb"
NEO4J_PASSWORD="bangbcfiai"

cd "$PROJECT_DIR"

echo "[INFO] Project dir: $PROJECT_DIR"

if [ ! -f ".env" ]; then
    echo "[WARN] .env file not found. Please create .env before running production mode."
fi


echo "[INFO] Checking Neo4j Docker container..."

if docker ps --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
    echo "[INFO] Neo4j already running"
elif docker ps -a --format '{{.Names}}' | grep -q "^${NEO4J_CONTAINER}$"; then
    echo "[INFO] Neo4j container exists but stopped. Starting..."
    docker start "$NEO4J_CONTAINER"
else
    echo "[INFO] Creating and starting Neo4j container..."
    docker run \
        --name "$NEO4J_CONTAINER" \
        -p 7474:7474 \
        -p 7687:7687 \
        -e NEO4J_AUTH=neo4j/${NEO4J_PASSWORD} \
        -d neo4j:5
fi

echo "[INFO] Waiting for Neo4j..."
sleep 8

echo "[INFO] Embedding graph if needed..."
python -m scripts.embed_graph || echo "[WARN] embed_graph failed or skipped"

echo "[INFO] Quick RAG sanity check..."
python -m tests.rag_retrieval_benchmark || echo "[WARN] RAG benchmark failed or skipped"

echo "[INFO] Starting FastAPI..."
python run.py