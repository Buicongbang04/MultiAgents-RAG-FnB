#!/usr/bin/env bash

set -e

echo "=================================================="
echo "MultiAgents-RAG-FnB Restart Script"
echo "=================================================="

PROJECT_DIR="$HOME/Documents/MultiAgents-RAG-FnB"

cd "$PROJECT_DIR"

echo "Activating conda env: fiai"

# source ~/miniconda3/etc/profile.d/conda.sh
# conda activate fiai

echo "Checking Neo4j..."

if ! pgrep -f "neo4j" > /dev/null; then
    echo "[INFO] Starting Neo4j..."
    docker run \
    --name neo4j-fnb \
    -p7474:7474 \
    -p7687:7687 \
    -e NEO4J_AUTH=neo4j/bangbcfiai \
    -d neo4j:5
else
    echo "[INFO] Neo4j already running"
fi

sleep 5

echo "Checking embedding graph..."

python -m scripts.embed_graph || true

echo "Quick RAG sanity check..."

python -m tests.rag_retrieval_benchmark || true

echo "Starting FastAPI..."

python run.py 
