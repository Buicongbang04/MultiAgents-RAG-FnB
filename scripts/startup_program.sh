echo "=====Starting Neo4j database====="
bash scripts/run_neo4j.sh
echo "=====Neo4j database started====="

echo "=====Generating mock data====="
python scripts/generate_mock_data.py
echo "=====Generating mock data completed====="

echo "=====Ingesting mock data into vector database====="
python -m scripts.ingest_mock_to_neo4j
echo "=====Ingesting mock data completed====="

echo "=====Starting embedding graph====="
python -m scripts.embed_graph
echo "=====Embedding graph completed====="

echo "=====Starting SGLang Server====="
bash scripts/SGLang_Server.sh
echo "=====SGLang Server started====="