docker run \
--name neo4j-fnb \
-p7474:7474 \
-p7687:7687 \
-e NEO4J_AUTH=neo4j/bangbcfiai \
-d neo4j:5