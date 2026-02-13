#!/usr/bin/env bash
set -euo pipefail

NAME="neo4j-test"
IMAGE="neo4j:2025.12.1"

if docker ps -a --format '{{.Names}}' | grep -qx "$NAME"; then
  docker start "$NAME" >/dev/null
else
  docker run -d \
    --name "$NAME" \
    -p 17474:7474 \
    -p 17687:7687 \
    -e NEO4J_AUTH=neo4j/obsidian \
    -e NEO4J_PLUGINS='["apoc"]' \
    "$IMAGE" >/dev/null
fi

echo "neo4j test instance ready: bolt://localhost:17687"
echo "export TEST_NEO4J_URI=bolt://localhost:17687"
echo "export TEST_NEO4J_USER=neo4j"
echo "export TEST_NEO4J_PASSWORD=obsidian"
