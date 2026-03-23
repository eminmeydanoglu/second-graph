# Obsidian Brain

## Neo4j

Start the Neo4j container:

```bash
docker run -d \
  --name obsidian-brain-neo4j \
  -p 7474:7474 \
  -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/obsidian \
  -v "$PWD/data/neo4j:/data" \
  neo4j:5
```

Neo4j Browser: `http://localhost:7474`

Connection details:

- Username: `neo4j`
- Password: `obsidian`
- Bolt URI: `bolt://localhost:7687`

## Setup

```bash
uv sync
```

## Run

Sync a vault to Neo4j:

```bash
uv run obsidian-brain sync-vault /path/to/vault
```

Run a search:

```bash
uv run obsidian-brain search "query"
```
