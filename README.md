# Obsidian Brain

Obsidian vault'unu AI ajan için "bilgi beyni" haline getiren sistem.

## Features

- **Graph Index**: Wikilinks, tags, folders → NetworkX graph + Neo4j
- **Vector Index**: Semantic search with local embeddings (multilingual)
- **CLI**: Build, search, inspect commands

## Installation

```bash
cd ~/code/obsidian-brain
uv sync
```

## Usage

### Build Graph
```bash
# Parse vault and create graph
uv run python -m src.cli build ~/obsidian_emin/obsidian_vault

# Import to Neo4j (optional)
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 -e NEO4J_AUTH=neo4j/obsidian neo4j:latest
uv run python -m src.cli neo4j-import data/graph.json --clear
```

### Embed Notes
```bash
# Generate embeddings for all notes
uv run python -m src.cli embed ~/obsidian_emin/obsidian_vault
```

### Search
```bash
# Semantic search
uv run python -m src.cli search "seni seviyorum"

# Inspect a note's connections
uv run python -m src.cli inspect data/graph.json "path/to/note.md"
```

## Tech Stack

- **Graph**: NetworkX + JSON / Neo4j
- **Vector DB**: sqlite-vec
- **Embeddings**: sentence-transformers (paraphrase-multilingual-MiniLM-L12-v2)
- **CLI**: Click

## Project Structure

```
src/
├── cli.py              # CLI commands
├── parser/
│   ├── markdown.py     # Wikilink, tag, frontmatter parser
│   └── vault.py        # Vault traversal
├── graph/
│   ├── builder.py      # NetworkX graph builder
│   ├── neo4j_storage.py # Neo4j import
│   └── visualize.py    # Pyvis visualization
└── vector/
    ├── embedder.py     # Local embeddings
    └── store.py        # sqlite-vec storage
```
