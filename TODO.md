# Obsidian Brain - TODO

## Extraction Agent Pipeline

### Sıradaki İşler

- [ ] **MCP Server** - Graph tool'larını agent'a expose et
  - search_entity, add_entity, add_relation, get_entity, list_entities
  - NetworkX graph'ı üzerinde çalışacak

- [ ] **Extraction Skills** - Agent'a ne yapacağını söyleyen instructions
  - graph.md schema'sını referans alacak
  - Tool kullanım örnekleri
  - Extraction akışı

- [ ] **İşlenmemiş Not Tracking** - Hangi notlar extract edildi
  - processed_notes.json veya SQLite
  - Hash ile değişiklik tespiti (incremental için)

### Sonraki Adımlar

- [ ] Incremental extraction (sadece değişen notlar)
- [ ] Entity deduplication stratejisi
- [ ] Confidence threshold ve filtering

## Research

- [ ] **Graphiti Repo Detaylı Araştırma**
  - MCP server tasarımı (`mcp_server/` folder)
  - Graph traversal algoritmaları (BFS, node distance reranking)
  - Hybrid search: BM25 + vector + graph traversal
  - Reranking stratejileri: RRF, MMR, cross-encoder
  - `search_utils.py` - port edilebilir algoritmalar
  - Custom entity types via Pydantic
  - Temporal model: valid_at, invalid_at
  - Ref: https://github.com/getzep/graphiti
