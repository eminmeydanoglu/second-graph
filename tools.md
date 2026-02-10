# MCP Tools Specification

Obsidian Brain için MCP tool tanımları. Anchor & Expand stratejisi üzerine kurulu.

## Retrieval Stratejisi

### Temel Prensip: Anchor & Expand

```
User Query
    │
    ▼
┌─────────────────────────────────────┐
│         ANCHOR PHASE                │
│  (Graph'a giriş noktası bul)        │
├─────────────────────────────────────┤
│  1. Entity Recognition (LLM)        │
│     → Soruda geçen Person/Project/  │
│       Goal/Concept isimlerini çıkar │
│                                     │
│  2. Vector Search (sqlite-vec)      │
│     → Anlamsal benzerlik ile        │
│       en yakın node'ları bul        │
│                                     │
│  3. Exact Match (Neo4j)             │
│     → Birebir isim eşleşmesi        │
└─────────────────────────────────────┘
    │
    ▼ anchor_ids: ['goal:ai_expert', 'person:emin']
    │
┌─────────────────────────────────────┐
│         EXPAND PHASE                │
│  (Graph yapısını kullanarak         │
│   bağlamı genişlet)                 │
├─────────────────────────────────────┤
│  Schema-aware traversal:            │
│  • CONTRIBUTES_TO (Project→Goal)    │
│  • MOTIVATES (Value→Goal)           │
│  • HAS_GOAL (Person→Goal)           │
│  • BELIEVES (Person→Belief)         │
│  • FEARS (Person→Fear)              │
│  • RELATED_TO (Concept→Concept)     │
└─────────────────────────────────────┘
    │
    ▼ context nodes + relations
    │
┌─────────────────────────────────────┐
│         FORMAT & RETURN             │
│  LLM-readable context               │
└─────────────────────────────────────┘
```

---

## Expansion Stratejileri

Query intent'e göre farklı traversal pattern'ları:

### 1. Hierarchy Expansion (Upstream)
**Kullanım:** "Neden?", "Ne için?", "Amacı ne?" soruları
**Pattern:** `Project → Goal → Value`
**İlişkiler:** `CONTRIBUTES_TO`, `MOTIVATES`

### 2. Context Expansion (Neighborhood)
**Kullanım:** "Ne durumda?", "Neyle ilgili?", "Kim?" soruları
**Pattern:** `Person ↔ Goal/Project/Belief`
**İlişkiler:** `HAS_GOAL`, `WORKS_ON`, `BELIEVES`, `INTERESTED_IN`

### 3. Knowledge Expansion (Lateral)
**Kullanım:** "Bağlantılı ne var?", "İlişkili konular?" soruları
**Pattern:** `Concept ↔ Concept`, `Belief ↔ Belief`
**İlişkiler:** `RELATED_TO`, `SUPPORTS`, `CONTRADICTS`

### 4. Fear/Avoidance Expansion
**Kullanım:** "Korkular?", "Riskler?", "Kaçınılan?" soruları
**Pattern:** `Person → Fear → Goal`
**İlişkiler:** `FEARS`, `AVOIDS`

---

## MCP Tools

### 1. `search_graph`

Ana arama tool'u. Anchor & Expand stratejisini uygular.

| Alan | Değer |
|------|-------|
| **Açıklama** | Kullanıcı sorgusuna göre graph'ta arama yapar. Önce anchor bulur, sonra ilişkiler üzerinden genişletir. |
| **Ne zaman kullan** | Kullanıcı bilgi sorduğunda, bağlam gerektiğinde, "X nedir/kimdir/neden" sorularında |

**Input:**
```typescript
{
  query: string,           // Kullanıcının sorusu
  node_types?: string[],   // Opsiyonel filtre: ["Goal", "Project", "Person"]
  expansion?: string,      // "hierarchy" | "context" | "knowledge" | "fear" | "auto"
  max_depth?: number,      // Traversal derinliği (default: 2)
  limit?: number           // Max sonuç sayısı (default: 10)
}
```

**Output:**
```typescript
{
  anchors: [
    {
      id: string,
      name: string,
      type: string,           // "Goal", "Person", "Concept", etc.
      match_type: string,     // "vector" | "exact" | "entity"
      score: number
    }
  ],
  context: [
    {
      subject: string,        // "Project: RL Roadmap"
      relation: string,       // "CONTRIBUTES_TO"
      object: string,         // "Goal: AI Expert"
      object_type: string,
      depth: number           // Anchor'dan uzaklık
    }
  ],
  summary: string            // LLM-readable özet
}
```

**İç Çalışma:**
1. Query'den entity extraction (LLM call)
2. Paralel: Vector search (sqlite-vec) + Exact match (Neo4j)
3. Anchor'ları birleştir ve dedupe et
4. Expansion stratejisini belirle (auto ise query intent'e göre)
5. Neo4j Cypher ile traversal
6. Sonuçları formatla ve döndür

---

### 2. `get_node`

Tek bir node'un detaylarını getirir.

| Alan | Değer |
|------|-------|
| **Açıklama** | Belirli bir node'un tüm property'lerini ve doğrudan bağlantılarını getirir. |
| **Ne zaman kullan** | Spesifik bir entity hakkında detay gerektiğinde |

**Input:**
```typescript
{
  node_id: string,           // "goal:ai_expert" veya "projects/foo.md"
  include_neighbors?: boolean // Doğrudan komşuları dahil et (default: true)
}
```

**Output:**
```typescript
{
  node: {
    id: string,
    type: string,
    name: string,
    summary?: string,
    properties: Record<string, any>,  // status, priority, confidence, etc.
    created_at?: string
  },
  incoming: [                // Bu node'a gelen ilişkiler
    { from_id: string, from_name: string, from_type: string, relation: string }
  ],
  outgoing: [                // Bu node'dan çıkan ilişkiler
    { to_id: string, to_name: string, to_type: string, relation: string }
  ]
}
```

**İç Çalışma:**
1. Neo4j'den node'u ID ile çek
2. `include_neighbors=true` ise incoming/outgoing edge'leri de çek
3. Property'leri formatla ve döndür

---

### 3. `list_nodes`

Belirli tipteki node'ları listeler.

| Alan | Değer |
|------|-------|
| **Açıklama** | Filtrelere göre node listesi döndürür. Özet görünüm. |
| **Ne zaman kullan** | "Tüm hedeflerim", "Aktif projeler", "Hangi kavramlar var?" |

**Input:**
```typescript
{
  type: string,              // "Goal" | "Project" | "Person" | "Concept" | ...
  filters?: {
    status?: string,         // "active" | "completed" | "paused"
    priority?: string,       // "core" | "important" | "minor"
    horizon?: string         // "life" | "yearly" | "monthly" | "weekly"
  },
  sort_by?: string,          // "name" | "created_at" | "status"
  limit?: number             // Default: 50
}
```

**Output:**
```typescript
{
  count: number,
  nodes: [
    {
      id: string,
      name: string,
      type: string,
      summary?: string,
      status?: string,
      // ... filtered properties
    }
  ]
}
```

**İç Çalışma:**
1. Neo4j'de label + filter ile query
2. Sort ve limit uygula
3. Özet formatında döndür

---

### 4. `traverse_path`

İki node arasındaki yolu bulur.

| Alan | Değer |
|------|-------|
| **Açıklama** | Başlangıç ve bitiş node'u arasındaki en kısa veya en anlamlı yolu bulur. |
| **Ne zaman kullan** | "X ile Y nasıl bağlantılı?", "Bu proje hangi değere hizmet ediyor?" |

**Input:**
```typescript
{
  from_id: string,           // Başlangıç node
  to_id: string,             // Hedef node
  max_depth?: number,        // Max path uzunluğu (default: 4)
  relation_filter?: string[] // Sadece bu ilişkileri kullan
}
```

**Output:**
```typescript
{
  found: boolean,
  path: [
    {
      node_id: string,
      node_name: string,
      node_type: string,
      relation_to_next?: string
    }
  ],
  path_description: string   // "Project X → CONTRIBUTES_TO → Goal Y → MOTIVATES → Value Z"
}
```

**İç Çalışma:**
1. Neo4j `shortestPath` veya `allShortestPaths` kullan
2. Relation filter varsa edge type'lara uygula
3. Path'i human-readable formata çevir

---

### 5. `find_related`

Bir node'a en çok bağlı/ilişkili node'ları bulur.

| Alan | Değer |
|------|-------|
| **Açıklama** | Belirli bir node'dan başlayarak en güçlü bağlantıları keşfeder. |
| **Ne zaman kullan** | "Bununla ilgili ne var?", "En bağlantılı konular?" |

**Input:**
```typescript
{
  node_id: string,
  depth?: number,            // Kaç adım gidilsin (default: 2)
  relation_types?: string[], // Filtre: ["RELATED_TO", "MENTIONS"]
  node_types?: string[],     // Filtre: ["Concept", "Goal"]
  limit?: number             // Default: 10
}
```

**Output:**
```typescript
{
  center: { id: string, name: string, type: string },
  related: [
    {
      id: string,
      name: string,
      type: string,
      relation_path: string,  // "RELATED_TO → MENTIONS"
      connection_strength: number  // Kaç farklı yoldan bağlı
    }
  ]
}
```

**İç Çalışma:**
1. Anchor node'dan başla
2. N-hop traversal yap
3. Her node için "connection strength" hesapla (kaç edge, hangi tipte)
4. Rank ve limit uygula

---

### 6. `semantic_search`

Pure vector search. Graph traversal yok.

| Alan | Değer |
|------|-------|
| **Açıklama** | Sadece anlamsal benzerlik ile arama. Hızlı, basit. |
| **Ne zaman kullan** | Spesifik bir konu/kelime araması, graph context gerekmediğinde |

**Input:**
```typescript
{
  query: string,
  node_types?: string[],     // Sadece bu tiplerde ara
  limit?: number             // Default: 10
}
```

**Output:**
```typescript
{
  results: [
    {
      id: string,
      name: string,
      type: string,
      content_snippet: string,  // İlk 200 karakter
      score: number             // 0-1 arası similarity
    }
  ]
}
```

**İç Çalışma:**
1. Query'yi embed et (sentence-transformers)
2. sqlite-vec'te cosine similarity search
3. Opsiyonel: type filter uygula
4. Score ile sırala ve döndür

---

### 7. `add_note_to_graph`

Yeni bir notu graph'a ekler (parsing + extraction).

| Alan | Değer |
|------|-------|
| **Açıklama** | Markdown notu parse eder, entity'leri extract eder, graph'a ekler. |
| **Ne zaman kullan** | Yeni not oluşturulduğunda veya güncellendiğinde |

**Input:**
```typescript
{
  path: string,              // Vault-relative path
  content: string,           // Markdown içerik
  extract_entities?: boolean // LLM ile entity extraction (default: true)
}
```

**Output:**
```typescript
{
  note_id: string,
  parsed: {
    title: string,
    tags: string[],
    wikilinks: string[],
    frontmatter: Record<string, any>
  },
  extracted?: {              // extract_entities=true ise
    entities: [
      { type: string, name: string, id: string }
    ],
    relations: [
      { from: string, relation: string, to: string }
    ]
  },
  stats: {
    nodes_created: number,
    edges_created: number
  }
}
```

**İç Çalışma:**
1. Markdown parser ile wikilinks, tags, frontmatter çıkar
2. Note node'u oluştur/güncelle
3. `extract_entities=true` ise:
   - LLM'e content gönder
   - Entity'leri (Goal, Project, Belief, vs.) extract et
   - İlişkileri (MENTIONS, BELIEVES, vs.) extract et
4. Graph'a yaz (NetworkX + Neo4j)
5. Vector store'a embedding ekle

---

### 8. `get_graph_stats`

Graph hakkında özet istatistikler.

| Alan | Değer |
|------|-------|
| **Açıklama** | Graph'ın genel durumu, node/edge sayıları, tip dağılımları. |
| **Ne zaman kullan** | Debug, overview, health check |

**Input:**
```typescript
{
  detailed?: boolean         // Tip bazlı breakdown (default: false)
}
```

**Output:**
```typescript
{
  total_nodes: number,
  total_edges: number,
  by_node_type?: Record<string, number>,  // {"Goal": 15, "Project": 8, ...}
  by_edge_type?: Record<string, number>,  // {"CONTRIBUTES_TO": 20, ...}
  last_updated: string
}
```

---

## Tool Seçim Rehberi

| Kullanıcı Niyeti | Önerilen Tool | Neden |
|------------------|---------------|-------|
| "X nedir?" | `search_graph` | Anchor + context gerekli |
| "Hedeflerim neler?" | `list_nodes(type="Goal")` | Liste görünümü yeterli |
| "X ile Y nasıl bağlı?" | `traverse_path` | Direkt path arama |
| "Bu projeyle ilgili ne var?" | `find_related` | Center-out expansion |
| "Kelime içeren notlar?" | `semantic_search` | Graph gerekmez |
| "X'in detayları?" | `get_node` | Tek node + neighbors |

---

## Cypher Query Patterns

### Hierarchy Expansion
```cypher
MATCH (anchor {id: $anchor_id})
MATCH path = (anchor)-[:CONTRIBUTES_TO|MOTIVATES*1..2]->(upper)
RETURN nodes(path), relationships(path)
```

### Context Expansion
```cypher
MATCH (anchor {id: $anchor_id})
MATCH (anchor)-[r:HAS_GOAL|WORKS_ON|BELIEVES|INTERESTED_IN|FEARS]-(context)
RETURN anchor, type(r) as relation, context
```

### Knowledge Expansion
```cypher
MATCH (anchor {id: $anchor_id})
MATCH (anchor)-[r:RELATED_TO|SUPPORTS|CONTRADICTS*1..2]-(related)
WHERE anchor <> related
RETURN DISTINCT related, collect(type(r)) as relations
```

### Shortest Path
```cypher
MATCH path = shortestPath(
  (start {id: $from_id})-[*..4]-(end {id: $to_id})
)
RETURN path
```

---

## Open Questions

- [ ] Entity Recognition için hangi LLM/prompt kullanılacak?
- [ ] Vector search ve graph traversal sonuçları nasıl birleştirilecek (RRF vs. simple merge)?
- [ ] Expansion strategy "auto" modunda intent detection nasıl yapılacak?
- [ ] Rate limiting / caching stratejisi?
