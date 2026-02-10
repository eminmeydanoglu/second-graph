# Graph Manipulator Tools - Implementation Plan

## Mimari Karar: Seçenek D

**VaultGraph sadece initial import için, runtime CRUD direkt Neo4j'e.**

```
┌─────────────────────────────────────────────────────────────┐
│                     INITIAL LOAD (one-time)                 │
│  Vault → parse → VaultGraph → import_vault_graph → Neo4j    │
│                                                             │
│  Bu akış değişmiyor. Mevcut kod aynen kalıyor.              │
└─────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────┐
│                     RUNTIME CRUD                            │
│  MCP Tools → Neo4jStorage.add_node() → Neo4j                │
│              Neo4jStorage.find_node()                       │
│              Neo4jStorage.update_node()                     │
│              Neo4jStorage.delete_node()                     │
│                                                             │
│  VaultGraph'a dokunmuyoruz. Direkt Neo4j.                   │
└─────────────────────────────────────────────────────────────┘
```

**Neden bu yaklaşım?**
- VaultGraph ve Neo4j senkronizasyon problemi yok
- Mevcut vault parsing kodu kırılmıyor
- En az değişiklik gerektiren çözüm
- Graphiti de aynı pattern'ı kullanıyor (direkt DB)

---

## Mevcut Yapı

```
src/
├── cli.py                    # Click CLI
├── parser/
│   ├── markdown.py           # Wikilink, tag, frontmatter parser
│   └── vault.py              # Vault traversal
├── graph/
│   ├── builder.py            # VaultGraph (NetworkX) - SADECE INITIAL IMPORT
│   ├── neo4j_storage.py      # Neo4jStorage - import var, CRUD eklenecek
│   └── visualize.py          # Pyvis visualization
├── vector/
│   ├── embedder.py           # sentence-transformers (384 dim)
│   └── store.py              # VectorStore (sqlite-vec) - entity desteği eklenecek
└── extraction/
    ├── reader.py             # LLM extraction JSON okuma
    └── merger.py             # ExtractionMerger
```

---

## Hedef Yapı

```
src/
├── cli.py                    # + mcp-server komutu
├── parser/...                # Değişiklik yok
├── graph/
│   ├── builder.py            # Değişiklik yok (sadece initial import)
│   ├── neo4j_storage.py      # + CRUD metodları
│   ├── visualize.py          # Değişiklik yok
│   └── schema.py             # YENİ: Node/Edge tipleri, validation
├── vector/
│   ├── embedder.py           # Değişiklik yok
│   └── store.py              # + entity embedding desteği
└── mcp/                      # YENİ
    └── server.py             # Graph Manipulator MCP server
```

**Not:** `service.py` ve `operations.py` layer'larını kaldırdım. Gereksiz soyutlama. 
Neo4jStorage'a direkt metodlar ekleyip MCP server'dan çağıracağız.

---

## Implementation Phases

### Phase 1: Schema Layer
**Dosya:** `src/graph/schema.py`

Node/Edge tiplerini ve validation'ı tanımla.

```python
from enum import Enum
from dataclasses import dataclass

class NodeType(Enum):
    NOTE = "Note"
    TAG = "Tag"
    GOAL = "Goal"
    PROJECT = "Project"
    BELIEF = "Belief"
    VALUE = "Value"
    PERSON = "Person"
    CONCEPT = "Concept"
    SOURCE = "Source"
    FEAR = "Fear"

class EdgeType(Enum):
    WIKILINK = "wikilink"
    TAGGED_WITH = "tagged_with"
    CONTRIBUTES_TO = "CONTRIBUTES_TO"
    WORKS_ON = "WORKS_ON"
    MENTIONS = "MENTIONS"
    BELIEVES = "BELIEVES"
    VALUES = "VALUES"
    SUPPORTS = "SUPPORTS"
    CONTRADICTS = "CONTRADICTS"
    MOTIVATES = "MOTIVATES"
    HAS_GOAL = "HAS_GOAL"
    HAS_VALUE = "HAS_VALUE"
    HAS_BELIEF = "HAS_BELIEF"
    KNOWS = "KNOWS"
    INTERESTED_IN = "INTERESTED_IN"
    LEARNED_FROM = "LEARNED_FROM"
    RELATED_TO = "RELATED_TO"
    FEARS = "FEARS"
    AVOIDS = "AVOIDS"

# Schema constraints: hangi source -> target geçerli
EDGE_CONSTRAINTS = {
    EdgeType.CONTRIBUTES_TO: {
        "sources": [NodeType.PROJECT, NodeType.GOAL],
        "targets": [NodeType.GOAL]
    },
    EdgeType.MOTIVATES: {
        "sources": [NodeType.VALUE],
        "targets": [NodeType.GOAL]
    },
    EdgeType.HAS_GOAL: {
        "sources": [NodeType.PERSON],
        "targets": [NodeType.GOAL]
    },
    # ... diğerleri
}

@dataclass
class ValidationResult:
    valid: bool
    errors: list[str]
    warnings: list[str]

def validate_node_type(node_type: str) -> bool:
    return node_type in [t.value for t in NodeType]

def validate_edge(from_type: str, to_type: str, relation: str) -> ValidationResult:
    # Constraint kontrolü
    pass

def generate_node_id(node_type: str, name: str) -> str:
    """goal:build_autonomous_agents formatında ID üret."""
    normalized = name.lower().replace(" ", "_").replace("-", "_")
    return f"{node_type.lower()}:{normalized}"
```

**Süre:** 1 saat

---

### Phase 2: Neo4j CRUD Operations
**Dosya:** `src/graph/neo4j_storage.py` (genişlet)

Mevcut class'a CRUD metodları ekle.

```python
class Neo4jStorage:
    # ... mevcut metodlar ...

    # === NODE OPERATIONS ===
    
    def add_node(self, node_type: str, node_id: str, name: str, properties: dict = None) -> dict:
        """Yeni node oluştur."""
        props = properties or {}
        props["id"] = node_id
        props["name"] = name
        props["created_at"] = datetime.now().isoformat()
        
        with self.driver.session() as session:
            result = session.run(f"""
                CREATE (n:{node_type} $props)
                RETURN n
            """, props=props)
            record = result.single()
            return dict(record["n"]) if record else None

    def get_node(self, node_id: str) -> dict | None:
        """Node'u ID ile getir."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n {id: $id})
                OPTIONAL MATCH (n)-[r]-(neighbor)
                RETURN n, collect({
                    neighbor: neighbor,
                    relation: type(r),
                    direction: CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END
                }) as connections
            """, id=node_id)
            record = result.single()
            if not record:
                return None
            return {
                "node": dict(record["n"]),
                "connections": record["connections"]
            }

    def find_nodes(self, name: str, node_type: str = None, match_type: str = "contains") -> list[dict]:
        """Node ara. match_type: exact, contains, starts_with"""
        type_filter = f":{node_type}" if node_type else ""
        
        if match_type == "exact":
            where = "WHERE n.name = $name"
        elif match_type == "starts_with":
            where = "WHERE n.name STARTS WITH $name"
        else:  # contains
            where = "WHERE toLower(n.name) CONTAINS toLower($name)"
        
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (n{type_filter})
                {where}
                RETURN n
                LIMIT 20
            """, name=name)
            return [dict(r["n"]) for r in result]

    def update_node(self, node_id: str, properties: dict) -> dict | None:
        """Node property'lerini güncelle."""
        properties["updated_at"] = datetime.now().isoformat()
        
        with self.driver.session() as session:
            result = session.run("""
                MATCH (n {id: $id})
                SET n += $props
                RETURN n
            """, id=node_id, props=properties)
            record = result.single()
            return dict(record["n"]) if record else None

    def delete_node(self, node_id: str) -> dict:
        """Node ve bağlı edge'leri sil."""
        with self.driver.session() as session:
            # Önce bağlantı sayısını al
            count_result = session.run("""
                MATCH (n {id: $id})-[r]-()
                RETURN count(r) as edge_count
            """, id=node_id)
            edge_count = count_result.single()["edge_count"]
            
            # Sonra sil
            session.run("""
                MATCH (n {id: $id})
                DETACH DELETE n
            """, id=node_id)
            
            return {"deleted": True, "edges_removed": edge_count}

    def merge_nodes(self, keep_id: str, merge_id: str) -> dict:
        """İki node'u birleştir. merge_id'nin edge'lerini keep_id'ye taşı, merge_id'yi sil."""
        with self.driver.session() as session:
            # Edge'leri taşı
            result = session.run("""
                MATCH (keep {id: $keep_id})
                MATCH (merge {id: $merge_id})
                
                // Incoming edge'leri taşı
                OPTIONAL MATCH (source)-[r_in]->(merge)
                WHERE source <> keep
                FOREACH (_ IN CASE WHEN r_in IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (source)-[new_r:MERGED_FROM]->(keep)
                )
                
                // Outgoing edge'leri taşı  
                OPTIONAL MATCH (merge)-[r_out]->(target)
                WHERE target <> keep
                FOREACH (_ IN CASE WHEN r_out IS NOT NULL THEN [1] ELSE [] END |
                    CREATE (keep)-[new_r:MERGED_TO]->(target)
                )
                
                WITH merge, count(*) as transferred
                DETACH DELETE merge
                RETURN transferred
            """, keep_id=keep_id, merge_id=merge_id)
            record = result.single()
            return {"merged": True, "edges_transferred": record["transferred"] if record else 0}

    # === EDGE OPERATIONS ===
    
    def add_edge(self, from_id: str, to_id: str, relation: str, properties: dict = None) -> dict:
        """İki node arasında edge oluştur."""
        props = properties or {}
        edge_id = f"edge:{uuid4().hex[:12]}"
        props["id"] = edge_id
        props["created_at"] = datetime.now().isoformat()
        
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (a {{id: $from_id}})
                MATCH (b {{id: $to_id}})
                CREATE (a)-[r:{relation} $props]->(b)
                RETURN r, a.name as from_name, b.name as to_name
            """, from_id=from_id, to_id=to_id, props=props)
            record = result.single()
            if not record:
                return {"success": False, "error": "Nodes not found"}
            return {
                "success": True,
                "edge_id": edge_id,
                "from": from_id,
                "to": to_id,
                "relation": relation
            }

    def get_edge(self, edge_id: str) -> dict | None:
        """Edge'i ID ile getir."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH (a)-[r {id: $id}]->(b)
                RETURN r, a.id as from_id, a.name as from_name, 
                       b.id as to_id, b.name as to_name, type(r) as relation
            """, id=edge_id)
            record = result.single()
            if not record:
                return None
            return {
                "edge": dict(record["r"]),
                "from_id": record["from_id"],
                "from_name": record["from_name"],
                "to_id": record["to_id"],
                "to_name": record["to_name"],
                "relation": record["relation"]
            }

    def delete_edge(self, edge_id: str) -> bool:
        """Edge'i sil."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r {id: $id}]->()
                DELETE r
                RETURN count(r) as deleted
            """, id=edge_id)
            return result.single()["deleted"] > 0

    def invalidate_edge(self, edge_id: str, reason: str = None) -> dict | None:
        """Edge'i soft-delete (invalid_at set et)."""
        with self.driver.session() as session:
            result = session.run("""
                MATCH ()-[r {id: $id}]->()
                SET r.invalid_at = datetime(), r.invalidation_reason = $reason
                RETURN r
            """, id=edge_id, reason=reason)
            record = result.single()
            return dict(record["r"]) if record else None

    # === QUERY OPERATIONS ===
    
    def get_neighbors(self, node_id: str, direction: str = "both", 
                      edge_types: list[str] = None, depth: int = 1) -> list[dict]:
        """Node'un komşularını getir."""
        type_filter = ""
        if edge_types:
            type_filter = ":" + "|".join(edge_types)
        
        if direction == "out":
            pattern = f"(n)-[r{type_filter}]->(neighbor)"
        elif direction == "in":
            pattern = f"(n)<-[r{type_filter}]-(neighbor)"
        else:
            pattern = f"(n)-[r{type_filter}]-(neighbor)"
        
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (n {{id: $id}})
                MATCH {pattern}
                RETURN neighbor, type(r) as relation, 
                       CASE WHEN startNode(r) = n THEN 'out' ELSE 'in' END as direction
            """, id=node_id)
            return [{"node": dict(r["neighbor"]), "relation": r["relation"], 
                     "direction": r["direction"]} for r in result]

    def find_path(self, from_id: str, to_id: str, max_depth: int = 4) -> dict | None:
        """İki node arası en kısa yolu bul."""
        with self.driver.session() as session:
            result = session.run(f"""
                MATCH (start {{id: $from_id}}), (end {{id: $to_id}})
                MATCH path = shortestPath((start)-[*..{max_depth}]-(end))
                RETURN [n IN nodes(path) | n.id] as node_ids,
                       [n IN nodes(path) | n.name] as node_names,
                       [r IN relationships(path) | type(r)] as relations
            """, from_id=from_id, to_id=to_id)
            record = result.single()
            if not record:
                return None
            return {
                "node_ids": record["node_ids"],
                "node_names": record["node_names"],
                "relations": record["relations"]
            }
```

**Süre:** 2-3 saat

---

### Phase 3: Vector Store Entity Support
**Dosya:** `src/vector/store.py` (genişlet)

Entity embedding desteği ekle.

```python
class VectorStore:
    def _init_db(self):
        # ... mevcut notes tablosu ...
        
        # YENİ: entities tablosu
        conn.execute("""
            CREATE TABLE IF NOT EXISTS entities (
                id INTEGER PRIMARY KEY,
                node_id TEXT UNIQUE NOT NULL,
                node_type TEXT NOT NULL,
                name TEXT NOT NULL,
                summary TEXT
            )
        """)
        
        conn.execute(f"""
            CREATE VIRTUAL TABLE IF NOT EXISTS entity_vectors USING vec0(
                embedding float[{self.dimensions}]
            )
        """)

    def add_entity(self, node_id: str, node_type: str, name: str, 
                   summary: str, embedding: list[float]) -> int:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        cursor.execute(
            "INSERT OR REPLACE INTO entities (node_id, node_type, name, summary) VALUES (?, ?, ?, ?)",
            (node_id, node_type, name, summary)
        )
        row_id = cursor.lastrowid
        
        cursor.execute(
            "INSERT OR REPLACE INTO entity_vectors (rowid, embedding) VALUES (?, ?)",
            (row_id, serialize_vector(embedding))
        )
        
        conn.commit()
        conn.close()
        return row_id

    def search_entities(self, query_embedding: list[float], 
                        node_types: list[str] = None, limit: int = 10) -> list[dict]:
        conn = self._get_conn()
        cursor = conn.cursor()
        
        type_filter = ""
        if node_types:
            placeholders = ",".join("?" * len(node_types))
            type_filter = f"AND e.node_type IN ({placeholders})"
        
        cursor.execute(f"""
            SELECT 
                e.node_id, e.node_type, e.name, e.summary, v.distance
            FROM entity_vectors v
            JOIN entities e ON e.id = v.rowid
            WHERE v.embedding MATCH ? AND k = ?
            {type_filter}
            ORDER BY v.distance
        """, (serialize_vector(query_embedding), limit, *(node_types or [])))
        
        results = []
        for row in cursor.fetchall():
            results.append({
                "node_id": row[0],
                "node_type": row[1],
                "name": row[2],
                "summary": row[3],
                "distance": row[4],
                "score": 1 - row[4]
            })
        
        conn.close()
        return results

    def delete_entity(self, node_id: str) -> bool:
        conn = self._get_conn()
        cursor = conn.cursor()
        cursor.execute("SELECT id FROM entities WHERE node_id = ?", (node_id,))
        row = cursor.fetchone()
        if row:
            cursor.execute("DELETE FROM entities WHERE id = ?", (row[0],))
            cursor.execute("DELETE FROM entity_vectors WHERE rowid = ?", (row[0],))
            conn.commit()
            conn.close()
            return True
        conn.close()
        return False
```

**Süre:** 1 saat

---

### Phase 4: MCP Server
**Dosya:** `src/mcp/server.py` (YENİ)

Graph Manipulator tool'larını expose et.

```python
"""Graph Manipulator MCP Server.

Bu server Graph Agent (subagent) tarafından kullanılır.
Ana ajan (Rafiq) "hatırla/unut" dediğinde, Graph Agent bu tool'ları çağırır.
"""

from datetime import datetime
from mcp.server.fastmcp import FastMCP

from ..graph.neo4j_storage import Neo4jStorage
from ..graph.schema import validate_node_type, validate_edge, generate_node_id
from ..vector.store import VectorStore
from ..vector.embedder import Embedder

mcp = FastMCP("Graph Manipulator", instructions="""
Graph üzerinde CRUD operasyonları yapan tool'lar.
Schema: Note, Tag, Goal, Project, Belief, Value, Person, Concept, Source, Fear
İlişkiler: CONTRIBUTES_TO, MOTIVATES, HAS_GOAL, BELIEVES, INTERESTED_IN, RELATED_TO, vb.
""")

# Global instances (init_server'da set edilecek)
storage: Neo4jStorage = None
vectors: VectorStore = None
embedder: Embedder = None

def init_server(neo4j_uri: str, neo4j_user: str, neo4j_password: str, vector_db: str):
    global storage, vectors, embedder
    storage = Neo4jStorage(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    vectors = VectorStore(vector_db)
    embedder = Embedder()

# === NODE TOOLS ===

@mcp.tool()
def add_node(node_type: str, name: str, summary: str = None, 
             properties: dict = None) -> dict:
    """Graph'a yeni node ekle.
    
    Args:
        node_type: Node tipi (Goal, Project, Person, Concept, vb.)
        name: Node adı
        summary: Açıklama (embedding için kullanılır)
        properties: Ek özellikler (status, priority, vb.)
    """
    if not validate_node_type(node_type):
        return {"success": False, "error": f"Invalid node type: {node_type}"}
    
    node_id = generate_node_id(node_type, name)
    props = properties or {}
    if summary:
        props["summary"] = summary
    
    # Neo4j'e yaz
    result = storage.add_node(node_type, node_id, name, props)
    
    # Summary varsa embedding oluştur
    if summary and result:
        emb = embedder.embed(f"{name}: {summary}")
        vectors.add_entity(node_id, node_type, name, summary, emb)
    
    return {"success": True, "node_id": node_id, "node": result}

@mcp.tool()
def get_node(node_id: str) -> dict:
    """Node detaylarını getir."""
    result = storage.get_node(node_id)
    if not result:
        return {"success": False, "error": "Node not found"}
    return {"success": True, **result}

@mcp.tool()
def find_node(name: str, node_type: str = None, match_type: str = "contains") -> dict:
    """Node ara.
    
    Args:
        name: Aranacak isim
        node_type: Opsiyonel tip filtresi
        match_type: exact, contains, starts_with
    """
    results = storage.find_nodes(name, node_type, match_type)
    return {"success": True, "count": len(results), "nodes": results}

@mcp.tool()
def update_node(node_id: str, properties: dict) -> dict:
    """Node property'lerini güncelle."""
    result = storage.update_node(node_id, properties)
    if not result:
        return {"success": False, "error": "Node not found"}
    
    # Summary değiştiyse embedding güncelle
    if "summary" in properties:
        node = storage.get_node(node_id)
        if node:
            name = node["node"].get("name", "")
            summary = properties["summary"]
            emb = embedder.embed(f"{name}: {summary}")
            vectors.add_entity(node_id, node["node"].get("type", ""), name, summary, emb)
    
    return {"success": True, "node": result}

@mcp.tool()
def delete_node(node_id: str) -> dict:
    """Node ve bağlı edge'leri sil."""
    result = storage.delete_node(node_id)
    vectors.delete_entity(node_id)
    return {"success": True, **result}

@mcp.tool()
def merge_nodes(keep_id: str, merge_id: str) -> dict:
    """İki node'u birleştir."""
    result = storage.merge_nodes(keep_id, merge_id)
    vectors.delete_entity(merge_id)
    return {"success": True, **result}

# === EDGE TOOLS ===

@mcp.tool()
def add_edge(from_id: str, to_id: str, relation: str, 
             properties: dict = None) -> dict:
    """İki node arasında ilişki oluştur.
    
    Args:
        from_id: Kaynak node ID
        to_id: Hedef node ID
        relation: İlişki tipi (CONTRIBUTES_TO, MOTIVATES, vb.)
        properties: Ek özellikler (confidence, fact, vb.)
    """
    return storage.add_edge(from_id, to_id, relation, properties)

@mcp.tool()
def get_edge(edge_id: str) -> dict:
    """Edge detaylarını getir."""
    result = storage.get_edge(edge_id)
    if not result:
        return {"success": False, "error": "Edge not found"}
    return {"success": True, **result}

@mcp.tool()
def delete_edge(edge_id: str) -> dict:
    """Edge'i sil."""
    success = storage.delete_edge(edge_id)
    return {"success": success}

@mcp.tool()
def invalidate_edge(edge_id: str, reason: str = None) -> dict:
    """Edge'i geçersiz işaretle (soft delete)."""
    result = storage.invalidate_edge(edge_id, reason)
    if not result:
        return {"success": False, "error": "Edge not found"}
    return {"success": True, "edge": result}

# === QUERY TOOLS ===

@mcp.tool()
def get_neighbors(node_id: str, direction: str = "both", 
                  edge_types: list[str] = None) -> dict:
    """Node'un komşularını getir."""
    results = storage.get_neighbors(node_id, direction, edge_types)
    return {"success": True, "count": len(results), "neighbors": results}

@mcp.tool()
def find_path(from_id: str, to_id: str, max_depth: int = 4) -> dict:
    """İki node arası yolu bul."""
    result = storage.find_path(from_id, to_id, max_depth)
    if not result:
        return {"success": False, "error": "No path found"}
    return {"success": True, **result}

@mcp.tool()
def search_entities(query: str, node_types: list[str] = None, limit: int = 10) -> dict:
    """Semantic search ile entity ara."""
    query_emb = embedder.embed(query)
    results = vectors.search_entities(query_emb, node_types, limit)
    return {"success": True, "count": len(results), "entities": results}

@mcp.tool()
def get_schema() -> dict:
    """Kullanılabilir node ve edge tiplerini getir."""
    from ..graph.schema import NodeType, EdgeType
    return {
        "node_types": [t.value for t in NodeType],
        "edge_types": [t.value for t in EdgeType]
    }

# Entry point
def main():
    import asyncio
    asyncio.run(mcp.run_stdio_async())

if __name__ == "__main__":
    main()
```

**Dosya:** `src/mcp/__init__.py`
```python
from .server import mcp, init_server, main
```

**Süre:** 2 saat

---

### Phase 5: CLI Integration
**Dosya:** `src/cli.py` (ekleme)

```python
@cli.command("mcp-server")
@click.option("--neo4j-uri", default="bolt://localhost:7687")
@click.option("--neo4j-user", default="neo4j")
@click.option("--neo4j-password", default="obsidian")
@click.option("--vector-db", type=click.Path(path_type=Path), default="data/vectors.db")
@click.option("--transport", type=click.Choice(["stdio", "sse"]), default="stdio")
def mcp_server(neo4j_uri, neo4j_user, neo4j_password, vector_db, transport):
    """Run Graph Manipulator MCP server."""
    from .mcp import init_server, mcp
    
    click.echo(f"Connecting to Neo4j: {neo4j_uri}")
    click.echo(f"Vector DB: {vector_db}")
    
    init_server(neo4j_uri, neo4j_user, neo4j_password, str(vector_db))
    
    click.echo(f"Starting MCP server ({transport})...")
    
    import asyncio
    if transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    else:
        asyncio.run(mcp.run_sse_async())
```

**Süre:** 30 dakika

---

### Phase 6: Tests
**Dosya:** `tests/test_neo4j_crud.py`

```python
import pytest
from src.graph.neo4j_storage import Neo4jStorage
from src.graph.schema import generate_node_id, validate_node_type

class TestSchema:
    def test_generate_node_id(self):
        assert generate_node_id("Goal", "Build AI") == "goal:build_ai"
        
    def test_validate_node_type(self):
        assert validate_node_type("Goal") == True
        assert validate_node_type("FakeType") == False

class TestNeo4jCRUD:
    @pytest.fixture
    def storage(self):
        s = Neo4jStorage(uri="bolt://localhost:7687", user="neo4j", password="obsidian")
        yield s
        s.close()
    
    def test_add_and_get_node(self, storage):
        node = storage.add_node("Goal", "goal:test_goal", "Test Goal", {"status": "active"})
        assert node is not None
        assert node["name"] == "Test Goal"
        
        fetched = storage.get_node("goal:test_goal")
        assert fetched is not None
        assert fetched["node"]["name"] == "Test Goal"
        
        # Cleanup
        storage.delete_node("goal:test_goal")
    
    def test_add_and_delete_edge(self, storage):
        # Setup
        storage.add_node("Project", "project:test", "Test Project", {})
        storage.add_node("Goal", "goal:test", "Test Goal", {})
        
        # Add edge
        result = storage.add_edge("project:test", "goal:test", "CONTRIBUTES_TO")
        assert result["success"] == True
        
        # Delete edge
        deleted = storage.delete_edge(result["edge_id"])
        assert deleted == True
        
        # Cleanup
        storage.delete_node("project:test")
        storage.delete_node("goal:test")
```

**Süre:** 1 saat

---

## Özet

| Phase | Dosya | Süre |
|-------|-------|------|
| 1 | `src/graph/schema.py` | 1 saat |
| 2 | `src/graph/neo4j_storage.py` | 2-3 saat |
| 3 | `src/vector/store.py` | 1 saat |
| 4 | `src/mcp/server.py` | 2 saat |
| 5 | `src/cli.py` | 30 dk |
| 6 | `tests/test_neo4j_crud.py` | 1 saat |

**Toplam:** ~8-10 saat

---

## Dependency Eklentileri (pyproject.toml)

```toml
dependencies = [
    # ... mevcut ...
    "mcp>=1.0.0",
    "pydantic>=2.0.0",
]
```

---

## Başlangıç Sırası

1. **Phase 1** - schema.py (diğerleri buna bağlı)
2. **Phase 2** - neo4j_storage.py (core CRUD)
3. **Phase 3** - store.py (entity embedding)
4. **Phase 4** - mcp/server.py (tool'ları expose et)
5. **Phase 5** - cli.py (entry point)
6. **Phase 6** - tests (doğrulama)
