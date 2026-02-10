# Source-Aware Reconciliation Implementation Plan

**Hedef:** Notlar (`.md`) değiştiğinde Graph veritabanını akıllıca güncellemek. Sadece dosyanın "sahip olduğu" verileri değiştirmek, Ajan veya LLM tarafından eklenen diğer verileri korumak.

---

## 1. Kavramsal Model: "Provenance" (Köken)

Graph üzerindeki her veri parçası (Node, Edge, Property) bir **Source (Kaynak)** etiketine sahip olmalıdır.

**Source ID Formatı:** `{type}:{identifier}`

| Tip | Örnek | Açıklama |
| :--- | :--- | :--- |
| **File** | `file:/vault/Projects/AI.md` | Fiziksel dosyadan gelen yapısal veriler (Wikilink, Tag, YAML). |
| **Agent** | `agent:rafiq` | Ajanın konuşma sırasında eklediği manuel hatıralar. |
| **Extraction** | `extraction:v1:/vault/Projects/AI.md` | LLM tarafından metin analizi ile çıkarılan semantik ilişkiler. |
| **System** | `system:init` | İlk kurulum veya sistem tarafından oluşturulan veriler. |

**Kural:** Bir `Reconciler` çalıştığında, **sadece hedeflediği Source ID'ye sahip verileri** değiştirme (silme/güncelleme) yetkisine sahiptir.

---

## 2. Gap Analysis (Eksikler)

| Bileşen | Mevcut Durum | Gereksinim |
| :--- | :--- | :--- |
| **Schema** | `source` alanı opsiyonel/tanımsız. | `source` alanı standartlaştırılmalı ve edge creation sırasında zorunlu/varsayılan olmalı. |
| **Storage** | CRUD var, filtreleme zayıf. | `delete_edges(node_id, filter={source: ...})` gibi yetenekler lazım. |
| **Sync Logic** | `NoteSynchronizer` (basit diff). | `source` parametresi alan ve diff işlemini buna göre yapan `SourceAwareSynchronizer`. |
| **MCP** | `sync_note` (basit). | `reconcile_note(path)` ve ileride `reconcile_extraction(path)`. |

---

## 3. Uygulama Adımları

### Phase 1: Schema & Storage (Altyapı)

**Dosyalar:** `src/graph/schema.py`, `src/graph/neo4j_storage.py`

1.  **Schema Tanımı:**
    *   `SourceType` enum (FILE, AGENT, EXTRACTION, SYSTEM).
    *   Helper: `generate_source_id(type, id)`.

2.  **Neo4j Storage Yetenekleri:**
    *   `add_edge`: `source` parametresi ekle (metadata içine).
    *   `get_edges_by_source(node_id, source)`: Belirli kaynaktan gelenleri listele.
    *   `delete_edges_by_source(node_id, source)`: Belirli kaynaktan gelenleri toplu sil (transactional).

### Phase 2: Source-Aware Synchronizer (Mantık)

**Dosyalar:** `src/graph/sync.py`

1.  `NoteSynchronizer` sınıfını güncelle:
    *   `__init__` veya metodlara `source_id` yeteneği kazandır.
    *   **Logic:**
        *   Mevcut veriyi çekerken `WHERE r.source = $source` filtresi kullan.
        *   Yeni eklenen edge'lere `source=$source` özelliğini bas.
        *   Silinenleri (sadece bu source'a ait olanları) sil.

### Phase 3: MCP & Agent Skills (Dışa Açılım)

**Dosyalar:** `src/mcp/server.py`

1.  `sync_note(path)` tool'unu güncelle:
    *   Otomatik olarak `file:{path}` source ID'sini kullansın.
    *   Dosyadaki `wikilink` ve `tag`'leri bu source ile graph'e yazsın.

2.  *(Future)* `reconcile_extraction(path, extracted_data)`:
    *   LLM'den gelen yapısal veriyi `extraction:{path}` source'u ile graph'e yazar.

### Phase 4: Tests

**Dosyalar:** `src/graph/tests/test_sync_source.py`

1.  Senaryo:
    *   `Note A` -> `Note B` (Source: File)
    *   `Note A` -> `Note C` (Source: Agent - "Rafiq hatırladı")
    *   **Action:** `Note A` güncellenir, linkler silinir.
    *   **Expectation:** `Note B` edge'i silinmeli (File source). `Note C` edge'i KALMALI (Agent source).

---

## 4. Zaman Planı

1.  **Phase 1 (Storage):** 1 saat
2.  **Phase 2 (Logic):** 1-2 saat
3.  **Phase 3 (MCP):** 30 dk
4.  **Phase 4 (Tests):** 1 saat

**Toplam:** ~4 saatlik efor.
