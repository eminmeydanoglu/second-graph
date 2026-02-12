# Planned Changes

## 1. Provenance: `r.source` → `r.source_note` ✅

**Karar:** Edge'lerde "hangi mekanizma oluşturdu" (file/agent/extraction) bilgisi gereksiz. Tek önemli soru: "bu edge hangi nottan çıkarıldı?"

**Yapılanlar:**
- `r.source` alanı kaldırıldı, tüm edge'lerde `r.source_note` kullanılıyor.
- Değer = node_id formatında (ör. `note:mimetik_teorisi_girard`).
- `get_edges_by_source` → `get_edges_by_source_note` olarak güncellendi.
- `delete_edges_by_source` → `delete_edges_by_source_note` olarak güncellendi.
- `SourceType` enum'u ve `generate_source_id` kaldırıldı.
- `_sync_wikilinks` artık `source_note=node_id` kullanıyor.
- Tüm testler güncellendi.

## 2. `_sync_wikilinks` relation-type filtrelemiyor ✅

**Yapılanlar:**
- `get_edges_by_source_note` ve `delete_edges_by_source_note`'a `relation` parametresi eklendi.
- `_sync_wikilinks` artık `relation="WIKILINK"` ile sadece wikilink edge'lerini çekip diff yapıyor.
- Farklı relation type'lar (MENTIONS, RELATED_TO, vb.) artık yanlışlıkla silinmiyor.

## 3. `merge_nodes_simple` edge property'leri düşürüyor ✅

**Yapılanlar:**
- `merge_nodes_simple` artık `confidence`, `fact`, `source_note` gibi edge property'lerini koruyor.
- Sadece `id` ve `created_at` yeniden oluşturuluyor.

## 4. Node/edge uniqueness garantisi yok ✅

**Yapılanlar:**
- `add_node`: `CREATE` → `MERGE` (idempotent upsert). Aynı id ile iki kez çağırılırsa node güncellenir, duplicate oluşmaz. `ON CREATE SET created_at`, `ON MATCH SET` props günceller.
- `add_edge`: `CREATE` → `MERGE` (aynı (from, to, relation_type) tuple'ı için). Duplicate edge oluşmaz, props güncellenir. Edge id ilk oluşturmada set edilir, sonraki çağrılarda korunur.
- `_create_indexes`: `CREATE INDEX ... ON (n.id)` → `CREATE CONSTRAINT ... REQUIRE n.id IS UNIQUE`. Hem index hem uniqueness garantisi.

## 5. Schema validation gevşek ✅

**Yapılanlar:**
- `server.py` `add_edge` MCP tool'u artık `strict=True` ile validate ediyor.
- Bilinmeyen edge type'lar → error (yazma reddedilir).
- Geçersiz source/target type constraint ihlalleri → error (yazma reddedilir).
- Geçerli ama uyarılı durumlar → warnings response'a ekleniyor.
