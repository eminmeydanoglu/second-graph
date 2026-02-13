# Task Backlog

## Durum

- [ ] Vector store integration'i tamamla: Neo4j ve vector store consistency sağla (özellikle `search_entities` dönen node'ların Neo4j'de varlığı garanti edilsin).
- [ ] Neo4j veri kaybı RCA: `docker stop/start` sonrası DB sıfırlanma kök nedenini yeniden üret, kanıtla, düzelt.
- [ ] Neo4j runtime hardening: named volume + pinned image + deterministic startup (compose) + healthcheck.
- [x] DB koruma önlemleri (ilk adım): destructive clear guard (`force=True`) + CLI explicit confirm (`type CLEAR`).
- [x] Test izolasyonu: integration testler ana DB yerine ayrı Neo4j instance (default `bolt://localhost:17687`) kullanıyor.
- [x] Graph source control v1: snapshot export + git commit + diff + restore workflow tasarla ve implemente et.
- [ ] Graph source control v2: operation log (event sourcing light) ile ince-grained değişiklik geçmişi.

## Graph Source Control v1 (Tamamlananlar)

- `graph-snapshot`: canlı Neo4j state'ini snapshot olarak kaydeder.
- `graph-log`: snapshot geçmişini listeler.
- `graph-diff <old> <new>`: iki snapshot arasındaki node/edge farkını verir.
- `graph-restore <ref>`: snapshot'tan geri yükler (`--apply` zorunlu, default dry-run).
- Opsiyonel `--git-commit`: snapshot dosyalarını git'e commit eder.

## Sonraki Plan (v2)

1. MCP write operasyonlarına event log ekle (`add_node`, `add_edge`, `update_node`, ...).
2. `graph-events` storage tasarla (append-only).
3. `graph-log --node <id>` ile node bazlı geçmiş sorgusu ekle.
4. `graph-revert <event_id>` için ters işlem (inverse op) akışı ekle.
