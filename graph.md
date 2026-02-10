# Graph Schema

Obsidian Brain knowledge graph tanımı.

## Node Types

| Type | Description | Source | ID Format |
|------|-------------|--------|-----------|
| `Note` | Obsidian markdown dosyası | Parsing | `{relative_path}` |
| `Tag` | Obsidian tag'i | Parsing | `tag:{name}` |
| `Goal` | Kullanıcının hedefi (hiyerarşik olabilir) | LLM Extraction | `goal:{normalized_name}` |
| `Project` | Bir hedefe hizmet eden organize çalışma | LLM Extraction | `project:{normalized_name}` |
| `Belief` | Kullanıcının dünya hakkındaki inancı ("X doğru") | LLM Extraction | `belief:{normalized_name}` |
| `Value` | Kullanıcının değer yargısı ("X önemli") | LLM Extraction | `value:{normalized_name}` |
| `Person` | Kullanıcı veya tanıdığı kişiler | LLM Extraction | `person:{normalized_name}` |
| `Concept` | Genel kavram/bilgi (RL, Stoicism, Graph Theory) | LLM Extraction | `concept:{normalized_name}` |
| `Source` | Bilgi kaynağı (kitap, makale, URL) | LLM Extraction | `source:{normalized_name}` |
| `Fear` | Kullanıcının korkusu/endişesi (negatif motivasyon) | LLM Extraction | `fear:{normalized_name}` |

## Node Properties

### Note
| Property | Type | Description |
|----------|------|-------------|
| `title` | string | Not başlığı |
| `folder` | string | Parent klasör adı |

### Tag
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Tag adı (# olmadan) |

### Goal
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Hedef adı |
| `summary` | string? | LLM üretir: Hedefin kısa açıklaması ve bağlamı |
| `status` | enum | `active` / `achieved` / `abandoned` |
| `horizon` | enum | `life` / `yearly` / `monthly` / `weekly` |

### Project
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Proje adı |
| `summary` | string? | LLM üretir: Projenin kısa açıklaması ve amacı |
| `status` | enum | `active` / `completed` / `paused` / `abandoned` |

### Belief
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | İnanç ifadesi |
| `summary` | string? | LLM üretir: İnancın bağlamı ve destekleyici bilgi |
| `confidence` | float (0-1) | Kullanıcının bu inanca ne kadar güvendiği |

### Value
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Değer adı |
| `priority` | enum | `core` / `important` / `minor` |

### Person
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Kişi adı |
| `summary` | string? | LLM üretir: Kişinin kim olduğu, ne yaptığı |
| `is_self` | boolean | Bu kullanıcının kendisi mi? |
| `relationship` | enum | `self` / `family` / `friend` / `colleague` / `mentor` / `acquaintance` |

### Concept
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Kavram adı |
| `summary` | string? | LLM üretir: Kavramın açıklaması ve bağlamı (embedding için kritik) |
| `domain` | string | Alan (AI, Philosophy, Health, vb.) |

### Source
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Kaynak adı |
| `summary` | string? | LLM üretir: Kaynağın içeriği ve önemi |
| `type` | enum | `book` / `article` / `url` / `person` |
| `url` | string? | Varsa URL |
| `author` | string? | Varsa yazar |

### Fear
| Property | Type | Description |
|----------|------|-------------|
| `name` | string | Korku/endişe ifadesi |
| `intensity` | enum | `major` / `minor` |

## Relation Types

| Type | Source → Target | Description | Source |
|------|-----------------|-------------|--------|
| `wikilink` | Note → Note | Explicit `[[link]]` | Parsing |
| `tagged_with` | Note → Tag | Not tag'e sahip | Parsing |
| `CONTRIBUTES_TO` | Project/Goal → Goal | Proje veya alt-goal üst hedefe hizmet eder | LLM Extraction |
| `WORKS_ON` | Note/Person → Project | Not veya kişi bu projeyle ilgili | LLM Extraction |
| `MENTIONS` | Note → any entity | Not bu entity'den bahsediyor | LLM Extraction |
| `BELIEVES` | Note → Belief | Not bu inancı ifade ediyor | LLM Extraction |
| `VALUES` | Note → Value | Not bu değeri yansıtıyor | LLM Extraction |
| `SUPPORTS` | Belief → Belief | Bir inanç diğerini destekler | LLM Extraction |
| `CONTRADICTS` | Belief → Belief | İki inanç çelişiyor | LLM Extraction |
| `MOTIVATES` | Value → Goal | Değer bu hedefe yön veriyor | LLM Extraction |
| `HAS_GOAL` | Person → Goal | Kişinin hedefi | LLM Extraction |
| `HAS_VALUE` | Person → Value | Kişinin değeri | LLM Extraction |
| `HAS_BELIEF` | Person → Belief | Kişinin inancı | LLM Extraction |
| `KNOWS` | Person → Person | Kişiler arası tanışıklık | LLM Extraction |
| `INTERESTED_IN` | Person → Concept | Kişinin ilgi alanı | LLM Extraction |
| `LEARNED_FROM` | Concept/Belief → Source/Note | Bilginin kaynağı (dış kaynak veya not) | LLM Extraction |
| `RELATED_TO` | Concept → Concept | Kavramlar arası ilişki | LLM Extraction |
| `FEARS` | Person → Fear | Kişinin korkusu/endişesi | LLM Extraction |
| `AVOIDS` | Fear → Goal | Korku bu hedeften kaçınmaya yol açıyor | LLM Extraction |

## Edge Properties

| Property | Type | Description | Applies To |
|----------|------|-------------|------------|
| `confidence` | float (0-1) | İlişki güvenilirliği | All |
| `source` | string | Veri kaynağı: `parsing` / `extracted` | All |
| `created_at` | datetime | Edge sisteme ne zaman eklendi | All |
| `valid_at` | datetime? | Gerçek dünyada ne zaman başladı | All |
| `invalid_at` | datetime? | Gerçek dünyada ne zaman bitti (null = hala geçerli) | All |

## ID Format Conventions

- Note: Vault-relative path (`projects/foo.md`)
- Tag: `tag:` prefix + tag name (`tag:ai/ml`)
- Entity: `entity:{type}:{normalized_name}` (TBD)

## Schema Constraints

| Source Type | Valid Targets |
|-------------|---------------|
| `Note` | Note, Tag, Project, Goal, Belief, Value, Concept, Person, Source, Fear |
| `Person` | Goal, Value, Belief, Fear, Concept, Person, Project |
| `Project` | Goal |
| `Goal` | Goal |
| `Value` | Goal |
| `Fear` | Goal |
| `Belief` | Belief, Source, Note |
| `Concept` | Concept, Source, Note |

## Data Sources

| Source | Phase | Produces |
|--------|-------|----------|
| Vault Parsing | Phase 1-2 | Note, Tag, wikilink, tagged_with |
| LLM Extraction | Phase 3 | Entity nodes, semantic relations, mentions |

## Open Questions

- [ ] Privacy/sensitivity tagging gerekli mi?
- [ ] Confidence threshold nedir? (düşük confidence edge'ler retrieval'da nasıl ele alınır)
- [ ] Entity deduplication stratejisi (aynı concept farklı isimlerle gelirse)
