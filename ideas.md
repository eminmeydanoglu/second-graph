# Ideas & Research Notes

Promising ideas from research papers, systems, and projects that could enhance Obsidian Brain.

---

## Microsoft GraphRAG - Community Detection & Hierarchical Summarization

**Source:** [GraphRAG - Microsoft Research](https://microsoft.github.io/graphrag/)  
**Paper:** [From Local to Global: A Graph RAG Approach](https://arxiv.org/pdf/2404.16130)  
**Status:** 🟢 High Value - Consider Implementation

### Core Concepts

#### 1. Hierarchical Leiden Community Detection

**Problem:** In large knowledge graphs, it's hard to understand the "big picture" or thematic organization.

**Solution:** Apply hierarchical community detection (Leiden algorithm) to automatically discover clusters of related entities/notes.

**How it works:**
- Uses **Leiden algorithm** to partition the graph into communities (better than Louvain)
- **Hierarchical clustering**: Creates multi-level hierarchy (zoom in/out)
- Maximizes **modularity**: groups that are densely connected internally, sparsely connected externally

**Example Output:**
```
Level 0: All nodes
├─ Level 1: "AI Research" community
│   ├─ Level 2: "Reinforcement Learning" sub-community
│   └─ Level 2: "Computer Vision" sub-community
├─ Level 1: "Philosophy" community
└─ Level 1: "Career Planning" community
```

**Value for Obsidian Brain:**
- Auto-discover thematic clusters in vault (AI notes, philosophy notes, daily logs...)
- Navigate knowledge at different granularities
- Detect emerging topics from note connections

**Implementation Complexity:** ⭐⭐ Medium
- Dependency: `graspologic` or `cdlib` library
- Input: Our existing NetworkX graph
- Output: Community hierarchy stored in Neo4j

---

#### 2. LLM-Generated Community Summaries

**Problem:** Even with communities detected, you still need to read many nodes to understand what a cluster represents.

**Solution:** For each community, generate an LLM summary describing:
1. **Title**: What is this community about? (5-10 words)
2. **Summary**: High-level overview (2-3 sentences)
3. **Key Findings**: Important entities and relationships
4. **Impact**: Why this cluster matters

**Example Prompt:**
```
Analyze this cluster from Emin's knowledge graph:

ENTITIES (12):
- RLHF Project.md (Note)
- Reinforcement Learning.md (Note)
- Imitation Learning.md (Concept)
- Career Goals.md (Note)
...

RELATIONSHIPS (8):
- RLHF Project → CONTRIBUTES_TO → Career Goals
- Reinforcement Learning → RELATED_TO → Imitation Learning
...

Generate a comprehensive summary report (JSON format).
```

**Value for Obsidian Brain:**
- **Agent context**: "Show me the AI cluster summary" gives instant overview
- **Discovery**: Find clusters you didn't know existed
- **Provenance**: Each summary cites specific notes/entities

**Implementation Complexity:** ⭐⭐⭐ Medium-High
- LLM cost: One summary per community (~10-50 communities)
- Prompt engineering: Need good template
- Storage: Add `Community` node type to schema

---

#### 3. Global Search (Map-Reduce over Communities)

**Problem:** Traditional vector search (local RAG) struggles with holistic questions:
- "What are my main focus areas?"
- "How do my projects relate to my values?"
- "What patterns emerge from my daily notes?"

**Solution:** Query community summaries instead of individual chunks.

**Process:**
1. **Map**: Embed query, find relevant community summaries
2. **Reduce**: LLM synthesizes answer from multiple summaries
3. **Cite**: Link back to original notes through communities

**Example:**
```python
query = "What is Emin working on right now?"

# Traditional RAG (local):
top_k_notes = vector_search(query, k=5)
# → Returns random chunks, fragmented context

# Global Search (GraphRAG):
relevant_communities = find_communities(query)
# → ["AI Projects" community, "Career Planning" community]

summaries = [c.summary for c in relevant_communities]
answer = llm_summarize(summaries, query)
# → "Emin is focused on RLHF implementation and AUV vision,
#    both aligned with his career goal of autonomous systems..."
```

**Value for Obsidian Brain:**
- Better for **big picture** queries
- More coherent, less hallucination (summaries are grounded)
- Complements local search (use both depending on query type)

**Implementation Complexity:** ⭐⭐ Medium
- Requires communities + summaries to exist
- Add new query mode to MCP server
- Relatively cheap LLM calls (summaries are pre-computed)

---

### Implementation Roadmap

**Phase 1: Community Detection** (~2 hours)
- [ ] Add `graspologic` dependency to `pyproject.toml`
- [ ] Create `src/graph/community.py` with Leiden clustering
- [ ] Integrate with `VaultGraph.build()` pipeline
- [ ] Add `Community` node type to schema

**Phase 2: Community Summarization** (~3 hours)
- [ ] Create prompt template for community summarization
- [ ] Implement `summarize_community()` with OpenAI API
- [ ] Store summaries in Neo4j `Community` nodes
- [ ] CLI command: `obsidian-brain generate-community-summaries`

**Phase 3: Global Search** (~2 hours)
- [ ] Implement `global_search()` query function
- [ ] Add MCP tool: `global_search(query: str)`
- [ ] Test with holistic queries
- [ ] Compare against local vector search

**Total Effort:** ~1 day (7 hours)

---

### Trade-offs

**Pros:**
- ✅ Auto-organization of knowledge (no manual tagging)
- ✅ Multi-granularity navigation (zoom in/out)
- ✅ Better holistic understanding
- ✅ Complements existing local search
- ✅ Community summaries useful for agent context

**Cons:**
- ❌ LLM cost for summarization (one-time per index rebuild)
- ❌ Communities change when graph changes (need re-clustering)
- ❌ Not as precise for specific entity queries (use local search)
- ❌ Adds complexity to codebase

---

### References

- **Paper:** [From Local to Global: A Graph RAG Approach to Question Answering](https://arxiv.org/pdf/2404.16130)
- **GitHub:** [microsoft/graphrag](https://github.com/microsoft/graphrag)
- **Docs:** [GraphRAG Documentation](https://microsoft.github.io/graphrag/)
- **Leiden Algorithm:** [Traag et al. (2019)](https://www.nature.com/articles/s41598-019-41695-z)
- **Original Louvain:** [Blondel et al. (2008)](https://arxiv.org/abs/0803.0476)

---

### Open Questions

- [ ] How often should we re-run community detection? (on every sync? nightly batch?)
- [ ] Should communities be read-only or can agents edit/split them?
- [ ] How to handle overlapping communities? (some nodes belong to multiple)
- [ ] Optimal `max_cluster_size` for personal vault? (10? 20? 50?)
- [ ] Community embeddings: embed summaries for similarity search?

---

### Related Ideas

- **DRIFT Search**: Hybrid local+community search (see GraphRAG docs)
- **Community Evolution Tracking**: Detect when clusters split/merge over time
- **Visual Graph Navigation**: Pyvis/D3 visualization colored by community
- **Agent Community Awareness**: "You're editing a note in the Philosophy cluster"

---

## [Future Sections]

(Other research ideas to be added here...)

