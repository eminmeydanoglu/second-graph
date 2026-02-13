"""Tests for Neo4j CRUD operations."""

import pytest

from src.graph.neo4j_storage import Neo4jStorage


class TestNeo4jCRUD:
    """Tests for Neo4j CRUD operations.

    These tests require a running Neo4j instance.
    Skip if Neo4j is not available.
    """

    @pytest.fixture
    def storage(self):
        """Create a Neo4j storage instance for testing."""
        try:
            s = Neo4jStorage(
                uri="bolt://localhost:7687", user="neo4j", password="obsidian"
            )
            # Test connection
            s.get_stats()
            yield s
            s.close()
        except Exception as e:
            pytest.skip(f"Neo4j not available: {e}")

    def test_add_and_get_node(self, storage):
        """Test adding and retrieving a node."""
        # Add node
        node = storage.add_node(
            "Goal",
            "goal:test_crud_goal",
            "Test CRUD Goal",
            {"status": "active", "priority": "high"},
        )
        assert node is not None
        assert node["name"] == "Test CRUD Goal"
        assert node["status"] == "active"

        # Get node
        fetched = storage.get_node("goal:test_crud_goal")
        assert fetched is not None
        assert fetched["node"]["name"] == "Test CRUD Goal"
        assert fetched["node"]["status"] == "active"

        # Cleanup
        storage.delete_node("goal:test_crud_goal")

    def test_find_nodes(self, storage):
        """Test finding nodes by name."""
        # Setup
        storage.add_node("Goal", "goal:find_test_1", "Find Me First", {})
        storage.add_node("Goal", "goal:find_test_2", "Find Me Second", {})
        storage.add_node("Project", "project:find_test", "Find Me Project", {})

        try:
            # Find by contains
            results = storage.find_nodes("Find Me", match_type="contains")
            assert len(results) >= 3

            # Find by type filter
            results = storage.find_nodes("Find Me", node_type="Goal")
            assert len(results) >= 2
            for r in results:
                assert "Goal" in r.get("_labels", [])

            # Find by starts_with
            results = storage.find_nodes("Find Me F", match_type="starts_with")
            assert len(results) >= 1
        finally:
            # Cleanup
            storage.delete_node("goal:find_test_1")
            storage.delete_node("goal:find_test_2")
            storage.delete_node("project:find_test")

    def test_update_node(self, storage):
        """Test updating node properties."""
        # Setup
        storage.add_node(
            "Goal", "goal:update_test", "Update Test", {"status": "pending"}
        )

        try:
            # Update
            updated = storage.update_node(
                "goal:update_test", {"status": "completed", "notes": "Done!"}
            )
            assert updated is not None
            assert updated["status"] == "completed"
            assert updated["notes"] == "Done!"
            assert "updated_at" in updated
        finally:
            storage.delete_node("goal:update_test")

    def test_delete_node(self, storage):
        """Test deleting a node."""
        storage.add_node("Goal", "goal:delete_test", "Delete Test", {})

        result = storage.delete_node("goal:delete_test")
        assert result["deleted"] is True

        # Verify deleted
        fetched = storage.get_node("goal:delete_test")
        assert fetched is None

    def test_add_and_delete_edge(self, storage):
        """Test adding and deleting edges."""
        # Setup nodes
        storage.add_node("Project", "project:edge_test", "Edge Test Project", {})
        storage.add_node("Goal", "goal:edge_test", "Edge Test Goal", {})

        try:
            # Add edge
            result = storage.add_edge(
                "project:edge_test",
                "goal:edge_test",
                "CONTRIBUTES_TO",
                {"confidence": 0.9},
            )
            assert result["success"] is True
            assert "edge_id" in result

            # Get edge
            edge = storage.get_edge(result["edge_id"])
            assert edge is not None
            assert edge["relation"] == "CONTRIBUTES_TO"

            # Delete edge
            deleted = storage.delete_edge(result["edge_id"])
            assert deleted is True

            # Verify deleted
            edge = storage.get_edge(result["edge_id"])
            assert edge is None
        finally:
            storage.delete_node("project:edge_test")
            storage.delete_node("goal:edge_test")

    def test_get_neighbors(self, storage):
        """Test getting neighbors of a node."""
        # Setup
        storage.add_node("Person", "person:neighbor_test", "Test Person", {})
        storage.add_node("Goal", "goal:neighbor_test_1", "Goal 1", {})
        storage.add_node("Goal", "goal:neighbor_test_2", "Goal 2", {})
        storage.add_edge("person:neighbor_test", "goal:neighbor_test_1", "HAS_GOAL")
        storage.add_edge("person:neighbor_test", "goal:neighbor_test_2", "HAS_GOAL")

        try:
            # Get all neighbors
            neighbors = storage.get_neighbors("person:neighbor_test")
            assert len(neighbors) >= 2

            # Get outgoing only
            neighbors = storage.get_neighbors("person:neighbor_test", direction="out")
            assert len(neighbors) >= 2
            for n in neighbors:
                assert n["direction"] == "out"

            # Filter by edge type
            neighbors = storage.get_neighbors(
                "person:neighbor_test", edge_types=["HAS_GOAL"]
            )
            assert len(neighbors) >= 2
        finally:
            storage.delete_node("person:neighbor_test")
            storage.delete_node("goal:neighbor_test_1")
            storage.delete_node("goal:neighbor_test_2")

    def test_find_path(self, storage):
        """Test finding path between nodes."""
        # Setup a simple path: A -> B -> C
        storage.add_node("Concept", "concept:path_a", "Path A", {})
        storage.add_node("Concept", "concept:path_b", "Path B", {})
        storage.add_node("Concept", "concept:path_c", "Path C", {})
        storage.add_edge("concept:path_a", "concept:path_b", "RELATED_TO")
        storage.add_edge("concept:path_b", "concept:path_c", "RELATED_TO")

        try:
            path = storage.find_path("concept:path_a", "concept:path_c")
            assert path is not None
            assert path["length"] == 2
            assert len(path["node_ids"]) == 3
            assert path["node_ids"][0] == "concept:path_a"
            assert path["node_ids"][-1] == "concept:path_c"
        finally:
            storage.delete_node("concept:path_a")
            storage.delete_node("concept:path_b")
            storage.delete_node("concept:path_c")

    def test_invalidate_edge(self, storage):
        """Test soft-deleting an edge."""
        storage.add_node("Belief", "belief:invalid_test", "Test Belief", {})
        storage.add_node("Goal", "goal:invalid_test", "Test Goal", {})
        edge_result = storage.add_edge(
            "belief:invalid_test", "goal:invalid_test", "SUPPORTS"
        )

        try:
            # Invalidate
            result = storage.invalidate_edge(
                edge_result["edge_id"], reason="No longer true"
            )
            assert result is not None
            assert "invalid_at" in result
            assert result["invalidation_reason"] == "No longer true"

            # Edge still exists but marked invalid
            edge = storage.get_edge(edge_result["edge_id"])
            assert edge is not None
        finally:
            storage.delete_edge(edge_result["edge_id"])
            storage.delete_node("belief:invalid_test")
            storage.delete_node("goal:invalid_test")


class TestNeo4jEdgeCases:
    """Edge case tests for Neo4j storage."""

    @pytest.fixture
    def storage(self):
        """Neo4j storage fixture; skips if Neo4j unavailable."""
        try:
            s = Neo4jStorage(
                uri="bolt://localhost:7687", user="neo4j", password="obsidian"
            )
            s.get_stats()
            s.clear()
            yield s
            s.clear()
            s.close()
        except Exception as exc:
            pytest.skip(f"Neo4j not available: {exc}")

    def test_merge_nodes_simple_reports_missing_merge_node(self, storage):
        """Merging with non-existent merge_id should not report success."""
        storage.add_node("Goal", "goal:keep_only", "Keep Only", {})

        result = storage.merge_nodes_simple("goal:keep_only", "goal:missing")

        assert result["merged"] is False
        assert "error" in result

    def test_merge_nodes_simple_preserves_edge_properties(self, storage):
        """Transferred edges should keep semantic properties."""
        storage.add_node("Goal", "goal:keep_props", "Keep Props", {})
        storage.add_node("Goal", "goal:merge_props", "Merge Props", {})
        storage.add_node("Goal", "goal:target_props", "Target Props", {})

        storage.add_edge(
            "goal:merge_props",
            "goal:target_props",
            "RELATED_TO",
            {
                "confidence": 0.92,
                "fact": "from merge node",
                "source_note": "note:test_props",
            },
        )

        result = storage.merge_nodes_simple("goal:keep_props", "goal:merge_props")
        assert result["merged"] is True

        edges = storage.find_edges(
            from_id="goal:keep_props", to_id="goal:target_props", relation="RELATED_TO"
        )
        assert len(edges) == 1
        edge_props = edges[0]["edge"]
        assert edge_props["confidence"] == 0.92
        assert edge_props["fact"] == "from merge node"
        assert edge_props["source_note"] == "note:test_props"

    def test_add_node_is_idempotent(self, storage):
        """Calling add_node twice with same id should not create duplicates."""
        storage.add_node("Goal", "goal:idem", "Original Name", {"status": "v1"})
        storage.add_node("Goal", "goal:idem", "Updated Name", {"status": "v2"})

        # get_node returns the single node — properties should be updated
        result = storage.get_node("goal:idem")
        assert result is not None
        assert result["node"]["name"] == "Updated Name"
        assert result["node"]["status"] == "v2"
        # created_at should be preserved from first creation
        assert "created_at" in result["node"]

    def test_add_edge_is_idempotent(self, storage):
        """Calling add_edge twice with same (from, to, type) should not duplicate."""
        storage.add_node("Note", "note:e_idem_a", "A", {})
        storage.add_node("Note", "note:e_idem_b", "B", {})

        r1 = storage.add_edge(
            "note:e_idem_a", "note:e_idem_b", "RELATED_TO", {"confidence": 0.5}
        )
        r2 = storage.add_edge(
            "note:e_idem_a", "note:e_idem_b", "RELATED_TO", {"confidence": 0.9}
        )

        assert r1["success"] and r2["success"]
        # Same edge id — MERGE reused the existing edge
        assert r1["edge_id"] == r2["edge_id"]

        # Only one edge exists
        edges = storage.find_edges(
            from_id="note:e_idem_a", to_id="note:e_idem_b", relation="RELATED_TO"
        )
        assert len(edges) == 1
        # Props were updated
        assert edges[0]["edge"]["confidence"] == 0.9


class TestSourceNoteEdges:
    """Tests for source_note-aware edge operations."""

    @pytest.fixture
    def storage(self):
        """Neo4j storage with cleanup."""
        try:
            s = Neo4jStorage(
                uri="bolt://localhost:7687", user="neo4j", password="obsidian"
            )
            s.get_stats()
            s.clear()
            yield s
            s.clear()
            s.close()
        except Exception as exc:
            pytest.skip(f"Neo4j not available: {exc}")

    def test_add_edge_with_source_note(self, storage):
        """Test adding an edge with source_note provenance."""
        storage.add_node("Note", "note:source_a", "Source A", {})
        storage.add_node("Note", "note:source_b", "Source B", {})

        result = storage.add_edge(
            "note:source_a",
            "note:source_b",
            "WIKILINK",
            source_note="note:source_a",
        )

        assert result["success"] is True

        edge = storage.get_edge(result["edge_id"])
        assert edge is not None
        assert edge["edge"]["source_note"] == "note:source_a"

    def test_get_edges_by_source_note(self, storage):
        """Test filtering edges by source_note."""
        storage.add_node("Note", "note:filter_a", "Filter A", {})
        storage.add_node("Note", "note:filter_b", "Filter B", {})
        storage.add_node("Note", "note:filter_c", "Filter C", {})

        storage.add_edge(
            "note:filter_a",
            "note:filter_b",
            "WIKILINK",
            source_note="note:filter_a",
        )
        storage.add_edge(
            "note:filter_a",
            "note:filter_c",
            "RELATED_TO",
            source_note="note:filter_a",
        )

        # Get all edges from this source_note
        all_edges = storage.get_edges_by_source_note("note:filter_a", "note:filter_a")
        assert len(all_edges) == 2

        # Get only WIKILINK edges from this source_note
        wikilink_edges = storage.get_edges_by_source_note(
            "note:filter_a", "note:filter_a", relation="WIKILINK"
        )
        assert len(wikilink_edges) == 1
        assert wikilink_edges[0]["target_name"] == "Filter B"

    def test_get_edges_by_source_note_direction(self, storage):
        """Test edge direction filtering with source_note."""
        storage.add_node("Note", "note:dir_a", "Dir A", {})
        storage.add_node("Note", "note:dir_b", "Dir B", {})

        storage.add_edge(
            "note:dir_a", "note:dir_b", "WIKILINK", source_note="note:dir_a"
        )

        out_edges = storage.get_edges_by_source_note(
            "note:dir_a", "note:dir_a", direction="out"
        )
        assert len(out_edges) == 1

        in_edges = storage.get_edges_by_source_note(
            "note:dir_b", "note:dir_a", direction="in"
        )
        assert len(in_edges) == 1

        out_edges_b = storage.get_edges_by_source_note(
            "note:dir_b", "note:dir_a", direction="out"
        )
        assert len(out_edges_b) == 0

    def test_delete_edges_by_source_note(self, storage):
        """Test deleting edges by source_note."""
        storage.add_node("Note", "note:del_a", "Del A", {})
        storage.add_node("Note", "note:del_b", "Del B", {})
        storage.add_node("Note", "note:del_c", "Del C", {})

        storage.add_edge(
            "note:del_a", "note:del_b", "WIKILINK", source_note="note:del_a"
        )
        storage.add_edge(
            "note:del_a", "note:del_c", "RELATED_TO", source_note="note:other"
        )

        # Delete only edges from note:del_a source
        deleted = storage.delete_edges_by_source_note("note:del_a", "note:del_a")
        assert deleted == 1

        # Verify del_a-sourced edge gone
        edges_a = storage.get_edges_by_source_note("note:del_a", "note:del_a")
        assert len(edges_a) == 0

        # Verify other-sourced edge still exists
        edges_other = storage.get_edges_by_source_note("note:del_a", "note:other")
        assert len(edges_other) == 1

    def test_delete_edges_by_source_note_with_relation_filter(self, storage):
        """Test deleting edges filtered by both source_note and relation."""
        storage.add_node("Note", "note:rf_a", "RF A", {})
        storage.add_node("Note", "note:rf_b", "RF B", {})
        storage.add_node("Note", "note:rf_c", "RF C", {})

        storage.add_edge("note:rf_a", "note:rf_b", "WIKILINK", source_note="note:rf_a")
        storage.add_edge(
            "note:rf_a", "note:rf_c", "RELATED_TO", source_note="note:rf_a"
        )

        # Delete only WIKILINK edges from this source
        deleted = storage.delete_edges_by_source_note(
            "note:rf_a", "note:rf_a", relation="WIKILINK"
        )
        assert deleted == 1

        # RELATED_TO edge should survive
        remaining = storage.get_edges_by_source_note("note:rf_a", "note:rf_a")
        assert len(remaining) == 1
        assert remaining[0]["relation"] == "RELATED_TO"

    def test_delete_edges_by_source_note_returns_count(self, storage):
        """Test delete returns accurate count."""
        storage.add_node("Note", "note:count_a", "Count A", {})
        storage.add_node("Note", "note:count_b", "Count B", {})
        storage.add_node("Note", "note:count_c", "Count C", {})

        storage.add_edge(
            "note:count_a", "note:count_b", "WIKILINK", source_note="note:count_a"
        )
        storage.add_edge(
            "note:count_a", "note:count_c", "WIKILINK", source_note="note:count_a"
        )

        deleted = storage.delete_edges_by_source_note("note:count_a", "note:count_a")
        assert deleted == 2

    def test_edge_without_source_note(self, storage):
        """Test edges without source_note are not returned by source_note filter."""
        storage.add_node("Note", "note:nosrc_a", "NoSrc A", {})
        storage.add_node("Note", "note:nosrc_b", "NoSrc B", {})

        storage.add_edge("note:nosrc_a", "note:nosrc_b", "WIKILINK")

        edges = storage.get_edges_by_source_note("note:nosrc_a", "note:nosrc_a")
        assert len(edges) == 0


class TestNeo4jVectorSearch:
    """Tests for native Neo4j vector search."""

    @pytest.fixture
    def storage(self):
        try:
            s = Neo4jStorage(
                uri="bolt://localhost:7687", user="neo4j", password="obsidian"
            )
            s.get_stats()
            s.clear()
            s.ensure_vector_index()
            yield s
            s.clear()
            s.close()
        except Exception as exc:
            pytest.skip(f"Neo4j not available: {exc}")

    def test_set_and_search_embedding(self, storage):
        """Store embedding on node and retrieve via vector search."""
        storage.add_node(
            "Concept", "concept:vec_test", "Vector Test", {"summary": "A test concept"}
        )
        embedding = [0.1] * 384
        assert storage.set_embedding("concept:vec_test", embedding) is True

        import time

        time.sleep(1)  # wait for index to catch up

        results = storage.search_similar(embedding, limit=5)
        assert len(results) >= 1
        found = [r for r in results if r["node_id"] == "concept:vec_test"]
        assert len(found) == 1
        assert found[0]["name"] == "Vector Test"
        assert found[0]["node_type"] == "Concept"
        assert "embedding" not in found[0]  # embedding must NOT be exposed

    def test_set_embedding_nonexistent_node(self, storage):
        """set_embedding on missing node returns False."""
        assert storage.set_embedding("concept:does_not_exist", [0.1] * 384) is False

    def test_clean_node_strips_embedding(self, storage):
        """get_node should never expose embedding property."""
        storage.add_node("Goal", "goal:clean_test", "Clean Test", {})
        storage.set_embedding("goal:clean_test", [0.5] * 384)

        result = storage.get_node("goal:clean_test")
        assert result is not None
        assert "embedding" not in result["node"]
        # Entity internal label should also be stripped
        assert "Entity" not in result["node"].get("_labels", [])

    def test_search_with_type_filter(self, storage):
        """search_similar respects node_types filter."""
        storage.add_node("Goal", "goal:filter_a", "Goal A", {})
        storage.add_node("Person", "person:filter_b", "Person B", {})
        emb = [0.2] * 384
        storage.set_embedding("goal:filter_a", emb)
        storage.set_embedding("person:filter_b", emb)

        import time

        time.sleep(1)

        results = storage.search_similar(emb, node_types=["Goal"], limit=10)
        for r in results:
            assert r["node_type"] == "Goal"

    def test_stats_hides_entity_label(self, storage):
        """get_stats should not include Entity in by_label."""
        storage.add_node("Concept", "concept:stats_test", "Stats Test", {})
        storage.set_embedding("concept:stats_test", [0.3] * 384)

        stats = storage.get_stats()
        assert "Entity" not in stats.get("by_label", {})

    def test_set_embedding_dimension_mismatch_raises(self, storage):
        """set_embedding should reject vectors with wrong dimensions."""
        storage.add_node("Goal", "goal:dim_mismatch", "Dim Mismatch", {})
        with pytest.raises(ValueError):
            storage.set_embedding("goal:dim_mismatch", [0.1] * 10)

    def test_search_dimension_mismatch_raises(self, storage):
        """search_similar should reject query vectors with wrong dimensions."""
        with pytest.raises(ValueError):
            storage.search_similar([0.1] * 10, limit=5)

    def test_vector_index_exists_after_ensure(self, storage):
        """ensure_vector_index should leave the expected index available."""
        storage.ensure_vector_index()

        with storage.driver.session() as session:
            record = session.run(
                """
                SHOW INDEXES YIELD name, type
                WHERE name = 'entity_embeddings'
                RETURN type
                """
            ).single()

        assert record is not None
        assert record["type"] == "VECTOR"
