"""Tests for Neo4j CRUD operations."""

import pytest

from src.graph.neo4j_storage import Neo4jStorage
from src.graph.schema import SourceType, generate_source_id


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

    def test_get_subgraph_missing_center_node_returns_empty_graph(self, storage):
        """Missing center node should not crash and should return empty result."""
        result = storage.get_subgraph("goal:does_not_exist")
        assert result == {"nodes": [], "edges": []}

    def test_merge_nodes_simple_reports_missing_merge_node(self, storage):
        """Merging with non-existent merge_id should not report success."""
        storage.add_node("Goal", "goal:keep_only", "Keep Only", {})

        result = storage.merge_nodes_simple("goal:keep_only", "goal:missing")

        assert result["merged"] is False
        assert "error" in result


class TestSourceAwareEdges:
    """Tests for source-aware edge operations."""

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

    def test_add_edge_with_source(self, storage):
        """Test adding an edge with source provenance."""
        storage.add_node("Note", "note:source_a", "Source A", {})
        storage.add_node("Note", "note:source_b", "Source B", {})

        source_id = generate_source_id(SourceType.FILE, "/vault/test.md")
        result = storage.add_edge(
            "note:source_a",
            "note:source_b",
            "WIKILINK",
            source=source_id,
        )

        assert result["success"] is True

        # Verify source is stored
        edge = storage.get_edge(result["edge_id"])
        assert edge is not None
        assert edge["edge"]["source"] == source_id

    def test_get_edges_by_source(self, storage):
        """Test filtering edges by source."""
        storage.add_node("Note", "note:filter_a", "Filter A", {})
        storage.add_node("Note", "note:filter_b", "Filter B", {})
        storage.add_node("Note", "note:filter_c", "Filter C", {})

        file_source = generate_source_id(SourceType.FILE, "/vault/test.md")
        agent_source = generate_source_id(SourceType.AGENT)

        # Add edges with different sources
        storage.add_edge(
            "note:filter_a", "note:filter_b", "WIKILINK", source=file_source
        )
        storage.add_edge(
            "note:filter_a", "note:filter_c", "RELATED_TO", source=agent_source
        )

        # Get only file-sourced edges
        file_edges = storage.get_edges_by_source("note:filter_a", file_source)
        assert len(file_edges) == 1
        assert file_edges[0]["target_name"] == "Filter B"

        # Get only agent-sourced edges
        agent_edges = storage.get_edges_by_source("note:filter_a", agent_source)
        assert len(agent_edges) == 1
        assert agent_edges[0]["target_name"] == "Filter C"

    def test_get_edges_by_source_direction(self, storage):
        """Test edge direction filtering with source."""
        storage.add_node("Note", "note:dir_a", "Dir A", {})
        storage.add_node("Note", "note:dir_b", "Dir B", {})

        source = generate_source_id(SourceType.FILE, "/vault/dir.md")
        storage.add_edge("note:dir_a", "note:dir_b", "WIKILINK", source=source)

        # Outgoing from A
        out_edges = storage.get_edges_by_source("note:dir_a", source, direction="out")
        assert len(out_edges) == 1

        # Incoming to B
        in_edges = storage.get_edges_by_source("note:dir_b", source, direction="in")
        assert len(in_edges) == 1

        # No outgoing from B
        out_edges_b = storage.get_edges_by_source("note:dir_b", source, direction="out")
        assert len(out_edges_b) == 0

    def test_delete_edges_by_source(self, storage):
        """Test deleting edges by source."""
        storage.add_node("Note", "note:del_a", "Del A", {})
        storage.add_node("Note", "note:del_b", "Del B", {})
        storage.add_node("Note", "note:del_c", "Del C", {})

        file_source = generate_source_id(SourceType.FILE, "/vault/del.md")
        agent_source = generate_source_id(SourceType.AGENT)

        storage.add_edge("note:del_a", "note:del_b", "WIKILINK", source=file_source)
        storage.add_edge("note:del_a", "note:del_c", "RELATED_TO", source=agent_source)

        # Delete only file-sourced edges
        deleted = storage.delete_edges_by_source("note:del_a", file_source)
        assert deleted == 1

        # Verify file edge gone
        file_edges = storage.get_edges_by_source("note:del_a", file_source)
        assert len(file_edges) == 0

        # Verify agent edge still exists
        agent_edges = storage.get_edges_by_source("note:del_a", agent_source)
        assert len(agent_edges) == 1

    def test_delete_edges_by_source_returns_count(self, storage):
        """Test delete returns accurate count."""
        storage.add_node("Note", "note:count_a", "Count A", {})
        storage.add_node("Note", "note:count_b", "Count B", {})
        storage.add_node("Note", "note:count_c", "Count C", {})

        source = generate_source_id(SourceType.FILE, "/vault/count.md")
        storage.add_edge("note:count_a", "note:count_b", "WIKILINK", source=source)
        storage.add_edge("note:count_a", "note:count_c", "WIKILINK", source=source)

        deleted = storage.delete_edges_by_source("note:count_a", source)
        assert deleted == 2

    def test_edge_without_source(self, storage):
        """Test edges without source are not returned by source filter."""
        storage.add_node("Note", "note:nosrc_a", "NoSrc A", {})
        storage.add_node("Note", "note:nosrc_b", "NoSrc B", {})

        # Add edge without source
        storage.add_edge("note:nosrc_a", "note:nosrc_b", "WIKILINK")

        # Should not be found when filtering by source
        source = generate_source_id(SourceType.FILE, "/vault/nosrc.md")
        edges = storage.get_edges_by_source("note:nosrc_a", source)
        assert len(edges) == 0
