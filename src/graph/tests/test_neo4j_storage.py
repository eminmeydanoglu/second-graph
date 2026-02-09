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
