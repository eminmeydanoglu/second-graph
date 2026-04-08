"""Tests for entity vector storage."""

import pytest

from src.vector.store import VectorStore


class TestVectorStoreEntities:
    """Tests for entity vector storage."""

    @pytest.fixture
    def store(self, tmp_path):
        """Create a temporary vector store."""
        db_path = tmp_path / "test_vectors.db"
        return VectorStore(db_path)

    def test_add_and_search_entity(self, store):
        """Test adding and searching entities."""
        # Add entity with embedding
        embedding = [0.1] * 384  # Simple test embedding
        store.add_entity(
            node_id="goal:test_entity",
            node_type="Goal",
            name="Build Autonomous Agents",
            summary="Create AI agents that can operate independently",
            embedding=embedding,
        )

        # Search
        results = store.search_entities(embedding, limit=5)
        assert len(results) > 0
        assert results[0]["node_id"] == "goal:test_entity"
        assert results[0]["score"] > 0.9  # Should be very similar

    def test_filter_by_type(self, store):
        """Test filtering entity search by type."""
        embedding = [0.1] * 384
        store.add_entity("goal:filter_test", "Goal", "Test Goal", "A goal", embedding)
        store.add_entity(
            "person:filter_test", "Person", "Test Person", "A person", embedding
        )

        # Search with type filter
        results = store.search_entities(embedding, node_types=["Goal"], limit=5)
        assert all(r["node_type"] == "Goal" for r in results)

    def test_delete_entity(self, store):
        """Test deleting an entity."""
        embedding = [0.1] * 384
        store.add_entity(
            "goal:delete_test", "Goal", "Delete Me", "To be deleted", embedding
        )

        # Delete
        result = store.delete_entity("goal:delete_test")
        assert result is True

        # Verify gone
        entity = store.get_entity("goal:delete_test")
        assert entity is None

    def test_update_entity(self, store):
        """Test updating an entity's embedding."""
        embedding1 = [0.1] * 384
        embedding2 = [0.9] * 384

        # Add
        store.add_entity("goal:update_test", "Goal", "Update Me", "V1", embedding1)

        # Update with new embedding
        store.add_entity("goal:update_test", "Goal", "Update Me", "V2", embedding2)

        # Search with new embedding should rank higher
        results = store.search_entities(embedding2, limit=5)
        assert results[0]["node_id"] == "goal:update_test"
        assert results[0]["summary"] == "V2"

    def test_get_entity(self, store):
        """Test getting an entity by ID."""
        embedding = [0.1] * 384
        store.add_entity("goal:get_test", "Goal", "Get Me", "Description", embedding)

        entity = store.get_entity("goal:get_test")
        assert entity is not None
        assert entity["node_id"] == "goal:get_test"
        assert entity["name"] == "Get Me"

    def test_get_entity_not_found(self, store):
        """Test getting non-existent entity."""
        entity = store.get_entity("goal:does_not_exist")
        assert entity is None

    def test_entity_count(self, store):
        """Test counting entities."""
        embedding = [0.1] * 384
        assert store.entity_count() == 0

        store.add_entity("goal:count_1", "Goal", "Goal 1", "Desc", embedding)
        assert store.entity_count() == 1

        store.add_entity("goal:count_2", "Goal", "Goal 2", "Desc", embedding)
        assert store.entity_count() == 2

    def test_clear_entities(self, store):
        """Test clearing all entities."""
        embedding = [0.1] * 384
        store.add_entity("goal:clear_1", "Goal", "Goal 1", "Desc", embedding)
        store.add_entity("goal:clear_2", "Goal", "Goal 2", "Desc", embedding)
        assert store.entity_count() == 2

        store.clear_entities()
        assert store.entity_count() == 0
