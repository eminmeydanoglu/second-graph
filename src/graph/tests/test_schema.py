"""Tests for graph schema validation."""

import pytest

from src.graph.schema import (
    NodeType,
    EdgeType,
    SourceType,
    generate_node_id,
    generate_source_id,
    validate_node_type,
    validate_edge_type,
    validate_edge,
    get_node_types,
    get_edge_types,
)


class TestSchema:
    """Test schema validation functions."""

    def test_generate_node_id_basic(self):
        """Test basic node ID generation."""
        assert generate_node_id("Goal", "Build AI") == "goal:build_ai"
        assert generate_node_id("Person", "Emin") == "person:emin"
        assert generate_node_id("Project", "Obsidian Brain") == "project:obsidian_brain"

    def test_generate_node_id_normalization(self):
        """Test that node IDs are normalized correctly."""
        # Handles hyphens
        assert (
            generate_node_id("Goal", "Build-Autonomous-Agents")
            == "goal:build_autonomous_agents"
        )
        # Handles multiple spaces
        assert generate_node_id("Goal", "Build   AI") == "goal:build_ai"
        # Handles leading/trailing whitespace
        assert generate_node_id("Goal", "  Build AI  ") == "goal:build_ai"

    def test_validate_node_type_valid(self):
        """Test that valid node types pass validation."""
        assert validate_node_type("Goal") is True
        assert validate_node_type("Person") is True
        assert validate_node_type("Project") is True
        assert validate_node_type("Concept") is True
        assert validate_node_type("Belief") is True
        assert validate_node_type("Value") is True

    def test_validate_node_type_invalid(self):
        """Test that invalid node types fail validation."""
        assert validate_node_type("FakeType") is False
        assert validate_node_type("") is False
        assert validate_node_type("goal") is False  # Case sensitive

    def test_validate_edge_type_valid(self):
        """Test that valid edge types pass validation."""
        assert validate_edge_type("CONTRIBUTES_TO") is True
        assert validate_edge_type("MOTIVATES") is True
        assert validate_edge_type("HAS_GOAL") is True
        assert validate_edge_type("RELATED_TO") is True

    def test_validate_edge_type_invalid(self):
        """Test that invalid edge types fail validation."""
        assert validate_edge_type("FAKE_RELATION") is False
        assert validate_edge_type("") is False

    def test_validate_edge_valid(self):
        """Test edge validation for valid combinations."""
        # Project -> Goal via CONTRIBUTES_TO is valid
        result = validate_edge("Project", "Goal", "CONTRIBUTES_TO")
        assert result.valid is True
        assert len(result.errors) == 0

        # Value -> Goal via MOTIVATES is valid
        result = validate_edge("Value", "Goal", "MOTIVATES")
        assert result.valid is True

        # Person -> Goal via HAS_GOAL is valid
        result = validate_edge("Person", "Goal", "HAS_GOAL")
        assert result.valid is True

    def test_validate_edge_warnings(self):
        """Test that constraint violations produce warnings (not errors) by default."""
        # Person -> Goal via CONTRIBUTES_TO violates constraints (Person not in sources)
        result = validate_edge("Person", "Goal", "CONTRIBUTES_TO")
        # With strict=False, this should produce warnings not errors
        assert result.valid is True
        assert len(result.warnings) > 0

    def test_validate_edge_strict(self):
        """Test strict validation mode."""
        # With strict=True, constraint violations become errors
        result = validate_edge("Person", "Goal", "CONTRIBUTES_TO", strict=True)
        assert result.valid is False
        assert len(result.errors) > 0

    def test_validate_edge_unknown_types(self):
        """Test validation with unknown types."""
        # Unknown types should warn but not error (non-strict mode)
        result = validate_edge("UnknownType", "Goal", "RELATED_TO")
        assert result.valid is True
        assert len(result.warnings) > 0

    def test_get_node_types(self):
        """Test that all node types are returned."""
        types = get_node_types()
        assert "Goal" in types
        assert "Person" in types
        assert "Project" in types
        assert len(types) == len(NodeType)

    def test_get_edge_types(self):
        """Test that all edge types are returned."""
        types = get_edge_types()
        assert "CONTRIBUTES_TO" in types
        assert "MOTIVATES" in types
        assert "RELATED_TO" in types
        assert len(types) == len(EdgeType)


class TestSourceType:
    """Tests for SourceType enum and generate_source_id."""

    def test_source_type_values(self):
        """Test SourceType enum has expected values."""
        assert SourceType.FILE.value == "file"
        assert SourceType.AGENT.value == "agent"
        assert SourceType.EXTRACTION.value == "extraction"
        assert SourceType.SYSTEM.value == "system"

    def test_generate_source_id_file(self):
        """Test file source ID generation."""
        source_id = generate_source_id(SourceType.FILE, "/vault/Projects/AI.md")
        assert source_id == "file:/vault/Projects/AI.md"

    def test_generate_source_id_agent(self):
        """Test agent source ID generation (no identifier)."""
        source_id = generate_source_id(SourceType.AGENT)
        assert source_id == "agent"

    def test_generate_source_id_extraction(self):
        """Test extraction source ID generation."""
        source_id = generate_source_id(SourceType.EXTRACTION, "v1:/vault/AI.md")
        assert source_id == "extraction:v1:/vault/AI.md"

    def test_generate_source_id_system(self):
        """Test system source ID generation."""
        source_id = generate_source_id(SourceType.SYSTEM, "init")
        assert source_id == "system:init"

    def test_source_id_format_consistency(self):
        """Test that source IDs follow expected format."""
        # FILE, EXTRACTION, SYSTEM use {type}:{identifier}
        assert generate_source_id(SourceType.FILE, "test") == "file:test"
        assert generate_source_id(SourceType.EXTRACTION, "test") == "extraction:test"
        assert generate_source_id(SourceType.SYSTEM, "test") == "system:test"
        # AGENT is just "agent" (no identifier)
        assert generate_source_id(SourceType.AGENT) == "agent"
