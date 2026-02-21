"""Synchronization logic between Markdown files and Knowledge Graph."""

import json
import logging
from pathlib import Path, PurePosixPath
from typing import Any

from ..parser.markdown import parse_note, ParsedNote
from .neo4j_storage import Neo4jStorage
from .routing_text import build_routing_text
from .schema import NodeType, EdgeType, generate_node_id
from ..vector.embedder import Embedder

log = logging.getLogger(__name__)


class NoteSynchronizer:
    """Synchronizes Obsidian notes with the Neo4j graph.

    Performs 'intelligent sync':
    1. Parses the note file.
    2. Compares with existing graph state.
    3. Adds missing edges/nodes.
    4. Removes deleted edges.
    5. Updates node properties.
    6. Stores embeddings in Neo4j (native vector index).
    """

    def __init__(
        self,
        storage: Neo4jStorage,
        embedder: Embedder,
    ):
        self.storage = storage
        self.embedder = embedder

    def sync_note_from_file(
        self,
        file_path: str | Path,
        vault_root: str | Path | None = None,
    ) -> dict[str, Any]:
        """Sync a single note file into the graph.

        Args:
            file_path: Absolute path to the markdown file.
            vault_root: Optional vault root for stable relative-path identity.

        Returns:
            Dict containing stats of changes applied.
        """
        path = Path(file_path)
        if not path.exists():
            return {"success": False, "error": f"File not found: {path}"}

        resolved_vault_root = self._resolve_vault_root(path, vault_root)

        try:
            note = parse_note(path, vault_root=resolved_vault_root)
        except Exception as e:
            return {"success": False, "error": f"Failed to parse note: {e}"}

        node_type = self._determine_node_type(note)
        node_id = self._note_id_from_path(note.path)
        vault_rel_path = self._normalize_note_path(note.path)
        display_name = Path(vault_rel_path).name

        frontmatter = self._sanitize_frontmatter(note.frontmatter)

        props = {
            "path": str(path.resolve()),
            "vault_rel_path": vault_rel_path,
            "stem": PurePosixPath(vault_rel_path).name,
            "title": note.title,
            "modified_at": path.stat().st_mtime,
            **frontmatter,
            "placeholder": False,
        }

        existing = self.storage.get_node(node_id)
        if existing:
            self.storage.update_node(node_id, {"name": display_name, **props})
            action = "updated"
        else:
            self.storage.add_node(node_type, node_id, display_name, props)
            action = "created"

        edge_stats = self._sync_wikilinks(node_id, note.wikilinks, note.path)
        tag_stats = self._sync_tags(node_id, note.tags)

        # Still update tags property for easy access, but now we also have real graph edges
        self.storage.update_node(node_id, {"tags": note.tags})

        self._update_embedding(node_id, node_type, note)

        return {
            "success": True,
            "node_id": node_id,
            "action": action,
            "edges": edge_stats,
            "tags": tag_stats,
            "source_note": node_id,
        }

    def _sanitize_frontmatter(self, frontmatter: dict[str, Any]) -> dict[str, Any]:
        """Prepare frontmatter values for Neo4j property constraints."""
        cleaned: dict[str, Any] = {}
        for key, value in frontmatter.items():
            if key == "title":
                continue
            cleaned[key] = self._sanitize_property_value(value)
        return cleaned

    def _sanitize_property_value(self, value: Any) -> Any:
        """Convert unsupported Neo4j property shapes into storable values."""
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        if isinstance(value, list):
            normalized = [self._sanitize_property_value(item) for item in value]
            if all(
                item is None or isinstance(item, (str, int, float, bool))
                for item in normalized
            ):
                return normalized
            return json.dumps(value, ensure_ascii=False, sort_keys=True)

        if isinstance(value, dict):
            return json.dumps(value, ensure_ascii=False, sort_keys=True)

        return str(value)

    # Backward compatibility alias
    source_note = sync_note_from_file

    def _determine_node_type(self, note: ParsedNote) -> str:
        """Determine canonical node type from frontmatter or default to Note."""
        if "type" in note.frontmatter:
            raw_type = str(note.frontmatter["type"])
            for node_type in NodeType:
                if node_type.value.lower() == raw_type.lower():
                    return node_type.value
        return NodeType.NOTE.value

    def _sync_wikilinks(
        self,
        node_id: str,
        wikilinks: list[str],
        source_note_path: Path,
    ) -> dict:
        """Diff and sync outgoing wikilinks for a note.

        Filters by relation=WIKILINK and source_note=node_id, so only
        wikilink edges owned by this note are affected.
        """
        stats = {"added": 0, "removed": 0, "kept": 0}

        current_edges = self.storage.get_edges_by_source_note(
            node_id, source_note=node_id, relation="WIKILINK", direction="out"
        )

        current_targets: dict[str, str] = {}
        for edge in current_edges:
            target_id = edge.get("target_id")
            edge_id = edge.get("edge", {}).get("id")
            if target_id and edge_id:
                current_targets[target_id] = edge_id

        desired_targets = set()
        for raw_target in wikilinks:
            target_id = self._resolve_wikilink_target(raw_target, source_note_path)
            if target_id:
                desired_targets.add(target_id)

        existing_targets = set(current_targets.keys())

        to_add = desired_targets - existing_targets
        to_remove = existing_targets - desired_targets
        to_keep = desired_targets & existing_targets

        stats["kept"] = len(to_keep)

        for target_id in to_remove:
            edge_id = current_targets[target_id]
            self.storage.delete_edge(edge_id)
            stats["removed"] += 1

        for target_id in to_add:
            self.storage.add_edge(
                node_id, target_id, EdgeType.WIKILINK.value, source_note=node_id
            )
            stats["added"] += 1

        return stats

    def _resolve_wikilink_target(
        self,
        raw_target: str,
        source_note_path: Path,
    ) -> str | None:
        """Resolve a wikilink target to a Note node ID.

        Resolution order (alias intentionally skipped):
        1) vault-relative path candidates
        2) stem-based candidates
        3) placeholder creation
        """
        target = self._normalize_wikilink_target(raw_target)
        if not target:
            return None

        source_rel_path = self._normalize_note_path(source_note_path)
        source_dir = PurePosixPath(source_rel_path).parent
        if str(source_dir) == ".":
            source_dir = PurePosixPath("")

        target_path = PurePosixPath(target)
        has_path = "/" in target

        candidate_paths: list[str] = []
        if has_path:
            candidate_paths.append(
                self._normalize_posix_path((source_dir / target_path).as_posix())
            )
            candidate_paths.append(self._normalize_posix_path(target_path.as_posix()))
        else:
            candidate_paths.append(
                self._normalize_posix_path((source_dir / target_path).as_posix())
            )
            candidate_paths.append(self._normalize_posix_path(target_path.as_posix()))

        dedup_paths: list[str] = []
        seen_paths: set[str] = set()
        for candidate in candidate_paths:
            if not candidate or candidate in seen_paths:
                continue
            seen_paths.add(candidate)
            dedup_paths.append(candidate)

        for rel_path in dedup_paths:
            match = self.storage.find_note_by_vault_rel_path(rel_path)
            if match:
                return match["id"]

        stem = target_path.name
        candidates = self.storage.find_note_candidates_by_stem(stem)
        if candidates:
            best = self._pick_best_note_candidate(candidates, source_dir)
            if best:
                return best["id"]

        placeholder_path = (
            dedup_paths[0] if dedup_paths else self._normalize_posix_path(stem)
        )
        placeholder_name = target_path.name
        target_id = generate_node_id(NodeType.NOTE.value, placeholder_path)

        if not self.storage.get_node(target_id):
            self.storage.add_node(
                NodeType.NOTE.value,
                target_id,
                placeholder_name,
                {
                    "placeholder": True,
                    "title": placeholder_name,
                    "vault_rel_path": placeholder_path,
                    "stem": PurePosixPath(placeholder_path).name,
                },
            )

        return target_id

    def _pick_best_note_candidate(
        self,
        candidates: list[dict],
        source_dir: PurePosixPath,
    ) -> dict | None:
        """Select best candidate note using folder proximity and quality."""
        source_parts = [part for part in source_dir.parts if part and part != "."]

        def key(node: dict) -> tuple[int, int, int, str]:
            placeholder_penalty = 1 if node.get("placeholder") else 0
            rel_path = self._normalize_posix_path(str(node.get("vault_rel_path") or ""))

            if rel_path:
                candidate_dir = PurePosixPath(rel_path).parent
                candidate_parts = [
                    part for part in candidate_dir.parts if part and part != "."
                ]
                shared = 0
                for src_part, cand_part in zip(source_parts, candidate_parts):
                    if src_part != cand_part:
                        break
                    shared += 1
                folder_distance = (len(source_parts) - shared) + (
                    len(candidate_parts) - shared
                )
                rel_len = len(rel_path)
            else:
                folder_distance = 999
                rel_len = 999

            return (placeholder_penalty, folder_distance, rel_len, node.get("id", ""))

        return min(candidates, key=key) if candidates else None

    def _resolve_vault_root(
        self,
        note_path: Path,
        vault_root: str | Path | None,
    ) -> Path:
        """Resolve vault root for stable path-based identity."""
        if vault_root is not None:
            root = Path(vault_root)
            return root if root.exists() else note_path.parent

        detected = self._detect_vault_root(note_path)
        return detected if detected else note_path.parent

    def _detect_vault_root(self, note_path: Path) -> Path | None:
        """Detect Obsidian vault root by looking for .obsidian directory."""
        for parent in [note_path.parent, *note_path.parents]:
            if (parent / ".obsidian").is_dir():
                return parent
        return None

    def _normalize_note_path(self, note_path: Path) -> str:
        """Normalize note path into a stable vault-relative key."""
        raw = note_path.as_posix()
        if raw.lower().endswith(".md"):
            raw = raw[:-3]
        normalized = self._normalize_posix_path(raw)
        return normalized or note_path.stem

    def _note_id_from_path(self, note_path: Path) -> str:
        """Build Note node ID from normalized file path (not title)."""
        path_key = self._normalize_note_path(note_path)
        return generate_node_id(NodeType.NOTE.value, path_key)

    def _normalize_wikilink_target(self, raw_target: str) -> str:
        """Normalize wikilink text to canonical target path/stem."""
        target = raw_target.strip()
        if not target:
            return ""

        if "#" in target:
            target = target.split("#", 1)[0]
        if "^" in target:
            target = target.split("^", 1)[0]

        target = target.strip()
        if target.lower().endswith(".md"):
            target = target[:-3]

        return self._normalize_posix_path(target)

    def _normalize_posix_path(self, raw_path: str) -> str:
        """Normalize a slash path by collapsing . and .. segments."""
        parts: list[str] = []
        for part in raw_path.replace("\\", "/").split("/"):
            token = part.strip()
            if not token or token == ".":
                continue
            if token == "..":
                if parts:
                    parts.pop()
                continue
            parts.append(token)
        return "/".join(parts)

    def _sync_tags(self, node_id: str, tags: list[str]) -> dict:
        """Diff and sync tag relationships for a note.

        Ensures Tag nodes exist and TAGGED_WITH edges are current.
        """
        stats = {"added": 0, "removed": 0, "kept": 0}

        # Get current TAGGED_WITH edges from this node
        # Note: TAGGED_WITH edges don't necessarily need source_note property
        # because they are intrinsic to the note's content.
        current_edges = self.storage.get_neighbors(
            node_id, direction="out", edge_types=[EdgeType.TAGGED_WITH.value]
        )

        current_tags: dict[str, str] = {}
        for neighbor in current_edges:
            # neighbor structure: {node: {name: "tagname", ...}, edge_id: "...", ...}
            tag_name = neighbor["node"].get("name")
            if tag_name:
                current_tags[tag_name] = neighbor["edge_id"]

        desired_tags = set(tags)
        existing_tag_names = set(current_tags.keys())

        to_add = desired_tags - existing_tag_names
        to_remove = existing_tag_names - desired_tags
        to_keep = desired_tags & existing_tag_names

        stats["kept"] = len(to_keep)

        for tag_name in to_remove:
            edge_id = current_tags[tag_name]
            self.storage.delete_edge(edge_id)
            stats["removed"] += 1

        for tag_name in to_add:
            tag_id = generate_node_id(NodeType.TAG.value, tag_name)

            if not self.storage.get_node(tag_id):
                self.storage.add_node(NodeType.TAG.value, tag_id, tag_name)

            self.storage.add_edge(
                node_id,
                tag_id,
                EdgeType.TAGGED_WITH.value,
                {"confidence": 1.0},
            )
            stats["added"] += 1

        return stats

    def _update_embedding(self, node_id: str, node_type: str, note: ParsedNote):
        """Store embedding on the Neo4j node."""
        frontmatter = {k: v for k, v in note.frontmatter.items() if k != "title"}
        props = {
            **frontmatter,
            "name": note.title,
            "tags": note.tags,
        }
        text = build_routing_text(node_type, props)

        embedding = self.embedder.embed(text)
        self.storage.set_embedding(node_id, embedding)
