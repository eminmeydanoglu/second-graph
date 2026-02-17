"""CLI for obsidian-brain."""

import asyncio
from pathlib import Path
import subprocess

import click

from .graph.scanner import VaultScanner
from .graph.sync import NoteSynchronizer
from .graph.neo4j_storage import Neo4jStorage
from .graph.routing_text import build_routing_text
from .graph.schema import get_node_types
from .vector.embedder import Embedder


@click.group()
def cli():
    """Obsidian Brain - Knowledge graph for your vault."""
    pass


def _canonical_node_type_from_labels(labels: list[str]) -> str:
    """Pick canonical schema type from node labels."""
    valid_types = get_node_types()
    for node_type in valid_types:
        if node_type in labels:
            return node_type
    return "Unknown"


@cli.command("sync-vault")
@click.argument("vault_path", type=click.Path(exists=True, path_type=Path))
@click.option("--uri", default="bolt://localhost:7687", help="Neo4j URI")
@click.option("--user", default="neo4j", help="Neo4j username")
@click.option("--password", default="obsidian", help="Neo4j password")
def sync_vault(vault_path: Path, uri: str, user: str, password: str):
    """Synchronize entire vault with Neo4j graph."""
    click.echo(f"Syncing vault: {vault_path} -> Neo4j ({uri})")

    storage = Neo4jStorage(uri=uri, user=user, password=password)
    try:
        storage.ensure_vector_index()
        embedder = Embedder()
        synchronizer = NoteSynchronizer(storage, embedder)
        scanner = VaultScanner(synchronizer)

        stats = scanner.scan_vault(vault_path)

        click.echo("\nSync Complete!")
        click.echo(f"Processed: {stats['processed']}")
        click.echo(f"Created:   {stats['created']}")
        click.echo(f"Updated:   {stats['updated']}")
        click.echo(f"Errors:    {stats['errors']}")

        if stats["errors"] > 0:
            click.echo("\nError Details:")
            for err in stats["error_details"]:
                click.echo(f"  - {err}")
    finally:
        storage.close()


@cli.command()
@click.argument("query", type=str)
@click.option("--uri", default="bolt://localhost:7687", help="Neo4j URI")
@click.option("--user", default="neo4j", help="Neo4j username")
@click.option("--password", default="obsidian", help="Neo4j password")
@click.option("--limit", "-n", type=int, default=5, help="Number of results")
def search(query: str, uri: str, user: str, password: str, limit: int):
    """Search graph by semantic similarity."""
    storage = Neo4jStorage(uri=uri, user=user, password=password)
    embedder = Embedder()

    try:
        click.echo(f"Searching for: {query}")
        query_embedding = embedder.embed(query)
        results = storage.search_similar(query_embedding, limit=limit)

        click.echo(f"\nFound {len(results)} results:\n")
        for i, r in enumerate(results, 1):
            click.echo(f"{i}. [{r['score']:.3f}] {r['name']} ({r['node_type']})")
            if r.get("summary"):
                click.echo(f"   {r['summary'][:200]}...")
            click.echo()
    finally:
        storage.close()


@cli.command("re-embed")
@click.option("--uri", default="bolt://localhost:7687", help="Neo4j URI")
@click.option("--user", default="neo4j", help="Neo4j username")
@click.option("--password", default="obsidian", help="Neo4j password")
def re_embed(uri: str, user: str, password: str):
    """Regenerate embeddings for all nodes using routing-text format."""
    storage = Neo4jStorage(uri=uri, user=user, password=password)
    embedder = Embedder()

    try:
        nodes = storage.list_nodes()
        total = len(nodes)
        click.echo(f"Re-embedding {total} nodes...")

        updated = 0
        skipped = 0
        errors = 0

        for index, node in enumerate(nodes, 1):
            node_id = node.get("id")
            if not node_id:
                skipped += 1
                continue

            node_type = _canonical_node_type_from_labels(node.get("_labels", []))
            text = build_routing_text(node_type, node)

            try:
                embedding = embedder.embed(text)
                if storage.set_embedding(node_id, embedding):
                    updated += 1
                else:
                    errors += 1
            except Exception as exc:
                errors += 1
                click.echo(f"  [error] {node_id}: {exc}")

            if index % 25 == 0 or index == total:
                click.echo(
                    f"  [{index}/{total}] updated={updated} skipped={skipped} errors={errors}"
                )

        click.echo(
            f"Done. updated={updated} skipped={skipped} errors={errors} total={total}"
        )
    finally:
        storage.close()


@cli.command()
@click.argument("node_id", type=str)
@click.option("--uri", default="bolt://localhost:7687", help="Neo4j URI")
@click.option("--user", default="neo4j", help="Neo4j username")
@click.option("--password", default="obsidian", help="Neo4j password")
def inspect(node_id: str, uri: str, user: str, password: str):
    """Inspect a node's connections in the graph."""
    storage = Neo4jStorage(uri=uri, user=user, password=password)

    try:
        result = storage.get_node(node_id)
        if not result:
            click.echo(f"Node not found: {node_id}")
            return

        node = result["node"]
        click.echo(f"\nNode: {node.get('name', node_id)}")
        click.echo(f"ID: {node.get('id')}")
        click.echo(f"Labels: {node.get('_labels', [])}")

        neighbors = storage.get_neighbors(node_id)
        if neighbors:
            click.echo(f"\nNeighbors ({len(neighbors)}):")
            for n in neighbors:
                rel = n["relation"]
                direction = "->" if n["direction"] == "out" else "<-"
                other = n["node"].get("name", "Unknown")
                click.echo(f"  {direction} [{rel}] {other} ({n['node'].get('id')})")
    finally:
        storage.close()


@cli.command("list-pending")
@click.argument("vault_path", type=click.Path(exists=True, path_type=Path))
@click.option("--db", "-d", type=click.Path(path_type=Path), default="data/vectors.db")
@click.option(
    "--format",
    "-f",
    type=click.Choice(["table", "paths"]),
    default="table",
    help="Output format: table (human) or paths (one per line, for piping)",
)
def list_pending(vault_path: Path, db: Path, format: str):
    """List notes that need knowledge graph extraction."""
    from .extraction.tracker import NoteTracker
    from .vector.store import VectorStore

    tracker = NoteTracker(VectorStore(str(db)))
    result = tracker.list_pending_notes(str(vault_path))

    if not result["success"]:
        click.echo(f"Error: {result['error']}", err=True)
        raise SystemExit(1)

    if format == "paths":
        for note in result["pending"]:
            click.echo(note["path"])
    else:
        click.echo(
            f"Pending: {result['pending_count']}  Unchanged: {result['unchanged_count']}"
        )
        for note in result["pending"]:
            status = "NEW" if note["status"] == "new" else "CHG"
            click.echo(f"  [{status}] {note['path']}")


@cli.command("graph-snapshot")
@click.option(
    "--neo4j-uri", default="bolt://localhost:7687", help="Neo4j connection URI"
)
@click.option("--neo4j-user", default="neo4j", help="Neo4j username")
@click.option("--neo4j-password", default="obsidian", help="Neo4j password")
@click.option(
    "--history-dir",
    type=click.Path(path_type=Path),
    default="data/graph-history",
    help="Directory where graph snapshots are stored",
)
@click.option("--message", "-m", default="", help="Snapshot message")
@click.option(
    "--git-commit",
    is_flag=True,
    help="Also commit snapshot files with git",
)
def graph_snapshot(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    history_dir: Path,
    message: str,
    git_commit: bool,
):
    """Create a versioned snapshot of the live Neo4j graph."""
    from .graph.history import SnapshotRepository

    storage = Neo4jStorage(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    try:
        payload = storage.export_snapshot()
    finally:
        storage.close()

    repo = SnapshotRepository(history_dir)
    meta = repo.create(payload, message=message)

    click.echo(f"Snapshot: {meta.snapshot_id}")
    click.echo(f"Nodes: {meta.nodes}  Edges: {meta.edges}")
    click.echo(f"File: {meta.file}")

    if git_commit:
        root_ok = subprocess.run(
            ["git", "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
        )
        if root_ok.returncode != 0:
            click.echo("Not a git repository, skipped git commit.")
            return

        subprocess.run(["git", "add", str(history_dir)], check=True)
        msg = f"graph snapshot {meta.snapshot_id}"
        if message:
            msg += f" - {message}"
        commit = subprocess.run(
            ["git", "commit", "-m", msg], capture_output=True, text=True
        )
        if commit.returncode == 0:
            click.echo("Snapshot committed to git.")
        else:
            stderr = (commit.stderr or "").strip()
            if "nothing to commit" in stderr.lower():
                click.echo("No changes to commit.")
            else:
                raise SystemExit(f"Git commit failed: {stderr or 'unknown error'}")


@cli.command("graph-log")
@click.option(
    "--history-dir",
    type=click.Path(path_type=Path),
    default="data/graph-history",
    help="Directory where graph snapshots are stored",
)
@click.option("--limit", type=int, default=20, help="How many snapshots to show")
def graph_log(history_dir: Path, limit: int):
    """Show snapshot history (git log equivalent for graph state)."""
    from .graph.history import SnapshotRepository

    repo = SnapshotRepository(history_dir)
    rows = repo.list(limit=limit)
    if not rows:
        click.echo("No snapshots yet.")
        return

    for row in rows:
        message = row.message or "(no message)"
        click.echo(f"{row.snapshot_id}  nodes={row.nodes} edges={row.edges}  {message}")


@cli.command("graph-diff")
@click.argument("old_ref", type=str)
@click.argument("new_ref", type=str)
@click.option(
    "--history-dir",
    type=click.Path(path_type=Path),
    default="data/graph-history",
    help="Directory where graph snapshots are stored",
)
def graph_diff(old_ref: str, new_ref: str, history_dir: Path):
    """Diff two graph snapshots."""
    from .graph.history import SnapshotRepository, diff_snapshots

    repo = SnapshotRepository(history_dir)
    try:
        old_snapshot = repo.load(old_ref)
        new_snapshot = repo.load(new_ref)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc
    diff = diff_snapshots(old_snapshot, new_snapshot)

    summary = diff["summary"]
    click.echo(
        " ".join(
            [
                f"added_nodes={summary['added_nodes']}",
                f"removed_nodes={summary['removed_nodes']}",
                f"updated_nodes={summary['updated_nodes']}",
                f"added_edges={summary['added_edges']}",
                f"removed_edges={summary['removed_edges']}",
                f"updated_edges={summary['updated_edges']}",
            ]
        )
    )


@cli.command("graph-restore")
@click.argument("snapshot_ref", type=str)
@click.option(
    "--neo4j-uri", default="bolt://localhost:7687", help="Neo4j connection URI"
)
@click.option("--neo4j-user", default="neo4j", help="Neo4j username")
@click.option("--neo4j-password", default="obsidian", help="Neo4j password")
@click.option(
    "--history-dir",
    type=click.Path(path_type=Path),
    default="data/graph-history",
    help="Directory where graph snapshots are stored",
)
@click.option(
    "--apply",
    is_flag=True,
    help="Actually restore. Without this flag, only show what would happen.",
)
@click.option(
    "--clear-first/--no-clear-first",
    default=True,
    help="Clear current graph before import",
)
def graph_restore(
    snapshot_ref: str,
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    history_dir: Path,
    apply: bool,
    clear_first: bool,
):
    """Restore graph state from a snapshot."""
    from .graph.history import SnapshotRepository

    repo = SnapshotRepository(history_dir)
    try:
        snapshot = repo.load(snapshot_ref)
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(str(exc)) from exc

    node_count = len(snapshot.get("nodes", []))
    edge_count = len(snapshot.get("edges", []))
    click.echo(f"Snapshot {snapshot_ref}: nodes={node_count} edges={edge_count}")

    if not apply:
        click.echo("Dry run only. Re-run with --apply to restore.")
        return

    if clear_first:
        typed = click.prompt(
            "This will DELETE ALL current graph data. Type RESTORE to continue",
            default="",
        )
        if typed != "RESTORE":
            click.echo("Aborted restore.")
            raise SystemExit(1)

    storage = Neo4jStorage(uri=neo4j_uri, user=neo4j_user, password=neo4j_password)
    try:
        result = storage.import_snapshot(snapshot, clear_first=clear_first)
        stats = storage.get_stats()
    finally:
        storage.close()

    click.echo(
        f"Restored nodes={result['nodes']} edges={result['edges']} -> db nodes={stats['nodes']} rels={stats['relationships']}"
    )


@cli.command("mcp-server")
@click.option(
    "--neo4j-uri", default="bolt://localhost:7687", help="Neo4j connection URI"
)
@click.option("--neo4j-user", default="neo4j", help="Neo4j username")
@click.option("--neo4j-password", default="obsidian", help="Neo4j password")
@click.option(
    "--vector-db",
    type=click.Path(path_type=Path),
    default="data/vectors.db",
    help="Path to vector database (legacy, for extraction tracking only)",
)
@click.option(
    "--transport",
    type=click.Choice(["stdio", "sse"]),
    default="stdio",
    help="MCP transport type",
)
def mcp_server(
    neo4j_uri: str,
    neo4j_user: str,
    neo4j_password: str,
    vector_db: Path,
    transport: str,
):
    """Run the Graph Manipulator MCP server.

    This server exposes knowledge graph CRUD operations as MCP tools.
    Used by Graph Agent (subagent) for runtime memory manipulation.
    """
    from .mcp import init_server, mcp

    click.echo(f"Connecting to Neo4j: {neo4j_uri}")

    init_server(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        vector_db=str(vector_db),
    )

    click.echo(f"Starting MCP server ({transport} transport)...")

    if transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    else:
        asyncio.run(mcp.run_sse_async())


if __name__ == "__main__":
    cli()
