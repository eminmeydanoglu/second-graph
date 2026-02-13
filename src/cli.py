"""CLI for obsidian-brain."""

import asyncio
from pathlib import Path

import click

from .graph.scanner import VaultScanner
from .graph.sync import NoteSynchronizer
from .graph.neo4j_storage import Neo4jStorage
from .vector.embedder import Embedder


@click.group()
def cli():
    """Obsidian Brain - Knowledge graph for your vault."""
    pass


@cli.command("sync-vault")
@click.argument("vault_path", type=click.Path(exists=True, path_type=Path))
@click.option("--uri", default="bolt://localhost:7687", help="Neo4j URI")
@click.option("--user", default="neo4j", help="Neo4j username")
@click.option("--password", default="obsidian", help="Neo4j password")
def sync_vault(vault_path: Path, uri: str, user: str, password: str):
    """Synchronize entire vault with Neo4j graph."""
    click.echo(f"Syncing vault: {vault_path} -> Neo4j ({uri})")

    storage = Neo4jStorage(uri=uri, user=user, password=password)
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

    click.echo(f"Searching for: {query}")
    query_embedding = embedder.embed(query)
    results = storage.search_similar(query_embedding, limit=limit)

    click.echo(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        click.echo(f"{i}. [{r['score']:.3f}] {r['name']} ({r['node_type']})")
        if r.get("summary"):
            click.echo(f"   {r['summary'][:200]}...")
        click.echo()

    storage.close()


@cli.command()
@click.argument("node_id", type=str)
@click.option("--uri", default="bolt://localhost:7687", help="Neo4j URI")
@click.option("--user", default="neo4j", help="Neo4j username")
@click.option("--password", default="obsidian", help="Neo4j password")
def inspect(node_id: str, uri: str, user: str, password: str):
    """Inspect a node's connections in the graph."""
    storage = Neo4jStorage(uri=uri, user=user, password=password)

    result = storage.get_node(node_id)
    if not result:
        click.echo(f"Node not found: {node_id}")
        storage.close()
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
