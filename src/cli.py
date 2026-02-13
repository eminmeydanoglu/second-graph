"""CLI for obsidian-brain."""

from pathlib import Path

import click

from .parser.vault import parse_vault, get_vault_stats
from .graph.builder import VaultGraph


@click.group()
def cli():
    """Obsidian Brain - Knowledge graph for your vault."""
    pass


@cli.command()
@click.argument("vault_path", type=click.Path(exists=True, path_type=Path))
def stats(vault_path: Path):
    """Show vault statistics."""
    click.echo(f"Scanning vault: {vault_path}")
    vault_stats = get_vault_stats(vault_path)
    click.echo(vault_stats)


@cli.command()
@click.argument("vault_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output", "-o", type=click.Path(path_type=Path), default="data/graph.json"
)
def build(vault_path: Path, output: Path):
    """Build knowledge graph from vault."""
    click.echo(f"Building graph from: {vault_path}")

    # Parse vault and build graph
    graph = VaultGraph()
    note_count = 0

    with click.progressbar(
        parse_vault(vault_path),
        label="Parsing notes",
        show_pos=True,
    ) as notes:
        for note in notes:
            graph.add_note(note)
            note_count += 1

    click.echo(f"Parsed {note_count} notes")

    # Resolve placeholders
    resolved = graph.resolve_placeholders()
    click.echo(f"Resolved {resolved} placeholder links")

    # Show stats
    stats = graph.get_stats()
    click.echo(stats)

    # Save
    graph.save(output)
    click.echo(f"Saved graph to: {output}")


@cli.command()
@click.argument("graph_path", type=click.Path(exists=True, path_type=Path))
@click.argument("note_path", type=str)
def inspect(graph_path: Path, note_path: str):
    """Inspect a note's connections in the graph."""
    graph = VaultGraph.load(graph_path)

    if note_path not in graph.graph:
        click.echo(f"Note not found: {note_path}")
        return

    node_data = graph.graph.nodes[note_path]
    click.echo(f"\n{node_data.get('title', note_path)}")
    click.echo(f"  Path: {note_path}")
    click.echo(f"  Folder: {node_data.get('folder', 'root')}")

    outlinks = graph.get_outlinks(note_path)
    if outlinks:
        click.echo(f"\n  Outlinks ({len(outlinks)}):")
        for link in outlinks[:10]:
            click.echo(f"    → {link}")
        if len(outlinks) > 10:
            click.echo(f"    ... and {len(outlinks) - 10} more")

    backlinks = graph.get_backlinks(note_path)
    if backlinks:
        click.echo(f"\n  Backlinks ({len(backlinks)}):")
        for link in backlinks[:10]:
            click.echo(f"    ← {link}")
        if len(backlinks) > 10:
            click.echo(f"    ... and {len(backlinks) - 10} more")

    tags = graph.get_tags(note_path)
    if tags:
        click.echo(f"\n  Tags: {', '.join(tags)}")


@cli.command("neo4j-import")
@click.argument("graph_path", type=click.Path(exists=True, path_type=Path))
@click.option("--uri", default="bolt://localhost:7687", help="Neo4j URI")
@click.option("--user", default="neo4j", help="Neo4j username")
@click.option("--password", default="obsidian", help="Neo4j password")
@click.option("--clear", is_flag=True, help="Clear existing data before import")
def neo4j_import(graph_path: Path, uri: str, user: str, password: str, clear: bool):
    """Import graph into Neo4j database."""
    from .graph.neo4j_storage import Neo4jStorage

    click.echo(f"Loading graph from: {graph_path}")
    graph = VaultGraph.load(graph_path)
    click.echo(f"Graph: {graph.get_stats()}")

    click.echo(f"Connecting to Neo4j: {uri}")
    storage = Neo4jStorage(uri=uri, user=user, password=password)

    if clear:
        click.echo("Clearing existing data...")
        storage.clear()

    click.echo("Importing nodes and relationships...")
    stats = storage.import_vault_graph(graph)
    click.echo(
        f"Imported: {stats['nodes']} nodes, {stats['relationships']} relationships"
    )

    db_stats = storage.get_stats()
    click.echo(f"Database stats: {db_stats}")

    storage.close()
    click.echo("\nNeo4j Browser: http://localhost:7474")
    click.echo("Login: neo4j / obsidian")


@cli.command()
@click.argument("vault_path", type=click.Path(exists=True, path_type=Path))
@click.option("--db", "-d", type=click.Path(path_type=Path), default="data/vectors.db")
@click.option(
    "--batch-size", "-b", type=int, default=50, help="Batch size for embedding API"
)
def embed(vault_path: Path, db: Path, batch_size: int):
    """Generate embeddings for all notes in vault."""
    from tqdm import tqdm
    from .vector import Embedder, VectorStore

    click.echo(f"Embedding vault: {vault_path}")

    # Collect all notes
    notes = list(parse_vault(vault_path))
    click.echo(f"Found {len(notes)} notes")

    # Initialize
    embedder = Embedder()
    store = VectorStore(db)

    # Prepare texts for batch embedding
    texts = []
    metadata = []
    for note in notes:
        # Use title + content for embedding
        text = f"# {note.title}\n\n{note.content}"
        texts.append(text)
        metadata.append(
            {
                "path": str(note.path),
                "title": note.title,
                "content": note.content[:2000],  # Store truncated content
            }
        )

    # Batch embed with progress bar
    click.echo("Generating embeddings...")
    embeddings = list(
        tqdm(
            embedder.embed_batch(texts, batch_size=batch_size),
            total=len(texts),
            desc="Embedding",
        )
    )

    # Store
    click.echo("Storing vectors...")
    for i, (emb, meta) in enumerate(zip(embeddings, metadata)):
        store.add(
            path=meta["path"],
            title=meta["title"],
            content=meta["content"],
            embedding=emb,
        )

    click.echo(f"Stored {store.count()} vectors in {db}")


@cli.command()
@click.argument("query", type=str)
@click.option(
    "--db",
    "-d",
    type=click.Path(exists=True, path_type=Path),
    default="data/vectors.db",
)
@click.option("--limit", "-n", type=int, default=5, help="Number of results")
def search(query: str, db: Path, limit: int):
    """Search notes by semantic similarity."""
    from .vector import Embedder, VectorStore

    embedder = Embedder()
    store = VectorStore(db)

    click.echo(f"Searching for: {query}")
    query_embedding = embedder.embed(query)

    results = store.search(query_embedding, limit=limit)

    click.echo(f"\nFound {len(results)} results:\n")
    for i, r in enumerate(results, 1):
        click.echo(f"{i}. [{r['score']:.3f}] {r['title']}")
        click.echo(f"   Path: {r['path']}")
        click.echo(f"   {r['content'][:200]}...")
        click.echo()


@cli.command("merge-extractions")
@click.argument("graph_path", type=click.Path(exists=True, path_type=Path))
@click.argument("extractions_path", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    default=None,
    help="Output path for merged graph (default: overwrite input)",
)
def merge_extractions(graph_path: Path, extractions_path: Path, output: Path | None):
    """Merge extracted entities and relationships into the graph."""
    from .extraction import ExtractionReader, ExtractionMerger

    # Show extraction stats
    click.echo(f"Reading extractions from: {extractions_path}")
    reader = ExtractionReader(extractions_path)
    ext_stats = reader.get_stats()
    click.echo(f"  Total extractions: {ext_stats['total_extractions']}")
    click.echo(f"  With note path: {ext_stats['with_path']}")
    click.echo(f"  Entities: {ext_stats['total_entities']}")
    click.echo(f"  Relationships: {ext_stats['total_relationships']}")

    # Load graph
    click.echo(f"\nLoading graph from: {graph_path}")
    graph = VaultGraph.load(graph_path)
    before_stats = graph.get_stats()
    click.echo(f"  Before: {before_stats.nodes} nodes, {before_stats.edges} edges")

    # Merge
    click.echo("\nMerging extractions...")
    merger = ExtractionMerger(graph)
    extractions = reader.read_all()
    merge_stats = merger.merge_all(extractions)
    click.echo(merge_stats)

    # After stats
    after_stats = graph.get_stats()
    click.echo(f"\nAfter merge:")
    click.echo(after_stats)

    # Save
    output_path = output or graph_path
    graph.save(output_path)
    click.echo(f"\nSaved merged graph to: {output_path}")


@cli.command("extraction-stats")
@click.argument("extractions_path", type=click.Path(exists=True, path_type=Path))
def extraction_stats(extractions_path: Path):
    """Show statistics about extraction file."""
    from .extraction import ExtractionReader

    reader = ExtractionReader(extractions_path)
    stats = reader.get_stats()

    click.echo(f"Extraction Stats for: {extractions_path}\n")
    click.echo(f"Total extractions: {stats['total_extractions']}")
    click.echo(f"  With path: {stats['with_path']}")
    click.echo(f"  Without path: {stats['without_path']}")
    click.echo(f"\nTotal entities: {stats['total_entities']}")
    click.echo(f"Total relationships: {stats['total_relationships']}")

    click.echo("\nEntity types:")
    for etype, count in sorted(stats["entity_types"].items(), key=lambda x: -x[1]):
        click.echo(f"  {etype}: {count}")

    click.echo("\nRelation types:")
    for rtype, count in sorted(stats["relation_types"].items(), key=lambda x: -x[1]):
        click.echo(f"  {rtype}: {count}")


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
    help="Path to vector database",
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
    click.echo(f"Vector DB: {vector_db}")

    init_server(
        neo4j_uri=neo4j_uri,
        neo4j_user=neo4j_user,
        neo4j_password=neo4j_password,
        vector_db=str(vector_db),
    )

    click.echo(f"Starting MCP server ({transport} transport)...")
    click.echo(
        "Available tools: add_node, get_node, find_node, update_node, delete_node,"
    )
    click.echo("                 add_edge, get_edge, delete_edge, invalidate_edge,")
    click.echo(
        "                 get_neighbors, find_path, search_entities, get_schema, get_stats"
    )

    import asyncio

    if transport == "stdio":
        asyncio.run(mcp.run_stdio_async())
    else:
        asyncio.run(mcp.run_sse_async())


if __name__ == "__main__":
    cli()
