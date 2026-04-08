"""Web-based graph visualization using pyvis."""

from pathlib import Path

from pyvis.network import Network

from .builder import VaultGraph


# Color scheme for node types
NODE_COLORS = {
    "Note": "#FCE38A",  # Yellow
    "Tag": "#95E1D3",  # Light green
    "Folder": "#AA96DA",  # Purple
    "Person": "#FF6B6B",  # Red
    "Concept": "#4ECDC4",  # Teal
    "Tool": "#45B7D1",  # Blue
    "Project": "#F38181",  # Coral
    "Organization": "#DDA0DD",  # Plum
}

NODE_SIZES = {
    "Note": 15,
    "Tag": 20,
    "Folder": 25,
    "Person": 20,
    "Concept": 15,
    "Tool": 15,
    "Project": 20,
    "Organization": 20,
}


def create_web_visualization(
    graph: VaultGraph,
    output_path: Path = Path("output/graph.html"),
    height: str = "900px",
    width: str = "100%",
    max_nodes: int | None = None,
    filter_edge_types: list[str] | None = None,
) -> Path:
    """Create an interactive web visualization of the graph.

    Args:
        graph: VaultGraph instance
        output_path: Where to save the HTML file
        height: Height of the visualization
        width: Width of the visualization
        max_nodes: Limit number of nodes (for large graphs)
        filter_edge_types: Only show these edge types (e.g., ["wikilink"])

    Returns:
        Path to the generated HTML file
    """
    net = Network(
        height=height,
        width=width,
        bgcolor="#1a1a2e",
        font_color="white",
        directed=True,
        select_menu=True,
        filter_menu=True,
    )

    # Physics settings for better layout
    net.set_options("""
    {
        "physics": {
            "forceAtlas2Based": {
                "gravitationalConstant": -100,
                "centralGravity": 0.01,
                "springLength": 200,
                "springConstant": 0.02
            },
            "solver": "forceAtlas2Based",
            "stabilization": {
                "iterations": 100
            }
        },
        "nodes": {
            "font": {
                "size": 12,
                "face": "arial"
            }
        },
        "edges": {
            "color": {
                "inherit": true
            },
            "smooth": {
                "type": "continuous"
            }
        },
        "interaction": {
            "hover": true,
            "navigationButtons": true,
            "keyboard": true
        }
    }
    """)

    # Filter nodes if needed
    nodes_to_add = list(graph.graph.nodes(data=True))
    if max_nodes and len(nodes_to_add) > max_nodes:
        # Prioritize by degree (most connected nodes)
        nodes_by_degree = sorted(
            [(n, d, graph.graph.degree(n)) for n, d in nodes_to_add],
            key=lambda x: -x[2],
        )
        nodes_to_add = [(n, d) for n, d, _ in nodes_by_degree[:max_nodes]]
        node_ids = {n for n, _ in nodes_to_add}
    else:
        node_ids = {n for n, _ in nodes_to_add}

    # Add nodes
    for node_id, data in nodes_to_add:
        node_type = data.get("type", "Note")
        title = data.get("name", node_id)

        # Create hover tooltip
        tooltip = f"<b>{title}</b><br>Type: {node_type}"
        if node_type == "Note":
            tooltip += f"<br>Folder: {data.get('folder', 'root')}"

        net.add_node(
            node_id,
            label=title[:30] + "..." if len(title) > 30 else title,
            title=tooltip,
            color=NODE_COLORS.get(node_type, "#888888"),
            size=NODE_SIZES.get(node_type, 15),
            group=node_type,
        )

    # Add edges
    for source, target, data in graph.graph.edges(data=True):
        # Skip if nodes not in filtered set
        if source not in node_ids or target not in node_ids:
            continue

        edge_type = data.get("type", "unknown")

        # Filter by edge type if specified
        if filter_edge_types and edge_type not in filter_edge_types:
            continue

        # Edge styling by type
        if edge_type == "wikilink":
            color = "#FFD93D"
            width = 2
        elif edge_type == "tagged_with":
            color = "#6BCB77"
            width = 1
        elif edge_type == "in_folder":
            color = "#9D4EDD"
            width = 1
        else:
            color = "#888888"
            width = 1

        net.add_edge(
            source,
            target,
            title=edge_type,
            color=color,
            width=width,
        )

    # Save
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    net.save_graph(str(output_path))

    return output_path
