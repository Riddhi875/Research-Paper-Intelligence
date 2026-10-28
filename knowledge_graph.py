import networkx as nx
from collections import Counter, defaultdict
from itertools import combinations
import plotly.graph_objects as go

# Optional matplotlib fallback
try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

# Try importing spacy safely
try:
    import spacy
    nlp = spacy.load("en_core_web_sm")
except Exception:
    nlp = None


NODE_COLORS = {
    "Paper": "#4f46e5",       # Indigo
    "Method": "#059669",      # Emerald
    "Dataset": "#d97706",     # Amber
    "Finding": "#0284c7",     # Sky
    "Limitation": "#e11d48",  # Rose
    "Concept": "#8b5cf6",     # Violet
}


def build_multipaper_knowledge_graph(papers: list, max_concepts_per_paper: int = 8) -> nx.Graph:
    """
    Construct a heterogeneous evidence-linked research knowledge graph across multiple papers.
    Includes Papers, Methods, Datasets, Findings, and Limitations.
    Edges are added only when supported by extracted evidence.
    """
    graph = nx.Graph()

    for paper in papers:
        paper_name = paper.get("filename", "Paper")
        graph.add_node(
            paper_name,
            label=paper_name,
            type="Paper",
            color=NODE_COLORS["Paper"],
            size=26,
            evidence=[{"paper_name": paper_name, "page_number": 1, "text": f"Source document: {paper_name}"}]
        )

        dimensions = paper.get("dimensions", {})

        # 1. Connect Methods
        if "Methodology" in dimensions and dimensions["Methodology"]["evidence"]:
            top_ev = dimensions["Methodology"]["evidence"][0]
            # Extract a concise method label
            method_label = f"Method: {paper_name[:18]}"
            graph.add_node(
                method_label,
                label=method_label,
                type="Method",
                color=NODE_COLORS["Method"],
                size=18,
                evidence=dimensions["Methodology"]["evidence"]
            )
            graph.add_edge(paper_name, method_label, relation="proposes", weight=2)

        # 2. Connect Datasets
        if "Datasets" in dimensions and dimensions["Datasets"]["evidence"]:
            top_ev = dimensions["Datasets"]["evidence"][0]
            dataset_label = f"Dataset/Eval: {paper_name[:18]}"
            graph.add_node(
                dataset_label,
                label=dataset_label,
                type="Dataset",
                color=NODE_COLORS["Dataset"],
                size=18,
                evidence=dimensions["Datasets"]["evidence"]
            )
            graph.add_edge(paper_name, dataset_label, relation="evaluated on", weight=2)

        # 3. Connect Limitations
        if "Limitations" in dimensions and dimensions["Limitations"]["evidence"]:
            lim_label = f"Limitation: {paper_name[:18]}"
            graph.add_node(
                lim_label,
                label=lim_label,
                type="Limitation",
                color=NODE_COLORS["Limitation"],
                size=18,
                evidence=dimensions["Limitations"]["evidence"]
            )
            graph.add_edge(paper_name, lim_label, relation="reports constraint", weight=2)

        # 4. Connect Extracted Key Concepts
        concepts = paper.get("key_concepts", [])[:max_concepts_per_paper]
        for c in concepts:
            c_name = c.title()
            if not graph.has_node(c_name):
                graph.add_node(
                    c_name,
                    label=c_name,
                    type="Concept",
                    color=NODE_COLORS["Concept"],
                    size=14,
                    evidence=[{"paper_name": paper_name, "page_number": 1, "text": f"Key concept identified in {paper_name}"}]
                )
            else:
                graph.nodes[c_name]["evidence"].append({
                    "paper_name": paper_name,
                    "page_number": 1,
                    "text": f"Cross-paper appearance in {paper_name}"
                })
            graph.add_edge(paper_name, c_name, relation="discusses", weight=1)

    return graph


def render_plotly_graph(graph: nx.Graph):
    """
    Render an interactive Plotly network graph visualization.
    Includes hover tooltips, category color codes, and layout positioning.
    """
    if graph.number_of_nodes() < 2:
        return None

    # Compute 2D spring layout
    pos = nx.spring_layout(graph, seed=42, k=1.2, iterations=80)

    edge_x = []
    edge_y = []
    for edge in graph.edges():
        x0, y0 = pos[edge[0]]
        x1, y1 = pos[edge[1]]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        line=dict(width=1.2, color="#cbd5e1"),
        hoverinfo="none",
        mode="lines"
    )

    node_x = []
    node_y = []
    node_text = []
    node_colors = []
    node_sizes = []

    for node in graph.nodes():
        x, y = pos[node]
        node_x.append(x)
        node_y.append(y)
        data = graph.nodes[node]
        node_type = data.get("type", "Concept")
        degree = graph.degree[node]
        ev_count = len(data.get("evidence", []))
        hover_info = (
            f"<b>{node}</b><br>"
            f"Type: {node_type}<br>"
            f"Connections: {degree}<br>"
            f"Evidence Excerpts: {ev_count}"
        )
        node_text.append(hover_info)
        node_colors.append(data.get("color", NODE_COLORS.get(node_type, "#64748b")))
        node_sizes.append(data.get("size", 16) + min(degree * 2, 16))

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers+text",
        text=[n if len(n) <= 22 else n[:20] + ".." for n in graph.nodes()],
        textposition="top center",
        hoverinfo="text",
        hovertext=node_text,
        marker=dict(
            color=node_colors,
            size=node_sizes,
            line=dict(width=1.5, color="#ffffff")
        )
    )

    fig = go.Figure(
        data=[edge_trace, node_trace],
        layout=go.Layout(
            showlegend=False,
            hovermode="closest",
            margin=dict(b=10, l=10, r=10, t=30),
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=580
        )
    )

    return fig


def render_matplotlib_graph(graph: nx.Graph):
    """Fallback static matplotlib visualization for the knowledge graph."""
    if plt is None or graph.number_of_nodes() == 0:
        return None

    fig, ax = plt.subplots(figsize=(10, 7))
    pos = nx.spring_layout(graph, seed=42, k=1.2)
    colors = [graph.nodes[n].get("color", "#6366f1") for n in graph.nodes()]
    sizes = [graph.nodes[n].get("size", 16) * 40 for n in graph.nodes()]

    nx.draw_networkx_nodes(graph, pos, node_color=colors, node_size=sizes, alpha=0.9, ax=ax)
    nx.draw_networkx_edges(graph, pos, alpha=0.3, edge_color="#94a3b8", ax=ax)
    nx.draw_networkx_labels(graph, pos, font_size=8, font_family="sans-serif", ax=ax)

    ax.axis("off")
    fig.tight_layout()
    return fig


def get_graph_metrics(graph: nx.Graph) -> dict:
    """Compute network summary statistics."""
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = round(nx.density(graph), 4) if num_nodes > 1 else 0.0

    centrality = nx.degree_centrality(graph) if num_nodes > 0 else {}
    top_central = sorted(centrality.items(), key=lambda x: x[1], reverse=True)[:5]

    return {
        "nodes": num_nodes,
        "edges": num_edges,
        "density": density,
        "top_central_nodes": top_central,
    }


# Backward-compatible function for single paper analysis
def build_knowledge_graph(text: str, max_concepts: int = 12) -> nx.Graph:
    """Backward-compatible single-paper graph builder."""
    if nlp is None:
        return nx.Graph()

    doc = nlp(text[:150000])
    graph = nx.Graph()
    concept_counts = Counter()
    concept_sentences = defaultdict(list)

    for sentence in doc.sents:
        sentence_concepts = set()
        for chunk in sentence.noun_chunks:
            phrase = " ".join(chunk.text.strip().lower().split())
            if 4 <= len(phrase) <= 35 and len(phrase.split()) <= 4:
                sentence_concepts.add(phrase)

        for phrase in sentence_concepts:
            concept_counts[phrase] += 1
            if len(concept_sentences[phrase]) < 3:
                concept_sentences[phrase].append(sentence.text.strip())

    selected = concept_counts.most_common(max_concepts)
    allowed = {c for c, _ in selected}

    for concept, freq in selected:
        graph.add_node(
            concept,
            label=concept.title(),
            frequency=freq,
            type="Concept",
            color=NODE_COLORS["Concept"],
            evidence=[{"paper_name": "Current Paper", "page_number": 1, "text": s} for s in concept_sentences[concept]]
        )

    for sentence in doc.sents:
        sentence_concepts = set()
        for chunk in sentence.noun_chunks:
            p = " ".join(chunk.text.strip().lower().split())
            if p in allowed:
                sentence_concepts.add(p)
        for first, second in combinations(sorted(sentence_concepts), 2):
            if graph.has_edge(first, second):
                graph[first][second]["weight"] += 1
            else:
                graph.add_edge(first, second, weight=1)

    return graph


def get_graph_data(graph: nx.Graph):
    """Convert graph nodes and edges to list representations."""
    nodes = [
        {
            "id": n,
            "label": data.get("label", n),
            "type": data.get("type", "Concept"),
            "evidence": data.get("evidence", [])
        }
        for n, data in graph.nodes(data=True)
    ]
    edges = [
        {
            "source": u,
            "target": v,
            "weight": data.get("weight", 1)
        }
        for u, v, data in graph.edges(data=True)
    ]
    return nodes, edges


def get_concept_details(graph: nx.Graph, concept: str):
    """Return supporting sentences and related concepts."""
    if concept not in graph:
        return None
    related = sorted(
        graph[concept].items(),
        key=lambda item: item[1].get("weight", 1),
        reverse=True
    )
    return {
        "concept": concept,
        "type": graph.nodes[concept].get("type", "Concept"),
        "evidence": graph.nodes[concept].get("evidence", []),
        "related": [
            {"concept": name, "weight": data.get("weight", 1)}
            for name, data in related
        ]
    }
