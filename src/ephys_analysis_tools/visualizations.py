from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from . import (
    compute_connectivity,
    graph_from_connectivity_result,
    compute_graph_metrics,
)


def generate_report_visuals(
    neuron_traces: dict[str, np.ndarray],
    output_dir: str | Path = "report_figures",
    connectivity_method: str = "pearson",
    absolute_connectivity: bool = True,
    graph_density: float = 0.15,
    graph_threshold: float | None = None,
    top_n_nodes: int = 25,
    show_node_labels: bool | None = None,
    **connectivity_kwargs: Any,
) -> dict[str, Path]:
    """
    Generate report-ready visuals from one electrophysiology dataset.

    Parameters
    ----------
    neuron_traces:
        Dictionary mapping neuron name to electrophysiology trace.
        Each value should be shape (N,) or squeezeable to shape (N,).

    output_dir:
        Directory where figures and CSV files will be written.

    connectivity_method:
        Connectivity method passed to compute_connectivity.
        Examples: "pearson", "spearman", "cosine",
        "cross_correlation_peak", "coherence".

    absolute_connectivity:
        If True, graph construction uses absolute connection strength.
        This makes strong positive and strong negative relationships both
        appear as strong edges. If False, signed values are preserved.

    graph_density:
        Fraction of strongest possible edges to keep.
        Used only if graph_threshold is None.

    graph_threshold:
        Absolute threshold for keeping edges.
        If provided, this is used instead of graph_density.

    top_n_nodes:
        Number of highest-strength nodes to show in the node summary plot.

    show_node_labels:
        Whether to label nodes in the graph plot.
        If None, labels are shown only for relatively small graphs.

    connectivity_kwargs:
        Extra keyword arguments passed to compute_connectivity.
        For example, coherence might need sampling-rate-related options
        depending on your implementation.

    Returns
    -------
    dict[str, Path]
        Paths to generated files.
    """

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = compute_connectivity(
        neuron_traces,
        method=connectivity_method,
        absolute=absolute_connectivity,
        **connectivity_kwargs,
    )

    if graph_threshold is not None:
        graph = graph_from_connectivity_result(
            result,
            threshold=graph_threshold,
        )
        graph_description = f"threshold={graph_threshold:g}"
    else:
        graph = graph_from_connectivity_result(
            result,
            density=graph_density,
        )
        graph_description = f"density={graph_density:g}"

    metrics = compute_graph_metrics(graph)

    paths: dict[str, Path] = {}

    paths["connectivity_heatmap"] = output_dir / "connectivity_heatmap.png"
    plot_connectivity_heatmap(
        matrix=result.matrix,
        names=result.names,
        output_path=paths["connectivity_heatmap"],
        title=f"Functional connectivity matrix\nmethod={connectivity_method}",
        absolute=absolute_connectivity,
    )

    paths["graph"] = output_dir / "functional_connectivity_graph.png"
    plot_functional_connectivity_graph(
        graph=graph,
        output_path=paths["graph"],
        title=f"Functional connectivity graph\nmethod={connectivity_method}, {graph_description}",
        show_labels=show_node_labels,
    )

    paths["metrics_table"] = output_dir / "network_metrics_table.png"
    plot_metrics_table(
        metrics=metrics,
        output_path=paths["metrics_table"],
        title="Network summary statistics",
    )

    paths["node_degree_strength"] = output_dir / "top_node_degree_strength.png"
    plot_top_node_degree_strength(
        graph=graph,
        output_path=paths["node_degree_strength"],
        top_n=top_n_nodes,
        title=f"Top {top_n_nodes} nodes by weighted strength",
    )

    paths["edge_weight_distribution"] = output_dir / "edge_weight_distribution.png"
    plot_edge_weight_distribution(
        graph=graph,
        output_path=paths["edge_weight_distribution"],
        title="Distribution of retained edge weights",
    )

    paths["density_sensitivity"] = output_dir / "density_sensitivity.png"
    plot_density_sensitivity(
        connectivity_result=result,
        output_path=paths["density_sensitivity"],
        densities=np.linspace(0.05, 0.50, 10),
        title="Network statistics across graph density thresholds",
    )

    paths["network_metrics_csv"] = output_dir / "network_metrics.csv"
    write_metrics_csv(metrics, paths["network_metrics_csv"])

    paths["node_metrics_csv"] = output_dir / "node_metrics.csv"
    write_node_metrics_csv(graph, paths["node_metrics_csv"])

    return paths


def plot_connectivity_heatmap(
    matrix: np.ndarray,
    names: list[str],
    output_path: Path,
    title: str,
    absolute: bool,
) -> None:
    """Plot the functional connectivity matrix as a heatmap."""

    fig, ax = plt.subplots(figsize=(8, 7))

    if absolute:
        image = ax.imshow(matrix, vmin=0, vmax=1)
    else:
        image = ax.imshow(matrix, vmin=-1, vmax=1)

    ax.set_title(title)
    ax.set_xlabel("Neuron")
    ax.set_ylabel("Neuron")

    n = len(names)

    if n <= 30:
        ax.set_xticks(np.arange(n))
        ax.set_yticks(np.arange(n))
        ax.set_xticklabels(names, rotation=90, fontsize=6)
        ax.set_yticklabels(names, fontsize=6)
    else:
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            -0.08,
            f"{n} neurons; labels hidden for readability",
            transform=ax.transAxes,
            ha="center",
            va="top",
        )

    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Connectivity strength")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_functional_connectivity_graph(
    graph: nx.Graph,
    output_path: Path,
    title: str,
    show_labels: bool | None,
) -> None:
    """Plot the thresholded functional connectivity graph."""

    fig, ax = plt.subplots(figsize=(9, 8))
    ax.set_title(title)

    n_nodes = graph.number_of_nodes()
    n_edges = graph.number_of_edges()

    if n_nodes == 0:
        ax.text(0.5, 0.5, "Graph has no nodes", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    if n_edges == 0:
        pos = nx.circular_layout(graph)
    else:
        pos = nx.spring_layout(graph, seed=7, weight="weight")

    degrees = dict(graph.degree())
    strengths = dict(graph.degree(weight="weight"))

    strength_values = np.array(list(strengths.values()), dtype=float)
    if strength_values.size > 0 and np.nanmax(strength_values) > 0:
        node_sizes = [
            150 + 900 * strengths[node] / np.nanmax(strength_values)
            for node in graph.nodes()
        ]
    else:
        node_sizes = [250 for _ in graph.nodes()]

    edge_weights = np.array(
        [data.get("weight", 1.0) for _, _, data in graph.edges(data=True)],
        dtype=float,
    )

    if edge_weights.size > 0 and np.nanmax(edge_weights) > 0:
        edge_widths = [
            0.5 + 4.0 * data.get("weight", 1.0) / np.nanmax(edge_weights)
            for _, _, data in graph.edges(data=True)
        ]
    else:
        edge_widths = 1.0

    nx.draw_networkx_edges(
        graph,
        pos,
        ax=ax,
        width=edge_widths,
        alpha=0.5,
    )

    nx.draw_networkx_nodes(
        graph,
        pos,
        ax=ax,
        node_size=node_sizes,
    )

    if show_labels is None:
        show_labels = n_nodes <= 35

    if show_labels:
        nx.draw_networkx_labels(
            graph,
            pos,
            ax=ax,
            font_size=7,
        )

    summary = (
        f"nodes={n_nodes}, edges={n_edges}, "
        f"density={nx.density(graph):.3f}, "
        f"mean degree={np.mean(list(degrees.values())):.2f}"
    )

    ax.text(
        0.5,
        -0.04,
        summary,
        transform=ax.transAxes,
        ha="center",
        va="top",
    )

    ax.axis("off")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_metrics_table(
    metrics: dict[str, Any],
    output_path: Path,
    title: str,
) -> None:
    """Plot selected graph statistics as a report-ready table."""

    preferred_order = [
        "n_nodes",
        "n_edges",
        "density",
        "degree_mean",
        "degree_max",
        "strength_mean",
        "strength_max",
        "edge_weight_mean",
        "edge_weight_std",
        "n_connected_components",
        "largest_component_fraction",
        "average_clustering",
        "average_clustering_weighted",
        "transitivity",
        "global_efficiency",
        "weighted_global_efficiency",
        "average_shortest_path_length_lcc",
        "weighted_average_shortest_path_length_lcc",
        "diameter_lcc",
        "degree_assortativity",
        "degree_centrality_mean",
        "degree_centrality_max",
        "betweenness_centrality_mean",
        "betweenness_centrality_max",
        "closeness_centrality_mean",
        "closeness_centrality_max",
        "n_communities",
        "modularity",
        "largest_community_fraction",
    ]

    rows = []

    for key in preferred_order:
        if key in metrics and _is_scalar_number(metrics[key]):
            rows.append((pretty_metric_name(key), format_metric(metrics[key])))

    if not rows:
        rows = [("No scalar metrics found", "")]

    fig_height = max(4, 0.35 * len(rows) + 1.4)
    fig, ax = plt.subplots(figsize=(8.5, fig_height))
    ax.set_title(title, pad=16)
    ax.axis("off")

    table = ax.table(
        cellText=rows,
        colLabels=["Metric", "Value"],
        loc="center",
        cellLoc="left",
        colLoc="left",
    )

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.25)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_top_node_degree_strength(
    graph: nx.Graph,
    output_path: Path,
    top_n: int,
    title: str,
) -> None:
    """Plot degree and weighted strength for the strongest nodes."""

    degrees = dict(graph.degree())
    strengths = dict(graph.degree(weight="weight"))

    sorted_nodes = sorted(
        graph.nodes(),
        key=lambda node: strengths.get(node, 0.0),
        reverse=True,
    )

    selected_nodes = sorted_nodes[:top_n]

    if not selected_nodes:
        fig, ax = plt.subplots(figsize=(8, 4))
        ax.text(0.5, 0.5, "No nodes to plot", ha="center", va="center")
        ax.axis("off")
        fig.tight_layout()
        fig.savefig(output_path, dpi=300)
        plt.close(fig)
        return

    x = np.arange(len(selected_nodes))
    degree_values = np.array([degrees[node] for node in selected_nodes], dtype=float)
    strength_values = np.array([strengths[node] for node in selected_nodes], dtype=float)

    fig, ax1 = plt.subplots(figsize=(10, 5))

    ax1.bar(x - 0.2, degree_values, width=0.4, label="Degree")
    ax1.set_ylabel("Degree")

    ax2 = ax1.twinx()
    ax2.bar(x + 0.2, strength_values, width=0.4, label="Weighted strength")
    ax2.set_ylabel("Weighted strength")

    ax1.set_title(title)
    ax1.set_xticks(x)
    ax1.set_xticklabels(selected_nodes, rotation=90, fontsize=7)

    handles_1, labels_1 = ax1.get_legend_handles_labels()
    handles_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(handles_1 + handles_2, labels_1 + labels_2, loc="upper right")

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_edge_weight_distribution(
    graph: nx.Graph,
    output_path: Path,
    title: str,
) -> None:
    """Plot histogram of retained graph edge weights."""

    weights = np.array(
        [data.get("weight", np.nan) for _, _, data in graph.edges(data=True)],
        dtype=float,
    )

    weights = weights[np.isfinite(weights)]

    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title)
    ax.set_xlabel("Edge weight")
    ax.set_ylabel("Count")

    if len(weights) == 0:
        ax.text(0.5, 0.5, "No retained edges", ha="center", va="center")
    else:
        ax.hist(weights, bins=min(30, max(5, int(np.sqrt(len(weights))))))

        ax.axvline(np.mean(weights), linestyle="--", label=f"Mean = {np.mean(weights):.3f}")
        ax.axvline(np.median(weights), linestyle=":", label=f"Median = {np.median(weights):.3f}")
        ax.legend()

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def plot_density_sensitivity(
    connectivity_result: Any,
    output_path: Path,
    densities: np.ndarray,
    title: str,
) -> None:
    """
    Show how several network statistics change as graph density changes.

    This is useful when you only have one dataset because it shows whether
    the network conclusions are stable across threshold choices.
    """

    records = []

    for density in densities:
        graph = graph_from_connectivity_result(
            connectivity_result,
            density=float(density),
        )
        metrics = compute_graph_metrics(graph)

        records.append(
            {
                "density_threshold": float(density),
                "average_clustering": _safe_float(metrics.get("average_clustering")),
                "global_efficiency": _safe_float(metrics.get("global_efficiency")),
                "modularity": _safe_float(metrics.get("modularity")),
                "largest_component_fraction": _safe_float(
                    metrics.get("largest_component_fraction")
                ),
            }
        )

    fig, ax = plt.subplots(figsize=(8, 5))

    x = [row["density_threshold"] for row in records]

    for metric_name in [
        "average_clustering",
        "global_efficiency",
        "modularity",
        "largest_component_fraction",
    ]:
        y = [row[metric_name] for row in records]
        if any(np.isfinite(value) for value in y):
            ax.plot(x, y, marker="o", label=pretty_metric_name(metric_name))

    ax.set_title(title)
    ax.set_xlabel("Retained graph density")
    ax.set_ylabel("Metric value")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def write_metrics_csv(metrics: dict[str, Any], output_path: Path) -> None:
    """Write scalar network metrics to CSV."""

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["metric", "value"])

        for key, value in sorted(metrics.items()):
            if _is_scalar_number(value):
                writer.writerow([key, value])


def write_node_metrics_csv(graph: nx.Graph, output_path: Path) -> None:
    """Write node-level graph metrics to CSV."""

    degree = dict(graph.degree())
    strength = dict(graph.degree(weight="weight"))

    if graph.number_of_nodes() > 0:
        degree_centrality = nx.degree_centrality(graph)
        betweenness_centrality = nx.betweenness_centrality(graph, weight=None)

        if graph.number_of_edges() > 0:
            closeness_centrality = nx.closeness_centrality(graph)
        else:
            closeness_centrality = {node: 0.0 for node in graph.nodes()}
    else:
        degree_centrality = {}
        betweenness_centrality = {}
        closeness_centrality = {}

    with output_path.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "node",
                "degree",
                "weighted_strength",
                "degree_centrality",
                "betweenness_centrality",
                "closeness_centrality",
            ]
        )

        for node in graph.nodes():
            writer.writerow(
                [
                    node,
                    degree.get(node, 0),
                    strength.get(node, 0.0),
                    degree_centrality.get(node, 0.0),
                    betweenness_centrality.get(node, 0.0),
                    closeness_centrality.get(node, 0.0),
                ]
            )


def pretty_metric_name(name: str) -> str:
    """Convert snake_case metric names into readable labels."""

    replacements = {
        "lcc": "largest connected component",
        "n": "number of",
    }

    parts = name.split("_")
    pretty_parts = [replacements.get(part, part) for part in parts]
    return " ".join(pretty_parts).capitalize()


def format_metric(value: Any) -> str:
    """Format numeric metrics for table display."""

    if value is None:
        return "NA"

    try:
        value_float = float(value)
    except TypeError:
        return str(value)

    if not math.isfinite(value_float):
        return "NA"

    if value_float.is_integer():
        return str(int(value_float))

    if abs(value_float) >= 100:
        return f"{value_float:.2f}"

    if abs(value_float) >= 10:
        return f"{value_float:.3f}"

    return f"{value_float:.4f}"


def _is_scalar_number(value: Any) -> bool:
    """Return True if value is a finite scalar numeric value."""

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return False

    return math.isfinite(value_float)


def _safe_float(value: Any) -> float:
    """Convert a value to float, returning NaN if conversion fails."""

    try:
        value_float = float(value)
    except (TypeError, ValueError):
        return float("nan")

    if not math.isfinite(value_float):
        return float("nan")

    return value_float