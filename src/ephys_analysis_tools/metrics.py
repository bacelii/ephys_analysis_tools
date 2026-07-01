"""Network statistics and comparison utilities for connectivity graphs."""

from __future__ import annotations

from collections.abc import Mapping
import math
import warnings

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike


def compute_graph_metrics(
    graph: nx.Graph,
    *,
    weight: str = "weight",
    distance: str = "distance",
    compute_communities: bool = True,
) -> dict[str, float | int]:
    """Compute graph-level network metrics for a connectivity graph.

    Connectivity edge weights usually mean "stronger is closer." For weighted
    path-based metrics, this function creates a positive-weight copy of the graph
    with ``distance = 1 / weight`` on each edge.
    """
    g = _positive_weight_graph(graph, weight=weight, distance=distance)
    n_nodes = g.number_of_nodes()
    n_edges = g.number_of_edges()

    metrics: dict[str, float | int] = {
        "n_nodes": n_nodes,
        "n_edges": n_edges,
        "density": float(nx.density(g)) if n_nodes > 1 else 0.0,
    }

    degrees = np.array([degree for _, degree in g.degree()], dtype=float)
    strengths = np.array([degree for _, degree in g.degree(weight=weight)], dtype=float)
    edge_weights = np.array(
        [data.get(weight, 1.0) for _, _, data in g.edges(data=True)],
        dtype=float,
    )

    metrics.update(
        {
            "degree_mean": _mean_or_nan(degrees),
            "degree_max": _max_or_nan(degrees),
            "strength_mean": _mean_or_nan(strengths),
            "strength_max": _max_or_nan(strengths),
            "edge_weight_mean": _mean_or_nan(edge_weights),
            "edge_weight_std": _std_or_nan(edge_weights),
        }
    )

    if n_nodes == 0:
        metrics.update(_empty_graph_metrics())
        return metrics

    components = list(nx.connected_components(g))
    n_components = len(components)
    largest_component = max(components, key=len) if components else set()
    largest_component_fraction = len(largest_component) / n_nodes if n_nodes else math.nan
    lcc = g.subgraph(largest_component).copy()

    metrics.update(
        {
            "n_connected_components": n_components,
            "largest_component_fraction": float(largest_component_fraction),
            "average_clustering": float(nx.average_clustering(g)),
            "average_clustering_weighted": float(nx.average_clustering(g, weight=weight)),
            "transitivity": float(nx.transitivity(g)),
            "global_efficiency": float(nx.global_efficiency(g)) if n_nodes > 1 else 0.0,
            "weighted_global_efficiency": float(_weighted_global_efficiency(g, distance=distance)),
            "average_shortest_path_length_lcc": _average_shortest_path_length(lcc),
            "weighted_average_shortest_path_length_lcc": _average_shortest_path_length(
                lcc, weight=distance
            ),
            "diameter_lcc": _diameter(lcc),
            "degree_assortativity": _degree_assortativity(g),
        }
    )

    metrics.update(_centrality_summaries(g, distance=distance))

    if compute_communities:
        metrics.update(_community_metrics(g, weight=weight))

    return metrics


def compute_node_metrics(
    graph: nx.Graph,
    *,
    weight: str = "weight",
    distance: str = "distance",
) -> dict[str, dict[str, float]]:
    """Compute node-level metrics keyed by node name."""
    g = _positive_weight_graph(graph, weight=weight, distance=distance)
    degree_centrality = nx.degree_centrality(g) if g.number_of_nodes() else {}
    betweenness = nx.betweenness_centrality(g, weight=distance, normalized=True)
    closeness = nx.closeness_centrality(g, distance=distance)
    clustering = nx.clustering(g, weight=weight)

    node_metrics: dict[str, dict[str, float]] = {}
    for node in g.nodes:
        node_metrics[str(node)] = {
            "degree": float(g.degree(node)),
            "strength": float(g.degree(node, weight=weight)),
            "degree_centrality": float(degree_centrality.get(node, math.nan)),
            "betweenness_centrality": float(betweenness.get(node, math.nan)),
            "closeness_centrality": float(closeness.get(node, math.nan)),
            "clustering": float(clustering.get(node, math.nan)),
        }
    return node_metrics


def compare_metric_dicts(
    reference: Mapping[str, float | int],
    comparison: Mapping[str, float | int],
) -> dict[str, dict[str, float]]:
    """Compare two graph-level metric dictionaries.

    The returned mapping is keyed by metric name. Each value contains
    ``reference``, ``comparison``, ``delta`` and ``relative_delta``. The delta is
    ``comparison - reference``.
    """
    result: dict[str, dict[str, float]] = {}
    for key in sorted(set(reference) & set(comparison)):
        a = reference[key]
        b = comparison[key]
        if not _is_number(a) or not _is_number(b):
            continue
        a_float = float(a)
        b_float = float(b)
        delta = b_float - a_float
        relative_delta = delta / abs(a_float) if a_float != 0 else math.nan
        result[key] = {
            "reference": a_float,
            "comparison": b_float,
            "delta": delta,
            "relative_delta": relative_delta,
        }
    return result


def compare_connectivity_matrices(
    reference: ArrayLike,
    comparison: ArrayLike,
    *,
    threshold: float = 0.0,
) -> dict[str, float | int]:
    """Compare two connectivity matrices using upper-triangle values.

    Parameters
    ----------
    reference, comparison:
        Square matrices with the same shape.
    threshold:
        Absolute threshold used to define binary edge sets for Jaccard overlap.
    """
    a = np.asarray(reference, dtype=float)
    b = np.asarray(comparison, dtype=float)
    if a.shape != b.shape:
        raise ValueError(f"matrices must have the same shape; got {a.shape} and {b.shape}")
    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError(f"matrices must be square; got shape {a.shape}")
    if not np.all(np.isfinite(a)) or not np.all(np.isfinite(b)):
        raise ValueError("matrices must contain only finite values")

    rows, cols = np.triu_indices(a.shape[0], k=1)
    avec = a[rows, cols]
    bvec = b[rows, cols]
    diff = bvec - avec

    a_norm = float(np.linalg.norm(avec))
    b_norm = float(np.linalg.norm(bvec))
    cosine = float(np.dot(avec, bvec) / (a_norm * b_norm)) if a_norm and b_norm else math.nan
    corr = _vector_correlation(avec, bvec)

    a_edges = np.abs(avec) > threshold
    b_edges = np.abs(bvec) > threshold
    union = np.logical_or(a_edges, b_edges).sum()
    intersection = np.logical_and(a_edges, b_edges).sum()
    jaccard = float(intersection / union) if union else math.nan

    return {
        "n_edges_reference": int(a_edges.sum()),
        "n_edges_comparison": int(b_edges.sum()),
        "edge_jaccard": jaccard,
        "mean_absolute_difference": float(np.mean(np.abs(diff))) if diff.size else 0.0,
        "frobenius_difference": float(np.linalg.norm(b - a)),
        "upper_triangle_correlation": corr,
        "upper_triangle_cosine": cosine,
    }


def compare_graphs(
    reference: nx.Graph,
    comparison: nx.Graph,
    *,
    weight: str = "weight",
) -> dict[str, dict[str, float]]:
    """Compute graph metrics for two graphs and compare them."""
    return compare_metric_dicts(
        compute_graph_metrics(reference, weight=weight),
        compute_graph_metrics(comparison, weight=weight),
    )


def _positive_weight_graph(graph: nx.Graph, *, weight: str, distance: str) -> nx.Graph:
    g = nx.Graph()
    g.add_nodes_from(graph.nodes(data=True))
    for u, v, data in graph.edges(data=True):
        raw_weight = data.get(weight, 1.0)
        try:
            w = float(raw_weight)
        except (TypeError, ValueError):
            continue
        if not np.isfinite(w) or w <= 0:
            continue
        attrs = dict(data)
        attrs[weight] = w
        attrs[distance] = 1.0 / w
        g.add_edge(u, v, **attrs)
    return g


def _weighted_global_efficiency(graph: nx.Graph, *, distance: str) -> float:
    n = graph.number_of_nodes()
    if n < 2:
        return 0.0
    lengths = dict(nx.all_pairs_dijkstra_path_length(graph, weight=distance))
    total = 0.0
    for source in graph.nodes:
        for target in graph.nodes:
            if source == target:
                continue
            d = lengths.get(source, {}).get(target, math.inf)
            if d > 0 and math.isfinite(d):
                total += 1.0 / d
    return total / (n * (n - 1))


def _centrality_summaries(graph: nx.Graph, *, distance: str) -> dict[str, float]:
    if graph.number_of_nodes() == 0:
        return {
            "degree_centrality_mean": math.nan,
            "degree_centrality_max": math.nan,
            "betweenness_centrality_mean": math.nan,
            "betweenness_centrality_max": math.nan,
            "closeness_centrality_mean": math.nan,
            "closeness_centrality_max": math.nan,
        }

    degree_centrality = np.array(list(nx.degree_centrality(graph).values()), dtype=float)
    betweenness = np.array(
        list(nx.betweenness_centrality(graph, weight=distance, normalized=True).values()),
        dtype=float,
    )
    closeness = np.array(
        list(nx.closeness_centrality(graph, distance=distance).values()),
        dtype=float,
    )
    return {
        "degree_centrality_mean": _mean_or_nan(degree_centrality),
        "degree_centrality_max": _max_or_nan(degree_centrality),
        "betweenness_centrality_mean": _mean_or_nan(betweenness),
        "betweenness_centrality_max": _max_or_nan(betweenness),
        "closeness_centrality_mean": _mean_or_nan(closeness),
        "closeness_centrality_max": _max_or_nan(closeness),
    }


def _community_metrics(graph: nx.Graph, *, weight: str) -> dict[str, float | int]:
    if graph.number_of_nodes() == 0:
        return {
            "n_communities": 0,
            "modularity": math.nan,
            "largest_community_fraction": math.nan,
        }
    if graph.number_of_edges() == 0:
        return {
            "n_communities": graph.number_of_nodes(),
            "modularity": 0.0,
            "largest_community_fraction": 1.0 / graph.number_of_nodes(),
        }

    communities = list(nx.community.greedy_modularity_communities(graph, weight=weight))
    largest = max((len(c) for c in communities), default=0)
    modularity = nx.community.modularity(graph, communities, weight=weight)
    return {
        "n_communities": len(communities),
        "modularity": float(modularity),
        "largest_community_fraction": float(largest / graph.number_of_nodes()),
    }


def _empty_graph_metrics() -> dict[str, float | int]:
    return {
        "n_connected_components": 0,
        "largest_component_fraction": math.nan,
        "average_clustering": math.nan,
        "average_clustering_weighted": math.nan,
        "transitivity": math.nan,
        "global_efficiency": math.nan,
        "weighted_global_efficiency": math.nan,
        "average_shortest_path_length_lcc": math.nan,
        "weighted_average_shortest_path_length_lcc": math.nan,
        "diameter_lcc": math.nan,
        "degree_assortativity": math.nan,
        "degree_centrality_mean": math.nan,
        "degree_centrality_max": math.nan,
        "betweenness_centrality_mean": math.nan,
        "betweenness_centrality_max": math.nan,
        "closeness_centrality_mean": math.nan,
        "closeness_centrality_max": math.nan,
        "n_communities": 0,
        "modularity": math.nan,
        "largest_community_fraction": math.nan,
    }


def _average_shortest_path_length(graph: nx.Graph, *, weight: str | None = None) -> float:
    if graph.number_of_nodes() < 2:
        return math.nan
    try:
        return float(nx.average_shortest_path_length(graph, weight=weight))
    except (nx.NetworkXError, nx.NetworkXNoPath):
        return math.nan


def _diameter(graph: nx.Graph) -> float:
    if graph.number_of_nodes() < 2:
        return math.nan
    try:
        return float(nx.diameter(graph))
    except nx.NetworkXError:
        return math.nan


def _degree_assortativity(graph: nx.Graph) -> float:
    if graph.number_of_nodes() < 2 or graph.number_of_edges() == 0:
        return math.nan
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        value = nx.degree_assortativity_coefficient(graph)
    return float(value)


def _vector_correlation(a: np.ndarray, b: np.ndarray) -> float:
    if a.size == 0:
        return math.nan
    if np.std(a) == 0 or np.std(b) == 0:
        return math.nan
    return float(np.corrcoef(a, b)[0, 1])


def _mean_or_nan(values: np.ndarray) -> float:
    return float(np.mean(values)) if values.size else math.nan


def _std_or_nan(values: np.ndarray) -> float:
    return float(np.std(values)) if values.size else math.nan


def _max_or_nan(values: np.ndarray) -> float:
    return float(np.max(values)) if values.size else math.nan


def _is_number(value: object) -> bool:
    return isinstance(value, (int, float, np.integer, np.floating))
