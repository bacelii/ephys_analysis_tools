import math

import networkx as nx
import numpy as np
import pytest

from ephys_analysis_tools.metrics import (
    compare_connectivity_matrices,
    compare_graphs,
    compare_metric_dicts,
    compute_graph_metrics,
    compute_node_metrics,
)


def test_compute_graph_metrics_on_triangle():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=1.0)
    graph.add_edge("b", "c", weight=1.0)
    graph.add_edge("a", "c", weight=1.0)

    metrics = compute_graph_metrics(graph)

    assert metrics["n_nodes"] == 3
    assert metrics["n_edges"] == 3
    assert metrics["density"] == pytest.approx(1.0)
    assert metrics["average_clustering"] == pytest.approx(1.0)
    assert metrics["global_efficiency"] == pytest.approx(1.0)
    assert metrics["largest_component_fraction"] == pytest.approx(1.0)


def test_compute_graph_metrics_on_path_has_expected_diameter():
    graph = nx.path_graph(["a", "b", "c"])
    nx.set_edge_attributes(graph, 1.0, "weight")

    metrics = compute_graph_metrics(graph)

    assert metrics["n_edges"] == 2
    assert metrics["diameter_lcc"] == pytest.approx(2.0)
    assert metrics["average_clustering"] == pytest.approx(0.0)
    assert metrics["degree_max"] == pytest.approx(2.0)


def test_compute_node_metrics_returns_expected_strengths():
    graph = nx.Graph()
    graph.add_edge("a", "b", weight=0.5)
    graph.add_edge("a", "c", weight=0.25)

    node_metrics = compute_node_metrics(graph)

    assert node_metrics["a"]["degree"] == pytest.approx(2.0)
    assert node_metrics["a"]["strength"] == pytest.approx(0.75)
    assert node_metrics["b"]["degree"] == pytest.approx(1.0)


def test_compare_metric_dicts_reports_delta_as_comparison_minus_reference():
    comparison = compare_metric_dicts(
        {"density": 0.25, "n_edges": 2},
        {"density": 0.5, "n_edges": 3},
    )

    assert comparison["density"]["delta"] == pytest.approx(0.25)
    assert comparison["density"]["relative_delta"] == pytest.approx(1.0)
    assert comparison["n_edges"]["delta"] == pytest.approx(1.0)


def test_compare_connectivity_matrices_reports_overlap_and_difference():
    a = np.array([
        [0.0, 0.8, 0.0],
        [0.8, 0.0, 0.1],
        [0.0, 0.1, 0.0],
    ])
    b = np.array([
        [0.0, 0.8, 0.2],
        [0.8, 0.0, 0.0],
        [0.2, 0.0, 0.0],
    ])

    comparison = compare_connectivity_matrices(a, b, threshold=0.05)

    assert comparison["n_edges_reference"] == 2
    assert comparison["n_edges_comparison"] == 2
    assert comparison["edge_jaccard"] == pytest.approx(1 / 3)
    assert comparison["mean_absolute_difference"] > 0


def test_compare_graphs_smoke_test():
    g1 = nx.path_graph(["a", "b", "c"])
    g2 = nx.complete_graph(["a", "b", "c"])
    nx.set_edge_attributes(g1, 1.0, "weight")
    nx.set_edge_attributes(g2, 1.0, "weight")

    comparison = compare_graphs(g1, g2)

    assert comparison["n_edges"]["delta"] == pytest.approx(1.0)
    assert comparison["density"]["delta"] > 0


def test_empty_graph_metrics_do_not_crash():
    graph = nx.Graph()

    metrics = compute_graph_metrics(graph)

    assert metrics["n_nodes"] == 0
    assert metrics["n_edges"] == 0
    assert math.isnan(metrics["global_efficiency"])
