import numpy as np
import pytest

from ephys_analysis_tools.graphs import graph_from_connectivity


def test_graph_from_connectivity_adds_nodes_and_thresholded_edges():
    matrix = np.array([
        [0.0, 0.8, 0.1],
        [0.8, 0.0, -0.7],
        [0.1, -0.7, 0.0],
    ])

    graph = graph_from_connectivity(
        matrix,
        names=["a", "b", "c"],
        threshold=0.5,
        use_absolute_weights=True,
    )

    assert set(graph.nodes) == {"a", "b", "c"}
    assert graph.number_of_edges() == 2
    assert graph["a"]["b"]["weight"] == pytest.approx(0.8)
    assert graph["b"]["c"]["weight"] == pytest.approx(0.7)
    assert graph["b"]["c"]["signed_weight"] == pytest.approx(-0.7)


def test_graph_from_connectivity_density_keeps_top_edge():
    matrix = np.array([
        [0.0, 0.8, 0.1],
        [0.8, 0.0, 0.7],
        [0.1, 0.7, 0.0],
    ])

    graph = graph_from_connectivity(matrix, names=["a", "b", "c"], density=1 / 3)

    assert graph.number_of_edges() == 1
    assert graph.has_edge("a", "b")


def test_graph_from_connectivity_rejects_wrong_number_of_names():
    with pytest.raises(ValueError, match="names"):
        graph_from_connectivity(np.eye(3), names=["a", "b"])
