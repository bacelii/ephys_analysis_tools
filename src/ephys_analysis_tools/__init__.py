"""Functional connectivity analysis tools for electrophysiology data."""

from .connectivity import (
    ConnectivityResult,
    compute_connectivity,
    compute_connectivity_matrix,
    threshold_connectivity,
)
from .graphs import graph_from_connectivity, graph_from_connectivity_result
from .metrics import (
    compare_connectivity_matrices,
    compare_graphs,
    compare_metric_dicts,
    compute_graph_metrics,
    compute_node_metrics,
)
from .timeseries import coerce_neuron_traces

__all__ = [
    "ConnectivityResult",
    "coerce_neuron_traces",
    "compute_connectivity",
    "compute_connectivity_matrix",
    "threshold_connectivity",
    "graph_from_connectivity",
    "graph_from_connectivity_result",
    "compute_graph_metrics",
    "compute_node_metrics",
    "compare_metric_dicts",
    "compare_connectivity_matrices",
    "compare_graphs",
]
