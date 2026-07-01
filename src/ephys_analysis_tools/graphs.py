"""Graph construction helpers for functional connectivity matrices."""

from __future__ import annotations

from collections.abc import Sequence

import networkx as nx
import numpy as np
from numpy.typing import ArrayLike

from .connectivity import ConnectivityResult, threshold_connectivity


def graph_from_connectivity(
    matrix: ArrayLike,
    *,
    names: Sequence[str] | None = None,
    threshold: float | None = None,
    density: float | None = None,
    absolute_for_threshold: bool = True,
    use_absolute_weights: bool = True,
    weight_attr: str = "weight",
    signed_weight_attr: str = "signed_weight",
) -> nx.Graph:
    """Build an undirected NetworkX graph from a connectivity matrix.

    Parameters
    ----------
    matrix:
        Square connectivity matrix.
    names:
        Node names in row/column order. If omitted, integer nodes are used.
    threshold, density:
        Optional sparsification controls passed to ``threshold_connectivity``.
    absolute_for_threshold:
        If True, choose edges by absolute magnitude.
    use_absolute_weights:
        If True, the main ``weight`` edge attribute is ``abs(connectivity)``.
        The original signed value is always stored under ``signed_weight_attr``.
    weight_attr:
        Edge attribute name used by NetworkX weighted algorithms.
    signed_weight_attr:
        Edge attribute name for the original signed connectivity value.
    """
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square; got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix contains NaN or infinite values")

    n = m.shape[0]
    node_names = list(range(n)) if names is None else list(names)
    if len(node_names) != n:
        raise ValueError(f"names must have length {n}; got {len(node_names)}")

    sparse = threshold_connectivity(
        m,
        threshold=threshold,
        density=density,
        absolute_for_threshold=absolute_for_threshold,
    )

    graph = nx.Graph()
    for idx, name in enumerate(node_names):
        graph.add_node(name, trace_index=idx)

    rows, cols = np.triu_indices(n, k=1)
    for i, j in zip(rows, cols, strict=True):
        signed_value = float(sparse[i, j])
        if signed_value == 0:
            continue
        graph.add_edge(
            node_names[i],
            node_names[j],
            **{
                weight_attr: abs(signed_value) if use_absolute_weights else signed_value,
                signed_weight_attr: signed_value,
            },
        )

    return graph


def graph_from_connectivity_result(
    result: ConnectivityResult,
    *,
    threshold: float | None = None,
    density: float | None = None,
    absolute_for_threshold: bool = True,
    use_absolute_weights: bool = True,
) -> nx.Graph:
    """Build a graph directly from a :class:`ConnectivityResult`."""
    return graph_from_connectivity(
        result.matrix,
        names=result.names,
        threshold=threshold,
        density=density,
        absolute_for_threshold=absolute_for_threshold,
        use_absolute_weights=use_absolute_weights,
    )
