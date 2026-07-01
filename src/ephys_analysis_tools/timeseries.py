"""Utilities for converting neuron trace dictionaries into matrices."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike


def coerce_neuron_traces(
    neuron_traces: Mapping[str, ArrayLike],
    names: Sequence[str] | None = None,
    *,
    allow_nan: bool = False,
) -> tuple[list[str], np.ndarray]:
    """Convert a mapping of neuron name -> 1D trace into a 2D matrix.

    Parameters
    ----------
    neuron_traces:
        Mapping from neuron names to one-dimensional time series. Values may be
        shaped ``(N,)``, ``(1, N)``, or ``(N, 1)``; singleton dimensions are
        squeezed.
    names:
        Optional explicit neuron order. If omitted, the insertion order of the
        mapping is used.
    allow_nan:
        If False, raise an error when traces contain NaN or infinite values.

    Returns
    -------
    names, data:
        ``names`` is the ordered list of neuron names. ``data`` has shape
        ``(n_neurons, n_timepoints)``.
    """
    if not neuron_traces:
        raise ValueError("neuron_traces must contain at least one neuron")

    ordered_names = list(neuron_traces.keys()) if names is None else list(names)
    if not ordered_names:
        raise ValueError("names must contain at least one neuron")

    missing = [name for name in ordered_names if name not in neuron_traces]
    if missing:
        raise KeyError(f"names contains neurons not present in neuron_traces: {missing}")

    traces: list[np.ndarray] = []
    n_timepoints: int | None = None

    for name in ordered_names:
        arr = np.asarray(neuron_traces[name], dtype=float).squeeze()

        if arr.ndim != 1:
            raise ValueError(
                f"trace for neuron {name!r} must be 1D after squeezing; "
                f"got shape {np.asarray(neuron_traces[name]).shape}"
            )
        if arr.size < 2:
            raise ValueError(f"trace for neuron {name!r} must have at least 2 samples")
        if not allow_nan and not np.all(np.isfinite(arr)):
            raise ValueError(f"trace for neuron {name!r} contains NaN or infinite values")

        if n_timepoints is None:
            n_timepoints = int(arr.size)
        elif arr.size != n_timepoints:
            raise ValueError(
                "all traces must have the same length; "
                f"expected {n_timepoints}, got {arr.size} for neuron {name!r}"
            )

        traces.append(arr)

    return ordered_names, np.vstack(traces)
