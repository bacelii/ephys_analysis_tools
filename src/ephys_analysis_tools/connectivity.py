"""Functional connectivity matrix estimators for electrophysiology traces."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
from numpy.typing import ArrayLike
from scipy.signal import coherence
from scipy.stats import rankdata

from .timeseries import coerce_neuron_traces

ConnectivityMethod = Literal[
    "pearson",
    "spearman",
    "cosine",
    "cross_correlation_peak",
    "coherence",
]


@dataclass(frozen=True)
class ConnectivityResult:
    """Container returned by :func:`compute_connectivity`.

    Attributes
    ----------
    names:
        Neuron names in the same order as the rows/columns of ``matrix``.
    matrix:
        Square functional connectivity matrix.
    method:
        Name of the estimator used to compute the matrix.
    metadata:
        Method parameters and bookkeeping information.
    """

    names: list[str]
    matrix: np.ndarray
    method: str
    metadata: dict[str, object] = field(default_factory=dict)


def compute_connectivity(
    neuron_traces: dict[str, ArrayLike],
    *,
    method: ConnectivityMethod = "pearson",
    names: list[str] | None = None,
    absolute: bool = True,
    diagonal: float = 0.0,
    max_lag: int | None = None,
    fs: float = 1.0,
    frequency_band: tuple[float, float] | None = None,
    nperseg: int | None = None,
) -> ConnectivityResult:
    """Compute a functional connectivity matrix from neuron traces.

    Parameters
    ----------
    neuron_traces:
        Mapping from neuron name to a 1D electrophysiology trace.
    method:
        Connectivity estimator. Options are ``"pearson"``, ``"spearman"``,
        ``"cosine"``, ``"cross_correlation_peak"``, and ``"coherence"``.
    names:
        Optional neuron order. If omitted, mapping insertion order is used.
    absolute:
        If True, convert signed similarities to their absolute value. This is
        often useful before graph-theoretic analysis because many graph metrics
        assume non-negative edge weights.
    diagonal:
        Value assigned to the diagonal after computation. For graph construction
        this should usually be 0.0 to avoid self-loops.
    max_lag:
        Maximum lag, in samples, used by ``cross_correlation_peak``. If None,
        all possible lags are considered.
    fs:
        Sampling frequency used by ``coherence``.
    frequency_band:
        Optional ``(low, high)`` frequency band over which coherence is averaged.
    nperseg:
        Segment length passed to ``scipy.signal.coherence``. If None, SciPy's
        default is used, capped to the trace length.

    Returns
    -------
    ConnectivityResult
        Names, square matrix, method name, and metadata.
    """
    ordered_names, data = coerce_neuron_traces(neuron_traces, names=names)
    matrix = compute_connectivity_matrix(
        data,
        method=method,
        absolute=absolute,
        diagonal=diagonal,
        max_lag=max_lag,
        fs=fs,
        frequency_band=frequency_band,
        nperseg=nperseg,
    )
    return ConnectivityResult(
        names=ordered_names,
        matrix=matrix,
        method=method,
        metadata={
            "absolute": absolute,
            "diagonal": diagonal,
            "max_lag": max_lag,
            "fs": fs,
            "frequency_band": frequency_band,
            "nperseg": nperseg,
        },
    )


def compute_connectivity_matrix(
    data: ArrayLike,
    *,
    method: ConnectivityMethod = "pearson",
    absolute: bool = True,
    diagonal: float = 0.0,
    max_lag: int | None = None,
    fs: float = 1.0,
    frequency_band: tuple[float, float] | None = None,
    nperseg: int | None = None,
) -> np.ndarray:
    """Compute a square connectivity matrix from ``(neurons, time)`` data."""
    x = np.asarray(data, dtype=float)
    if x.ndim != 2:
        raise ValueError(f"data must be 2D with shape (n_neurons, n_timepoints); got {x.shape}")
    if x.shape[0] < 1:
        raise ValueError("data must contain at least one neuron")
    if x.shape[1] < 2:
        raise ValueError("data must contain at least two timepoints")
    if not np.all(np.isfinite(x)):
        raise ValueError("data contains NaN or infinite values")

    if method == "pearson":
        matrix = _pearson_connectivity(x)
    elif method == "spearman":
        matrix = _spearman_connectivity(x)
    elif method == "cosine":
        matrix = _cosine_connectivity(x)
    elif method == "cross_correlation_peak":
        matrix = _cross_correlation_peak_connectivity(x, max_lag=max_lag)
    elif method == "coherence":
        matrix = _coherence_connectivity(
            x,
            fs=fs,
            frequency_band=frequency_band,
            nperseg=nperseg,
        )
    else:
        raise ValueError(
            "unknown connectivity method "
            f"{method!r}; expected pearson, spearman, cosine, "
            "cross_correlation_peak, or coherence"
        )

    matrix = np.asarray(matrix, dtype=float)
    matrix = np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0)

    if absolute:
        matrix = np.abs(matrix)

    # Floating-point operations can create tiny asymmetries. The estimators here
    # are intended to define undirected functional connectivity, so enforce that.
    matrix = (matrix + matrix.T) / 2.0
    np.fill_diagonal(matrix, float(diagonal))
    return matrix


def threshold_connectivity(
    matrix: ArrayLike,
    *,
    threshold: float | None = None,
    density: float | None = None,
    absolute_for_threshold: bool = True,
) -> np.ndarray:
    """Threshold a square connectivity matrix while preserving edge weights.

    Exactly one of ``threshold`` or ``density`` may be provided.

    Parameters
    ----------
    matrix:
        Square connectivity matrix.
    threshold:
        Keep edges whose score is greater than or equal to this threshold.
    density:
        Keep the strongest fraction of possible undirected edges. For example,
        ``density=0.1`` keeps approximately the top 10 percent of possible
        off-diagonal edges. Zero-weight edges are not retained.
    absolute_for_threshold:
        If True, threshold by absolute value while preserving the original signed
        value in the returned matrix.
    """
    m = _validate_square_matrix(matrix)
    if threshold is not None and density is not None:
        raise ValueError("provide either threshold or density, not both")
    if threshold is not None and threshold < 0:
        raise ValueError("threshold must be non-negative")
    if density is not None and not (0 <= density <= 1):
        raise ValueError("density must be between 0 and 1")

    n = m.shape[0]
    out = np.zeros_like(m, dtype=float)
    if n < 2:
        return out

    rows, cols = np.triu_indices(n, k=1)
    values = m[rows, cols]
    scores = np.abs(values) if absolute_for_threshold else values
    finite = np.isfinite(scores)

    if density is not None:
        positive = finite & (scores > 0)
        candidate_indices = np.flatnonzero(positive)
        if density == 0 or candidate_indices.size == 0:
            return out
        n_possible = rows.size
        n_keep = int(np.ceil(density * n_possible))
        n_keep = min(n_keep, candidate_indices.size)
        order = candidate_indices[np.argsort(scores[candidate_indices])[::-1]]
        selected = order[:n_keep]
    elif threshold is not None:
        selected = np.flatnonzero(finite & (scores >= threshold) & (scores > 0))
    else:
        selected = np.flatnonzero(finite & (scores != 0))

    out[rows[selected], cols[selected]] = values[selected]
    out[cols[selected], rows[selected]] = values[selected]
    return out


def _pearson_connectivity(x: np.ndarray) -> np.ndarray:
    matrix = np.corrcoef(x, rowvar=True)
    if np.ndim(matrix) == 0:
        matrix = np.array([[float(matrix)]])
    return matrix


def _spearman_connectivity(x: np.ndarray) -> np.ndarray:
    ranked = rankdata(x, axis=1)
    return _pearson_connectivity(ranked)


def _cosine_connectivity(x: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(x, axis=1)
    normalized = np.divide(
        x,
        norms[:, None],
        out=np.zeros_like(x, dtype=float),
        where=norms[:, None] != 0,
    )
    return normalized @ normalized.T


def _cross_correlation_peak_connectivity(
    x: np.ndarray,
    *,
    max_lag: int | None,
) -> np.ndarray:
    if max_lag is not None and max_lag < 0:
        raise ValueError("max_lag must be non-negative or None")

    n_neurons, n_timepoints = x.shape
    centered = x - x.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    matrix = np.zeros((n_neurons, n_neurons), dtype=float)
    all_lags = np.arange(-(n_timepoints - 1), n_timepoints)

    if max_lag is None:
        lag_mask = np.ones_like(all_lags, dtype=bool)
    else:
        lag_mask = np.abs(all_lags) <= max_lag

    for i in range(n_neurons):
        for j in range(i, n_neurons):
            denom = norms[i] * norms[j]
            if denom == 0:
                value = 0.0
            else:
                corr = np.correlate(centered[i], centered[j], mode="full") / denom
                window = corr[lag_mask]
                value = float(window[np.argmax(np.abs(window))]) if window.size else 0.0
            matrix[i, j] = value
            matrix[j, i] = value

    return matrix


def _coherence_connectivity(
    x: np.ndarray,
    *,
    fs: float,
    frequency_band: tuple[float, float] | None,
    nperseg: int | None,
) -> np.ndarray:
    if fs <= 0:
        raise ValueError("fs must be positive")
    if frequency_band is not None:
        low, high = frequency_band
        if low < 0 or high <= low:
            raise ValueError("frequency_band must be a tuple (low, high) with 0 <= low < high")

    n_neurons, n_timepoints = x.shape
    segment_length = nperseg
    if segment_length is not None:
        if segment_length < 2:
            raise ValueError("nperseg must be at least 2")
        segment_length = min(segment_length, n_timepoints)

    matrix = np.zeros((n_neurons, n_neurons), dtype=float)
    for i in range(n_neurons):
        matrix[i, i] = 1.0
        for j in range(i + 1, n_neurons):
            freqs, cxy = coherence(x[i], x[j], fs=fs, nperseg=segment_length)
            if frequency_band is not None:
                low, high = frequency_band
                mask = (freqs >= low) & (freqs <= high)
                if not np.any(mask):
                    raise ValueError(
                        "frequency_band does not overlap coherence frequency bins; "
                        f"available range is {freqs.min()} to {freqs.max()}"
                    )
                cxy = cxy[mask]
            value = float(np.nanmean(cxy))
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _validate_square_matrix(matrix: ArrayLike) -> np.ndarray:
    m = np.asarray(matrix, dtype=float)
    if m.ndim != 2 or m.shape[0] != m.shape[1]:
        raise ValueError(f"matrix must be square; got shape {m.shape}")
    if not np.all(np.isfinite(m)):
        raise ValueError("matrix contains NaN or infinite values")
    return m
