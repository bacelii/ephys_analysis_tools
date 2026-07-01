import numpy as np
import pytest

from ephys_analysis_tools.connectivity import (
    compute_connectivity,
    compute_connectivity_matrix,
    threshold_connectivity,
)


def test_pearson_connectivity_can_preserve_signed_correlations():
    traces = {
        "a": np.array([0, 1, 2, 3, 4]),
        "b": np.array([0, 1, 2, 3, 4]),
        "c": np.array([4, 3, 2, 1, 0]),
    }

    result = compute_connectivity(traces, method="pearson", absolute=False)

    assert result.names == ["a", "b", "c"]
    assert result.matrix.shape == (3, 3)
    assert result.matrix[0, 1] == pytest.approx(1.0)
    assert result.matrix[0, 2] == pytest.approx(-1.0)
    assert np.allclose(np.diag(result.matrix), 0.0)


def test_spearman_detects_monotonic_relationship():
    traces = {
        "linear": np.array([0, 1, 2, 3, 4]),
        "quadratic": np.array([0, 1, 4, 9, 16]),
    }

    result = compute_connectivity(traces, method="spearman", absolute=False)

    assert result.matrix[0, 1] == pytest.approx(1.0)


def test_absolute_true_converts_negative_correlation_to_positive_strength():
    traces = {
        "a": np.array([0, 1, 2, 3, 4]),
        "b": np.array([4, 3, 2, 1, 0]),
    }

    result = compute_connectivity(traces, method="pearson", absolute=True)

    assert result.matrix[0, 1] == pytest.approx(1.0)


def test_cosine_connectivity_handles_zero_vector():
    data = np.array([
        [1, 0, 0],
        [0, 0, 0],
    ])

    matrix = compute_connectivity_matrix(data, method="cosine", absolute=False)

    assert matrix[0, 1] == pytest.approx(0.0)
    assert np.all(np.isfinite(matrix))


def test_cross_correlation_peak_detects_lagged_signal():
    base = np.array([0, 0, 1, 0, 0, 0], dtype=float)
    shifted = np.array([0, 0, 0, 1, 0, 0], dtype=float)
    data = np.vstack([base, shifted])

    matrix = compute_connectivity_matrix(
        data,
        method="cross_correlation_peak",
        absolute=True,
        max_lag=1,
    )

    assert matrix[0, 1] > 0.5
    assert matrix[0, 1] == pytest.approx(matrix[1, 0])


def test_coherence_connectivity_returns_symmetric_bounded_matrix():
    t = np.linspace(0, 1, 128, endpoint=False)
    traces = {
        "a": np.sin(2 * np.pi * 10 * t),
        "b": np.sin(2 * np.pi * 10 * t),
    }

    result = compute_connectivity(
        traces,
        method="coherence",
        fs=128,
        frequency_band=(8, 12),
        nperseg=64,
    )

    assert result.matrix.shape == (2, 2)
    assert result.matrix[0, 1] == pytest.approx(result.matrix[1, 0])
    assert 0.0 <= result.matrix[0, 1] <= 1.0


def test_threshold_connectivity_by_threshold_preserves_signed_values():
    matrix = np.array([
        [0.0, 0.9, -0.4],
        [0.9, 0.0, 0.2],
        [-0.4, 0.2, 0.0],
    ])

    sparse = threshold_connectivity(matrix, threshold=0.3, absolute_for_threshold=True)

    assert sparse[0, 1] == pytest.approx(0.9)
    assert sparse[0, 2] == pytest.approx(-0.4)
    assert sparse[1, 2] == pytest.approx(0.0)


def test_threshold_connectivity_by_density_keeps_strongest_edges():
    matrix = np.array([
        [0.0, 0.9, 0.4],
        [0.9, 0.0, 0.2],
        [0.4, 0.2, 0.0],
    ])

    sparse = threshold_connectivity(matrix, density=1 / 3)

    assert np.count_nonzero(np.triu(sparse, 1)) == 1
    assert sparse[0, 1] == pytest.approx(0.9)


def test_threshold_connectivity_rejects_threshold_and_density_together():
    with pytest.raises(ValueError, match="either threshold or density"):
        threshold_connectivity(np.eye(3), threshold=0.2, density=0.5)
