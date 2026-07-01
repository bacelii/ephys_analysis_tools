import numpy as np
import pytest

from ephys_analysis_tools.timeseries import coerce_neuron_traces


def test_coerce_neuron_traces_preserves_default_order_and_squeezes():
    traces = {
        "n1": np.array([[0, 1, 2, 3]]),
        "n2": np.array([4, 5, 6, 7]),
    }

    names, data = coerce_neuron_traces(traces)

    assert names == ["n1", "n2"]
    assert data.shape == (2, 4)
    np.testing.assert_array_equal(data[0], np.array([0, 1, 2, 3]))


def test_coerce_neuron_traces_allows_explicit_order():
    traces = {
        "n1": np.array([0, 1, 2]),
        "n2": np.array([3, 4, 5]),
    }

    names, data = coerce_neuron_traces(traces, names=["n2", "n1"])

    assert names == ["n2", "n1"]
    np.testing.assert_array_equal(data[0], np.array([3, 4, 5]))


def test_coerce_neuron_traces_rejects_mismatched_lengths():
    traces = {
        "n1": np.array([0, 1, 2]),
        "n2": np.array([3, 4]),
    }

    with pytest.raises(ValueError, match="same length"):
        coerce_neuron_traces(traces)


def test_coerce_neuron_traces_rejects_nan_by_default():
    traces = {"n1": np.array([0.0, np.nan, 2.0])}

    with pytest.raises(ValueError, match="NaN"):
        coerce_neuron_traces(traces)
