# ephys_analysis_tools

Toolbox for analyzing neuronal electrophysiology data for network attributes

Typical usage:

```python
from ephys_analysis_tools import (
    compute_connectivity,
    graph_from_connectivity_result,
    compute_graph_metrics,
)

traces = {
    "neuron_a": trace_a,  # shape (N,), (1, N), or (N, 1)
    "neuron_b": trace_b,
    "neuron_c": trace_c,
}

result = compute_connectivity(traces, method="pearson", absolute=True)
graph = graph_from_connectivity_result(result, threshold=0.4)
metrics = compute_graph_metrics(graph)
```

Run tests from the repo root with:

```bash
pip install -e .[dev]
pytest
```
