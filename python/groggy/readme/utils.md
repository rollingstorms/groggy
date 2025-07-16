# Groggy Python: utils.py

Utility functions for graph generation, conversion, and benchmarking.

- **create_random_graph**: Fast random graph generator (Python/Rust backends).
- **convert_networkx_graph**: Import from NetworkX.
- **to_networkx, to_pandas, to_pytorch**: Export to other data science formats.

## Example
```python
from groggy.utils import create_random_graph
G = create_random_graph(100, 0.1, use_rust=True)
```
