# FFI Methods with Non-Trivial Implementation

This document lists all methods in the FFI layer of Groggy that contain non-trivial logic or implementation (i.e., not just thin wrappers or direct delegation to the Rust core). Methods are grouped by file, with a brief description of their purpose or unique logic.

---

## python-groggy/src/ffi/utils.rs
- `python_value_to_attr_value(value: &PyAny) -> PyResult<RustAttrValue>`
  - Converts Python values to Rust attribute values, with type inference, error handling, and data structure conversion.
- `attr_value_to_python_value(py: Python, attr_value: &RustAttrValue) -> PyResult<PyObject>`
  - Converts Rust attribute values to Python objects, handling all supported types.
- `graph_error_to_py_err(error: GraphError) -> PyErr`
  - Maps Rust errors to Python exceptions with custom messages.

---

## python-groggy/src/ffi/core/subgraph.rs
- `PySubgraph` struct and methods:
  - `from_core_subgraph`, `new`, `new_with_inner`, `ensure_inner`, `get_nodes`, `get_edges`, `set_graph_reference`
  - `filter_nodes`, `filter_edges`, `to_graph`, `to_networkx`, `get_node_attribute_column`, `get_edge_attribute_column`, `__getitem__`, `table`, `edges_table`, `set`, `update`, `phase1_clustering_coefficient`, `phase1_transitivity`, `phase1_density`, `merge_with`, `intersect_with`, `subtract_from`, `calculate_similarity`
  - These methods include logic for subgraph construction, attribute access, filtering, conversion, and trait-based operations.

---

## python-groggy/src/ffi/core/array.rs
- `PyGraphArray` struct and methods:
  - `new`, `__getitem__`, `to_list`, `mean`, `std`, `min`, `max`, `count`, `has_null`, `null_count`, `drop_na`, `fill_na`, `quantile`, `percentile`, `median`, `unique`, `value_counts`, `describe`, `to_numpy`, `to_pandas`, `to_scipy_sparse`, `__gt__`, `__lt__`, `__ge__`, `__le__`, `__eq__`, `__ne__`
  - Implements array slicing, indexing, statistical operations, and conversions to/from Python types, including display logic.

---

## python-groggy/src/ffi/core/table.rs
- `PyGraphTable` struct and methods:
  - `new`, `from_graph_nodes`, `from_graph_edges`, `__getitem__`, `head`, `tail`, `sort_by`, `mean`, `sum`, `describe`, `to_dict`, `group_by`, `matrix`, `filter_by_degree`, `filter_by_connectivity`, `filter_by_distance`, `inner_join`, `left_join`, `right_join`, `outer_join`, `union`, `intersect`, `fill_na`, `drop_na`, `has_null`, `null_count`, `to_numpy`, `to_pandas`, `agg`, `sum`, `mean`, `count`
  - Implements DataFrame-like operations, joins, groupby, aggregation, and display logic.

---

## python-groggy/src/ffi/core/views.rs
- `PyNodeView` and `PyEdgeView` classes and methods:
  - `__getitem__`, `__setitem__`, `__contains__`, `keys`, `values`, `neighbors`, `items`, `update`, `to_dict`, `item`, `__iter__`, `__str__`, `endpoints`, `source`, `target`
  - Attribute access, mutation, batch access, and conversion for node/edge views.

---

## python-groggy/src/ffi/api/graph.rs
- `PyGraph` struct and methods:
  - `add_nodes`, `add_edges`, `group_nodes_by_attribute`, `add_graph`, `set_node_attributes`, `set_edge_attributes`, `filter_nodes`, `filter_edges`, `neighbors`, `degree`, `in_degree`, `out_degree`, `neighborhood`, `add_edge`, `aggregate`, `view`, `is_connected`, `table`, `edges_table`
  - These methods contain logic for batch processing, attribute mapping, performance optimizations, and data conversion.

---

## python-groggy/src/ffi/api/graph_analytics.rs
- `PyGraphAnalytics` methods:
  - `connected_components`, `bfs`, `dfs`, `shortest_path`, `has_path`, `degree`, `memory_statistics`, `get_summary`
  - Implement analytics logic, including attribute setting and result conversion.

---

## python-groggy/src/ffi/api/graph_query.rs
- `PyGraphQuery` methods:
  - `filter_nodes`, `filter_edges`, `filter_subgraph_nodes`, `aggregate`, `execute`, `get_stats`, `aggregate_custom_nodes`
  - Implement advanced query logic, including parsing, filtering, and aggregation.

---

## python-groggy/src/ffi/api/graph_version.rs
- `PyGraphVersion` methods:
  - `commit`, `create_branch`, `checkout_branch`, `branches`, `commit_history`, `historical_view`, `has_uncommitted_changes`, `create_snapshot`, `restore_snapshot`, `get_history`, `get_node_mapping`, `get_info`
  - Implement version control and history logic, including state management and summary generation.

---

## python-groggy/src/ffi/core/history.rs
- `PyCommit`, `PyBranchInfo`, `PyHistoryStatistics`, `PyHistoricalView` methods:
  - Methods for commit/branch/history representation and summary.

---

## python-groggy/src/ffi/errors.rs
- No non-trivial logic (delegates to utils).

---

This list is based on the current state of the FFI codebase and highlights all methods with meaningful implementation logic beyond simple delegation or wrapping.
