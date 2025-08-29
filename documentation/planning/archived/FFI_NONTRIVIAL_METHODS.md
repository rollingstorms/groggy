# Non-Trivial FFI Methods in `python-groggy/src/ffi`

This document lists all non-trivial (non-wrapper, orchestration, or logic-containing) methods in the Rust FFI layer under `python-groggy/src/ffi`. Methods are included if they contain batching, parsing, orchestration, or any logic beyond direct delegation to core Rust methods.

---

## api/graph_query.rs

- **filter_subgraph_nodes** (line ~91):
  - Filters nodes within a subgraph, parses filter, intersects with subgraph nodes, and builds a new subgraph. Contains orchestration and logic.
- **aggregate** (line ~158):
  - Aggregates attribute values for nodes or edges, supports custom node lists, builds result dict. Contains orchestration and branching logic.
- **execute** (line ~214):
  - Parses and executes query strings, dispatches to filter methods, error handling for unsupported patterns. Contains parsing and dispatch logic.
- **aggregate_custom_nodes** (impl block, after line ~257):
  - Helper for aggregation over custom node lists, computes statistics, type handling. Contains nontrivial logic.

## api/graph_version.rs

- **get_node_mapping** (line ~108):
  - Scans all nodes for a given attribute, builds a mapping dict, type conversion. Contains orchestration and logic.
- **create_snapshot** (line ~116):
  - Commits current state and builds a snapshot info dict. Orchestration logic.
- **restore_snapshot** (line ~124):
  - Parses snapshot ID, checks existence, error handling. Contains logic.
- **get_history** (line ~132):
  - Builds a summary dict of version history, collects branches, state info. Contains orchestration.

## utils.rs

- **python_value_to_attr_value** (line 6):
  - Converts arbitrary Python values to Rust attribute values, type dispatch, error handling. Contains nontrivial logic.
- **attr_value_to_python_value** (line 146):
  - Converts Rust attribute values to Python objects, type dispatch. Contains logic.
- **graph_error_to_py_err** (line 167):
  - Converts Rust errors to Python exceptions, pattern matching. Contains logic.

## core/array.rs

- **PyGraphArray::new** (line ~15):
  - Converts Python objects to Rust attribute values, builds array. Contains logic.
- **__getitem__** (line ~30):
  - Handles advanced indexing (int, slice, list, mask), error handling. Contains orchestration and logic.
- **to_list** (line ~120):
  - Converts array to Python list, type conversion. Contains logic.
- **unique** (line ~200):
  - Finds unique values, builds new array. Contains logic.
- **value_counts** (line ~240):
  - Counts value frequencies, builds dict. Contains logic.
- **describe** (line ~270):
  - Builds statistical summary. Contains logic.
- **drop_na** (line ~320):
  - Drops nulls, builds new array. Contains logic.
- **fill_na** (line ~340):
  - Fills nulls, builds new array. Contains logic.
- **_get_display_data** (line ~400):
  - Builds display data dict for Python. Contains logic.
- **comparison operators** (__gt__, __lt__, __ge__, __le__, __eq__, __ne__):
  - All contain type dispatch and logic for elementwise comparison.

---

*This list is not exhaustive for the entire folder, but covers the main files with significant non-trivial FFI logic. For a full audit, repeat this process for all files in `python-groggy/src/ffi`.*
