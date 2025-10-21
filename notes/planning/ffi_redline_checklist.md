# Redline Refactor Checklist – FFI → Core


This is a cut‑here, implement‑there plan. Treat each bullet as a PR-able task.


### ffi/api/graph.rs
**Move implementation to `core` (leave thin FFI delegate):**
- `_get_node_attribute_column` → **core::attributes** — Attribute access should have a single implementation in core.
- `add_edges` → **core::traversal** — Traversal/enumeration belong in core.
- `add_nodes` → **core::traversal** — Traversal/enumeration belong in core.
- `aggregate` → **core::analytics** — Analytics and aggregation belong in core.
- `attribute_exists_on_edges` → **core::attributes** — Traversal/enumeration belong in core.
- `attribute_exists_on_nodes` → **core::attributes** — Traversal/enumeration belong in core.
- `branches` → **core::module** — Versioning/history logic should be core, not FFI.
- `checkout_branch` → **core::module** — Versioning/history logic should be core, not FFI.
- `commit` → **core::module** — Versioning/history logic should be core, not FFI.
- `commit_history` → **core::module** — Versioning/history logic should be core, not FFI.
- `create_branch` → **core::module** — Versioning/history logic should be core, not FFI.
- `create_edges_accessor_internal` → **core::traversal** — Traversal/enumeration belong in core.
- `create_nodes_accessor_internal` → **core::traversal** — Traversal/enumeration belong in core.
- `degree` → **core::traversal** — Traversal/enumeration belong in core.
- `edge_ids` → **core::traversal** — Traversal/enumeration belong in core.
- `edges_table` → **core::traversal** — Traversal/enumeration belong in core.
- `filter_edges` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
- `filter_nodes` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
- `get_edge_attribute` → **core::attributes** — Attribute access should have a single implementation in core.
- `get_edge_attributes` → **core::attributes** — Attribute access should have a single implementation in core.
- `get_edge_ids` → **core::traversal** — Traversal/enumeration belong in core.
- `get_edge_ids_array` → **core::traversal** — Traversal/enumeration belong in core.
- `get_node_attribute` → **core::attributes** — Attribute access should have a single implementation in core.
- `get_node_ids` → **core::traversal** — Traversal/enumeration belong in core.
- `get_node_ids_array` → **core::traversal** — Traversal/enumeration belong in core.
- `group_nodes_by_attribute` → **core::attributes** — Traversal/enumeration belong in core.
- `group_nodes_by_attribute_internal` → **core::attributes** — Traversal/enumeration belong in core.
- `in_degree` → **core::traversal** — Traversal/enumeration belong in core.
- `neighbors` → **core::traversal** — Traversal/enumeration belong in core.
- `node_ids` → **core::traversal** — Traversal/enumeration belong in core.
- `out_degree` → **core::traversal** — Traversal/enumeration belong in core.
- `remove_edges` → **core::traversal** — Traversal/enumeration belong in core.
- `remove_nodes` → **core::traversal** — Traversal/enumeration belong in core.
- `validate_edge_filter_attributes` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
- `validate_node_filter_attributes` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
Refactor steps:
  1) Create/extend target module with the new function/trait.
  2) Port logic from FFI, add unit tests in core (cover NaN/order, error cases).
  3) Replace FFI body with a delegate call; keep only Py↔Rust conversions.
  4) Add integration test at FFI level to assert parity.

**Move to unified `parser` module (FFI should only route):**
- `sparse_adjacency_matrix` → **parser::expr** — Query/expr parsing should be centralized; FFI should route only.
Refactor steps:
  1) Implement expression parsing in a single parser crate/module.
  2) Return a typed predicate or IR; FFI passes it to core filters.
  3) Delete regex/dup parsing paths from FFI.

**Adopt shared display adapter in `core-display`:**
- `__repr__` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
Refactor steps:
  1) Define DisplayAdapter trait implemented by Array/Matrix/Table.
  2) FFI calls adapter; remove per-type helpers.

**Keep in FFI (thin wrappers only):**
- `__getattr__` — Likely a thin wrapper; if body grows, migrate to core.
- `add_edge` — Likely a thin wrapper; if body grows, migrate to core.
- `add_graph` — Likely a thin wrapper; if body grows, migrate to core.
- `add_node` — Likely a thin wrapper; if body grows, migrate to core.
- `adjacency` — Likely a thin wrapper; if body grows, migrate to core.
- `adjacency_matrix` — Likely a thin wrapper; if body grows, migrate to core.
- `adjacency_matrix_to_graph_matrix` — Likely a thin wrapper; if body grows, migrate to core.
- `adjacency_matrix_to_py_graph_matrix` — Likely a thin wrapper; if body grows, migrate to core.
- `adjacency_matrix_to_py_object` — Likely a thin wrapper; if body grows, migrate to core.
- `analytics` — Likely a thin wrapper; if body grows, migrate to core.
- `dense_adjacency_matrix` — Likely a thin wrapper; if body grows, migrate to core.
- `density` — Likely a thin wrapper; if body grows, migrate to core.
- `edge_attribute_keys` — Likely a thin wrapper; if body grows, migrate to core.
- `edge_endpoints` — Likely a thin wrapper; if body grows, migrate to core.
- `get_edge_endpoints` — Likely a thin wrapper; if body grows, migrate to core.
- `get_node_mapping` — Likely a thin wrapper; if body grows, migrate to core.
- `has_edge_attribute` — Likely a thin wrapper; if body grows, migrate to core.
- `has_node_attribute` — Likely a thin wrapper; if body grows, migrate to core.
- `historical_view` — Likely a thin wrapper; if body grows, migrate to core.
- `is_connected` — Likely a thin wrapper; if body grows, migrate to core.
- `laplacian_matrix` — Likely a thin wrapper; if body grows, migrate to core.
- `neighborhood` — Likely a thin wrapper; if body grows, migrate to core.
- `new` — Keep conversions and object init in FFI; push logic down.
- `node_attribute_keys` — Likely a thin wrapper; if body grows, migrate to core.
- `resolve_string_id_to_node` — Likely a thin wrapper; if body grows, migrate to core.
- `set_edge_attr` — Likely a thin wrapper; if body grows, migrate to core.
- `set_edge_attribute` — Likely a thin wrapper; if body grows, migrate to core.
- `set_edge_attributes` — Likely a thin wrapper; if body grows, migrate to core.
- `set_node_attr` — Likely a thin wrapper; if body grows, migrate to core.
- `set_node_attribute` — Likely a thin wrapper; if body grows, migrate to core.
- `set_node_attributes` — Likely a thin wrapper; if body grows, migrate to core.
- `shortest_path` — Likely a thin wrapper; if body grows, migrate to core.
- `subgraph_adjacency_matrix` — Likely a thin wrapper; if body grows, migrate to core.
- `table` — Likely a thin wrapper; if body grows, migrate to core.
- `transition_matrix` — Likely a thin wrapper; if body grows, migrate to core.
- `view` — Likely a thin wrapper; if body grows, migrate to core.
- `weighted_adjacency_matrix` — Likely a thin wrapper; if body grows, migrate to core.
Hygiene steps:
  1) Ensure zero business logic; keep conversions/docstrings.
  2) Add doc-tests for signature/arg behavior.


### ffi/core/table.rs
**Move implementation to `core` (leave thin FFI delegate):**
- `count` → **core::traversal** — Analytics and aggregation belong in core.
- `filter_by_connectivity` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
- `filter_by_degree` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
- `filter_by_distance` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
- `from_graph_edges` → **core::traversal** — Traversal/enumeration belong in core.
- `from_graph_nodes` → **core::traversal** — Traversal/enumeration belong in core.
- `mean` → **core::analytics** — Analytics and aggregation belong in core.
- `null_count` → **core::traversal** — Analytics and aggregation belong in core.
- `sum` → **core::module** — Analytics and aggregation belong in core.
Refactor steps:
  1) Create/extend target module with the new function/trait.
  2) Port logic from FFI, add unit tests in core (cover NaN/order, error cases).
  3) Replace FFI body with a delegate call; keep only Py↔Rust conversions.
  4) Add integration test at FFI level to assert parity.

**Adopt shared display adapter in `core-display`:**
- `__repr__` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
- `_get_display_data` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
- `_repr_html_` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
Refactor steps:
  1) Define DisplayAdapter trait implemented by Array/Matrix/Table.
  2) FFI calls adapter; remove per-type helpers.

**Keep in FFI (thin wrappers only):**
- `__getitem__` — Likely a thin wrapper; if body grows, migrate to core.
- `__iter__` — Likely a thin wrapper; if body grows, migrate to core.
- `__next__` — Likely a thin wrapper; if body grows, migrate to core.
- `_try_rich_display` — Likely a thin wrapper; if body grows, migrate to core.
- `_try_rich_html_display` — Likely a thin wrapper; if body grows, migrate to core.
- `agg` — Likely a thin wrapper; if body grows, migrate to core.
- `data` — Likely a thin wrapper; if body grows, migrate to core.
- `describe` — Likely a thin wrapper; if body grows, migrate to core.
- `drop_na` — Likely a thin wrapper; if body grows, migrate to core.
- `fill_na` — Likely a thin wrapper; if body grows, migrate to core.
- `group_by` — Likely a thin wrapper; if body grows, migrate to core.
- `has_null` — Likely a thin wrapper; if body grows, migrate to core.
- `head` — Likely a thin wrapper; if body grows, migrate to core.
- `inner_join` — Likely a thin wrapper; if body grows, migrate to core.
- `intersect` — Likely a thin wrapper; if body grows, migrate to core.
- `iter` — Likely a thin wrapper; if body grows, migrate to core.
- `left_join` — Likely a thin wrapper; if body grows, migrate to core.
- `matrix` — Likely a thin wrapper; if body grows, migrate to core.
- `new` — Keep conversions and object init in FFI; push logic down.
- `outer_join` — Likely a thin wrapper; if body grows, migrate to core.
- `preview` — Likely a thin wrapper; if body grows, migrate to core.
- `right_join` — Likely a thin wrapper; if body grows, migrate to core.
- `sort_by` — Likely a thin wrapper; if body grows, migrate to core.
- `tail` — Likely a thin wrapper; if body grows, migrate to core.
- `to_dict` — Likely a thin wrapper; if body grows, migrate to core.
- `to_numpy` — Likely a thin wrapper; if body grows, migrate to core.
- `to_pandas` — Likely a thin wrapper; if body grows, migrate to core.
- `union` — Likely a thin wrapper; if body grows, migrate to core.
Hygiene steps:
  1) Ensure zero business logic; keep conversions/docstrings.
  2) Add doc-tests for signature/arg behavior.


### ffi/core/subgraph.rs
**Move implementation to `core` (leave thin FFI delegate):**
- `_get_edge_attribute_column` → **core::attributes** — Attribute access should have a single implementation in core.
- `_get_node_attribute_column` → **core::attributes** — Attribute access should have a single implementation in core.
- `connected_components` → **core::analytics** — Analytics and aggregation belong in core.
- `degree` → **core::traversal** — Traversal/enumeration belong in core.
- `edge_ids` → **core::traversal** — Traversal/enumeration belong in core.
- `edges` → **core::traversal** — Traversal/enumeration belong in core.
- `edges_table` → **core::traversal** — Traversal/enumeration belong in core.
- `filter_edges` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
- `filter_nodes` → **core::filters** — Predicate execution is a core concern; keep FFI thin.
- `get_edge_attribute_column` → **core::attributes** — Attribute access should have a single implementation in core.
- `get_node_attribute_column` → **core::attributes** — Attribute access should have a single implementation in core.
- `node_ids` → **core::traversal** — Traversal/enumeration belong in core.
- `nodes` → **core::traversal** — Traversal/enumeration belong in core.
Refactor steps:
  1) Create/extend target module with the new function/trait.
  2) Port logic from FFI, add unit tests in core (cover NaN/order, error cases).
  3) Replace FFI body with a delegate call; keep only Py↔Rust conversions.
  4) Add integration test at FFI level to assert parity.

**Adopt shared display adapter in `core-display`:**
- `__repr__` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
- `__str__` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
Refactor steps:
  1) Define DisplayAdapter trait implemented by Array/Matrix/Table.
  2) FFI calls adapter; remove per-type helpers.

**Keep in FFI (thin wrappers only):**
- `__getitem__` — Likely a thin wrapper; if body grows, migrate to core.
- `attr_value_to_python_value` — Likely a thin wrapper; if body grows, migrate to core.
- `calculate_similarity` — Likely a thin wrapper; if body grows, migrate to core.
- `density` — Likely a thin wrapper; if body grows, migrate to core.
- `ensure_inner` — Likely a thin wrapper; if body grows, migrate to core.
- `from_core_subgraph` — Keep conversions and object init in FFI; push logic down.
- `graph` — Likely a thin wrapper; if body grows, migrate to core.
- `intersect_with` — Likely a thin wrapper; if body grows, migrate to core.
- `is_connected` — Likely a thin wrapper; if body grows, migrate to core.
- `merge_with` — Likely a thin wrapper; if body grows, migrate to core.
- `new` — Keep conversions and object init in FFI; push logic down.
- `new_with_inner` — Keep conversions and object init in FFI; push logic down.
- `phase1_clustering_coefficient` — Likely a thin wrapper; if body grows, migrate to core.
- `phase1_density` — Likely a thin wrapper; if body grows, migrate to core.
- `phase1_transitivity` — Likely a thin wrapper; if body grows, migrate to core.
- `python_value_to_attr_value` — Likely a thin wrapper; if body grows, migrate to core.
- `set` — Likely a thin wrapper; if body grows, migrate to core.
- `subtract_from` — Keep conversions and object init in FFI; push logic down.
- `table` — Likely a thin wrapper; if body grows, migrate to core.
- `to_graph` — Likely a thin wrapper; if body grows, migrate to core.
- `to_networkx` — Likely a thin wrapper; if body grows, migrate to core.
- `update` — Likely a thin wrapper; if body grows, migrate to core.
Hygiene steps:
  1) Ensure zero business logic; keep conversions/docstrings.
  2) Add doc-tests for signature/arg behavior.


### ffi/core/array.rs
**Move implementation to `core` (leave thin FFI delegate):**
- `__eq__` → **core::compare** — Comparison/masking/statistics must live in core.
- `__ge__` → **core::compare** — Comparison/masking/statistics must live in core.
- `__gt__` → **core::compare** — Comparison/masking/statistics must live in core.
- `__le__` → **core::compare** — Comparison/masking/statistics must live in core.
- `__lt__` → **core::compare** — Comparison/masking/statistics must live in core.
- `__ne__` → **core::compare** — Comparison/masking/statistics must live in core.
- `count` → **core::traversal** — Analytics and aggregation belong in core.
- `null_count` → **core::traversal** — Analytics and aggregation belong in core.
- `percentile` → **core::analytics** — Comparison/masking/statistics must live in core.
- `value_counts` → **core::traversal** — Analytics and aggregation belong in core.
Refactor steps:
  1) Create/extend target module with the new function/trait.
  2) Port logic from FFI, add unit tests in core (cover NaN/order, error cases).
  3) Replace FFI body with a delegate call; keep only Py↔Rust conversions.
  4) Add integration test at FFI level to assert parity.

**Move to unified `parser` module (FFI should only route):**
- `to_scipy_sparse` → **parser::expr** — Query/expr parsing should be centralized; FFI should route only.
Refactor steps:
  1) Implement expression parsing in a single parser crate/module.
  2) Return a typed predicate or IR; FFI passes it to core filters.
  3) Delete regex/dup parsing paths from FFI.

**Adopt shared display adapter in `core-display`:**
- `__repr__` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
- `_get_display_data` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
- `_repr_html_` → **core_display::Adapter** — Use a single display adapter in core; FFI calls it.
Refactor steps:
  1) Define DisplayAdapter trait implemented by Array/Matrix/Table.
  2) FFI calls adapter; remove per-type helpers.

**Keep in FFI (thin wrappers only):**
- `__getitem__` — Likely a thin wrapper; if body grows, migrate to core.
- `__iter__` — Likely a thin wrapper; if body grows, migrate to core.
- `__next__` — Likely a thin wrapper; if body grows, migrate to core.
- `_get_dtype` — Likely a thin wrapper; if body grows, migrate to core.
- `_try_rich_display` — Likely a thin wrapper; if body grows, migrate to core.
- `_try_rich_html_display` — Likely a thin wrapper; if body grows, migrate to core.
- `describe` — Likely a thin wrapper; if body grows, migrate to core.
- `drop_na` — Likely a thin wrapper; if body grows, migrate to core.
- `fill_na` — Likely a thin wrapper; if body grows, migrate to core.
- `from_py_objects` — Keep conversions and object init in FFI; push logic down.
- `has_null` — Likely a thin wrapper; if body grows, migrate to core.
- `max` — Likely a thin wrapper; if body grows, migrate to core.
- `max` — Likely a thin wrapper; if body grows, migrate to core.
- `min` — Likely a thin wrapper; if body grows, migrate to core.
- `min` — Likely a thin wrapper; if body grows, migrate to core.
- `new` — Keep conversions and object init in FFI; push logic down.
- `preview` — Likely a thin wrapper; if body grows, migrate to core.
- `to_list` — Likely a thin wrapper; if body grows, migrate to core.
- `to_numpy` — Likely a thin wrapper; if body grows, migrate to core.
- `to_pandas` — Likely a thin wrapper; if body grows, migrate to core.
- `unique` — Likely a thin wrapper; if body grows, migrate to core.
- `values` — Likely a thin wrapper; if body grows, migrate to core.
Hygiene steps:
  1) Ensure zero business logic; keep conversions/docstrings.
  2) Add doc-tests for signature/arg behavior.


## Remaining files (apply the same pattern)


- ffi/api/graph_analytics.rs: move **2** → core, **0** → parser, **0** → core-display; keep **5** thin in FFI.

- ffi/api/graph_query.rs: move **0** → core, **7** → parser, **0** → core-display; keep **0** thin in FFI.

- ffi/api/graph_version.rs: move **5** → core, **0** → parser, **2** → core-display; keep **6** thin in FFI.

- ffi/core/accessors.rs: move **2** → core, **0** → parser, **2** → core-display; keep **16** thin in FFI.

- ffi/core/attributes.rs: move **7** → core, **0** → parser, **0** → core-display; keep **4** thin in FFI.

- ffi/core/history.rs: move **2** → core, **0** → parser, **3** → core-display; keep **0** thin in FFI.

- ffi/core/matrix.rs: move **5** → core, **0** → parser, **3** → core-display; keep **23** thin in FFI.

- ffi/core/neighborhood.rs: move **1** → core, **0** → parser, **4** → core-display; keep **6** thin in FFI.

- ffi/core/query.rs: move **0** → core, **18** → parser, **0** → core-display; keep **0** thin in FFI.

- ffi/core/traversal.rs: move **0** → core, **0** → parser, **3** → core-display; keep **3** thin in FFI.

- ffi/core/views.rs: move **1** → core, **0** → parser, **2** → core-display; keep **28** thin in FFI.

- ffi/display.rs: move **0** → core, **0** → parser, **1** → core-display; keep **9** thin in FFI.

- ffi/traits/subgraph_operations.rs: move **11** → core, **0** → parser, **0** → core-display; keep **20** thin in FFI.

- ffi/types.rs: move **0** → core, **0** → parser, **3** → core-display; keep **7** thin in FFI.

- ffi/utils.rs: move **0** → core, **0** → parser, **0** → core-display; keep **3** thin in FFI.