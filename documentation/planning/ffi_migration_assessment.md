# FFI Migration Assessment & Recommendations (Heuristic)

This assessment classifies non‑trivial FFI methods by **intended home** using name-based heuristics.

**Legend**: core = move to Rust core; ffi = keep as thin py<->rust wrapper; parser = centralize query parsing; core-display = shared display adapter in core.

## Summary

- Move to **core**: 103
- Keep in **ffi (thin)**: 239
- Move to **parser**: 27
- Move to **core-display**: 32

## Cross‑cutting refactors to do first

1) Core comparator/masks with explicit NaN policy
2) Unified filter engine + parser
3) Core attribute access service
4) Core analytics module
5) Core display adapter trait

## ffi/api/graph.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `add_nodes` | **core** | Traversal/enumeration belong in core. |
| `get_node_attribute` | **core** | Attribute access should have a single implementation in core. |
| `get_edge_attribute` | **core** | Attribute access should have a single implementation in core. |
| `get_edge_attributes` | **core** | Attribute access should have a single implementation in core. |
| `node_ids` | **core** | Traversal/enumeration belong in core. |
| `edge_ids` | **core** | Traversal/enumeration belong in core. |
| `add_edges` | **core** | Traversal/enumeration belong in core. |
| `filter_nodes` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `filter_edges` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `group_nodes_by_attribute` | **core** | Traversal/enumeration belong in core. |
| `remove_nodes` | **core** | Traversal/enumeration belong in core. |
| `remove_edges` | **core** | Traversal/enumeration belong in core. |
| `aggregate` | **core** | Analytics and aggregation belong in core. |
| `commit` | **core** | Versioning/history logic should be core, not FFI. |
| `create_branch` | **core** | Versioning/history logic should be core, not FFI. |
| `checkout_branch` | **core** | Versioning/history logic should be core, not FFI. |
| `branches` | **core** | Versioning/history logic should be core, not FFI. |
| `commit_history` | **core** | Versioning/history logic should be core, not FFI. |
| `neighbors` | **core** | Traversal/enumeration belong in core. |
| `degree` | **core** | Traversal/enumeration belong in core. |
| `in_degree` | **core** | Traversal/enumeration belong in core. |
| `out_degree` | **core** | Traversal/enumeration belong in core. |
| `edges_table` | **core** | Traversal/enumeration belong in core. |
| `create_nodes_accessor_internal` | **core** | Traversal/enumeration belong in core. |
| `create_edges_accessor_internal` | **core** | Traversal/enumeration belong in core. |
| `get_node_ids` | **core** | Traversal/enumeration belong in core. |
| `get_edge_ids` | **core** | Traversal/enumeration belong in core. |
| `get_node_ids_array` | **core** | Traversal/enumeration belong in core. |
| `get_edge_ids_array` | **core** | Traversal/enumeration belong in core. |
| `group_nodes_by_attribute_internal` | **core** | Traversal/enumeration belong in core. |
| `_get_node_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `validate_node_filter_attributes` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `validate_edge_filter_attributes` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `attribute_exists_on_nodes` | **core** | Traversal/enumeration belong in core. |
| `attribute_exists_on_edges` | **core** | Traversal/enumeration belong in core. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `adjacency_matrix_to_py_graph_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `adjacency_matrix_to_py_object` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `add_node` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `resolve_string_id_to_node` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `density` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_node_attribute` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_node_attr` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_edge_attribute` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_edge_attr` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `edge_endpoints` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_node_attributes` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_edge_attributes` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `analytics` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `add_edge` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `shortest_path` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `historical_view` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `get_node_mapping` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `adjacency_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `adjacency` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `weighted_adjacency_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `dense_adjacency_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `laplacian_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `transition_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `subgraph_adjacency_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `neighborhood` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `add_graph` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `view` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `is_connected` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `table` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `adjacency_matrix_to_graph_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `node_attribute_keys` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `edge_attribute_keys` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `has_node_attribute` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `has_edge_attribute` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `get_edge_endpoints` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__getattr__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `sparse_adjacency_matrix` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
## ffi/api/graph_analytics.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `connected_components` | **core** | Analytics and aggregation belong in core. |
| `get_summary` | **core** | Analytics and aggregation belong in core. |
| `bfs` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `dfs` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `shortest_path` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `has_path` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `memory_statistics` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/api/graph_query.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `filter_nodes` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `filter_edges` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `filter_subgraph_nodes` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `aggregate` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `execute` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `get_stats` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `aggregate_custom_nodes` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
## ffi/api/graph_version.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `from_commit_info` | **core** | Versioning/history logic should be core, not FFI. |
| `commit` | **core** | Versioning/history logic should be core, not FFI. |
| `create_branch` | **core** | Versioning/history logic should be core, not FFI. |
| `checkout_branch` | **core** | Versioning/history logic should be core, not FFI. |
| `branches` | **core** | Versioning/history logic should be core, not FFI. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `historical_view` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `create_snapshot` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `restore_snapshot` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `get_history` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `get_node_mapping` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `get_info` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/core/accessors.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `_get_node_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `_get_edge_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `__str__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__str__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `attr_value_to_python_value` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__next__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__iter__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__len__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `attributes` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `table` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `all` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__next__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__iter__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__len__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `attributes` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `table` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `all` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__getattr__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/core/array.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `count` | **core** | Analytics and aggregation belong in core. |
| `null_count` | **core** | Analytics and aggregation belong in core. |
| `percentile` | **core** | Comparison/masking/statistics must live in core. |
| `value_counts` | **core** | Analytics and aggregation belong in core. |
| `__gt__` | **core** | Comparison/masking/statistics must live in core. |
| `__lt__` | **core** | Comparison/masking/statistics must live in core. |
| `__ge__` | **core** | Comparison/masking/statistics must live in core. |
| `__le__` | **core** | Comparison/masking/statistics must live in core. |
| `__eq__` | **core** | Comparison/masking/statistics must live in core. |
| `__ne__` | **core** | Comparison/masking/statistics must live in core. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `_repr_html_` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `_get_display_data` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `_try_rich_display` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `_try_rich_html_display` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__iter__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_list` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `min` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `max` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `has_null` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `drop_na` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `fill_na` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `unique` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `describe` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `values` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `preview` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_numpy` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_pandas` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `_get_dtype` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `min` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `max` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__next__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `from_py_objects` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `to_scipy_sparse` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
## ffi/core/attributes.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `get_node_attribute` | **core** | Attribute access should have a single implementation in core. |
| `get_edge_attribute` | **core** | Attribute access should have a single implementation in core. |
| `get_node_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `get_edge_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `get_node_attributes_for_nodes` | **core** | Attribute access should have a single implementation in core. |
| `get_node_attributes_dict` | **core** | Attribute access should have a single implementation in core. |
| `get_edge_attributes_dict` | **core** | Attribute access should have a single implementation in core. |
| `set_node_attributes_from_dict` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `set_edge_attributes_from_dict` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `set_node_attribute_bulk` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_edge_attribute_bulk` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/core/history.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `get_node_ids` | **core** | Traversal/enumeration belong in core. |
| `get_edge_ids` | **core** | Traversal/enumeration belong in core. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
## ffi/core/matrix.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `get_column_by_name` | **core** | Attribute access should have a single implementation in core. |
| `get_column` | **core** | Attribute access should have a single implementation in core. |
| `sum_axis` | **core** | Analytics and aggregation belong in core. |
| `mean_axis` | **core** | Analytics and aggregation belong in core. |
| `std_axis` | **core** | Analytics and aggregation belong in core. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `_repr_html_` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `_get_display_data` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `zeros` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `identity` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `from_graph_attributes` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `get_cell` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `get_row` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `iter_rows` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `iter_columns` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `transpose` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `multiply` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `inverse` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `power` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `elementwise_multiply` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `determinant` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_pandas` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `_try_rich_display` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `_try_rich_html_display` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__iter__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `data` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `preview` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `dense` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_numpy` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/core/neighborhood.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `avg_nodes_per_neighborhood` | **core** | Traversal/enumeration belong in core. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__str__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `subgraph` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `neighborhoods` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__iter__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__next__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `avg_time_per_neighborhood_ms` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/core/query.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `equals` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `greater_than` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `less_than` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `not_equals` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `greater_than_or_equal` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `less_than_or_equal` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `attribute_equals` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `attribute_filter` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `and_filters` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `or_filters` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `attribute_equals` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `attribute_filter` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `and_filters` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `or_filters` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `source_attribute_equals` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `target_attribute_equals` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `source_or_target_attribute_equals` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
| `source_or_target_attribute_in` | **parser** | Query/expr parsing should be centralized; FFI should route only. |
## ffi/core/subgraph.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `nodes` | **core** | Traversal/enumeration belong in core. |
| `edges` | **core** | Traversal/enumeration belong in core. |
| `node_ids` | **core** | Traversal/enumeration belong in core. |
| `edge_ids` | **core** | Traversal/enumeration belong in core. |
| `degree` | **core** | Traversal/enumeration belong in core. |
| `filter_edges` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `connected_components` | **core** | Analytics and aggregation belong in core. |
| `get_node_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `get_edge_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `edges_table` | **core** | Traversal/enumeration belong in core. |
| `_get_node_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `_get_edge_attribute_column` | **core** | Attribute access should have a single implementation in core. |
| `filter_nodes` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__str__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `python_value_to_attr_value` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `attr_value_to_python_value` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `from_core_subgraph` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `new_with_inner` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `ensure_inner` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `density` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `is_connected` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `update` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `table` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `graph` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_graph` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_networkx` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `phase1_clustering_coefficient` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `phase1_transitivity` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `phase1_density` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `merge_with` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `intersect_with` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `subtract_from` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `calculate_similarity` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/core/table.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `from_graph_nodes` | **core** | Traversal/enumeration belong in core. |
| `from_graph_edges` | **core** | Traversal/enumeration belong in core. |
| `filter_by_degree` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `filter_by_connectivity` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `filter_by_distance` | **core** | Predicate execution is a core concern; keep FFI thin. |
| `null_count` | **core** | Analytics and aggregation belong in core. |
| `sum` | **core** | Analytics and aggregation belong in core. |
| `mean` | **core** | Analytics and aggregation belong in core. |
| `count` | **core** | Analytics and aggregation belong in core. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `_repr_html_` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `_get_display_data` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__next__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `_try_rich_display` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `_try_rich_html_display` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `head` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `tail` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `sort_by` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `describe` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_dict` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `group_by` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `inner_join` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `left_join` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `right_join` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `outer_join` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `union` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `intersect` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__iter__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `iter` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `data` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `preview` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_numpy` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `fill_na` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `drop_na` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `has_null` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_pandas` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `agg` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/core/traversal.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
## ffi/core/views.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `neighbors` | **core** | Traversal/enumeration belong in core. |
| `__str__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__str__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__setitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `id` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__contains__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `keys` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `values` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `items` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `update` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_dict` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__iter__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__next__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `clone` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__getitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__setitem__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `id` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `edge_id` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `source` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `target` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `endpoints` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__contains__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `keys` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `values` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `items` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `update` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_dict` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__iter__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__next__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `clone` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/display.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `pydict_to_hashmap` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `python_to_json_value` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `py_format_array` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `py_format_matrix` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `py_format_table` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `py_format_data_structure` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `py_detect_display_type` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `register_display_functions` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/traits/subgraph_operations.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `nodes` | **core** | Traversal/enumeration belong in core. |
| `edges` | **core** | Traversal/enumeration belong in core. |
| `node_count` | **core** | Analytics and aggregation belong in core. |
| `edge_count` | **core** | Analytics and aggregation belong in core. |
| `summary` | **core** | Analytics and aggregation belong in core. |
| `neighbors` | **core** | Traversal/enumeration belong in core. |
| `degree` | **core** | Traversal/enumeration belong in core. |
| `get_node_attribute` | **core** | Attribute access should have a single implementation in core. |
| `get_edge_attribute` | **core** | Attribute access should have a single implementation in core. |
| `connected_components` | **core** | Analytics and aggregation belong in core. |
| `subgraph_from_edges` | **core** | Traversal/enumeration belong in core. |
| `is_empty` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `contains_node` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `contains_edge` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `edge_endpoints` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `has_edge_between` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `clustering_coefficient` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `transitivity` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `density` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `merge_with` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `intersect_with` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `subtract_from` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `calculate_similarity` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `bfs_subgraph` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `dfs_subgraph` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `shortest_path_subgraph` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `induced_subgraph` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `collapse_to_node` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_node_attrs` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `set_edge_attrs` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `entity_type` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/types.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__str__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `__repr__` | **core-display** | Use a single display adapter in core; FFI calls it. |
| `py_new` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `value` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `type_name` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `__hash__` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `to_object` | **ffi** | Keep conversions and object init in FFI; push logic down. |
| `compute_stats` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `sample_values` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
## ffi/utils.rs

| Method | Recommendation | Rationale |
|---|---|---|
| `python_value_to_attr_value` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `attr_value_to_python_value` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |
| `graph_error_to_py_err` | **ffi** | Likely a thin wrapper; if body grows, migrate to core. |