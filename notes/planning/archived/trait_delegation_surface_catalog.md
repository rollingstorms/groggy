# Trait Delegation Surface Catalog

Reference catalogue of Python-facing classes currently using dynamic delegation. Each section enumerates the methods slated for explicit trait-backed wrappers per the stabilization plan.

## Graph & Subgraph Surface

- Dynamic entry points today: `PyGraph.__getattr__`, `PyNeighborhoodSubgraph.__getattr__`
- Target trait(s): `SubgraphOps`, plus companion traits for similarity/hierarchy as outlined in the plan
- Methods to expose explicitly on `PyGraph`, `PySubgraph`, and `PyNeighborhoodSubgraph`:

```
adjacency_list, bfs, calculate_similarity, child_meta_nodes, clustering_coefficient, collapse,
connected_components, contains_edge, contains_node, degree, density, dfs, edge_count,
edge_endpoints, edge_ids, edges, edges_table, entity_type, filter_edges, filter_nodes,
get_edge_attribute, get_node_attribute, group_by, has_edge, has_edge_between, has_meta_nodes,
has_node, has_path, hierarchy_level, in_degree, induced_subgraph, intersect_with,
is_connected, is_empty, merge_with, meta_nodes, neighborhood, neighbors, node_count,
node_ids, nodes, out_degree, parent_meta_node, sample, set_edge_attrs, set_node_attrs,
shortest_path_subgraph, subgraph_from_edges, subtract_from, summary, table, to_edges,
to_graph, to_matrix, to_networkx, to_nodes, transitivity, viz, __getitem__, __len__,
__repr__, __str__
```

> Notes: `group_by`, `table`, `to_nodes`, `to_edges`, and `to_matrix` have prototype implementations and may move into dedicated traits (`TableOps`, `ArrayOps`). `collapse` currently aliases `collapse_to_node_with_defaults`; we will decide whether to expose it directly or via `HierarchyOps`.

> _Update 2025-01-XX_: `PyGraph` already has extensive explicit method exposure (71+ methods). Assessment revealed:
> - **Already Explicit**: `node_count`, `edge_count`, `has_node`, `has_edge`, `density`, `is_empty`, `node_ids`, `edge_ids`, `filter_nodes`, `filter_edges`, `bfs`, `dfs`, `neighborhood`, `shortest_path`, `to_networkx`, `to_matrix`, `table`, `group_by`, and many more.
> - **Newly Added Explicit**: `connected_components`, `clustering_coefficient`, `transitivity`, `has_path`, `sample` - using new `with_full_view` helper for efficient cached delegation.
> - **Still Dynamic via `__getattr__`**: Node/edge attribute dictionaries (e.g., `graph.salary` returns dict of node values), plus remaining subgraph methods like `induced_subgraph`, `subgraph_from_edges`, `merge_with`, `intersect_with`, etc.
>
> The `with_full_view` helper has been implemented to standardize delegation through the cached view, avoiding expensive subgraph recreation on each call.
>
> _Update 2025-01-XX_: Extended `PyGraph` with additional explicit methods for common operations:
> - **Count & Check**: `node_count`, `edge_count`, `has_node`, `has_edge`, `has_edge_between`, `is_empty`
> - **Filtering**: `filter_nodes`, `filter_edges`
> - **Properties**: `density`, `node_ids`, `edge_ids`
> - **Degree & Neighbors**: `degree`, `in_degree`, `out_degree`, `neighbors`
>
> These 22 methods (7 original + 15 new) now bypass dynamic delegation and execute directly through
> the efficient `with_full_view` helper. Remaining methods in the catalog still use `__getattr__`.

## Table Surface

### Base Table (delegation target for nodes/edges tables)

- Dynamic entry points: `PyNodesTable.__getattr__`, `PyEdgesTable.__getattr__`
- Target trait(s): `TableOps` (to be expanded), numeric helpers, streaming helpers
- Methods to re-expose explicitly (current `PyBaseTable` surface):

```
__getitem__, __iter__, __len__, __repr__, __setitem__, __str__, _repr_html_, add_prefix,
add_suffix, agg, aggregate, append, append_row, apply, apply_to_columns, apply_to_rows,
assign, check_outliers, column, column_info, column_names, columns, corr, corr_columns,
cov, cov_columns, cummax, cummin, cumsum, describe, drop_columns, drop_duplicates,
drop_rows, dropna, dropna_subset, expanding, expanding_all, extend, extend_rows, fillna,
fillna_all, filter, from_csv, from_dict, from_json, from_parquet, get_column_numeric,
get_column_raw, get_percentile, group_by, group_by_agg, groupby, groupby_single,
has_column, has_nulls, head, intersect, is_empty, isin, isna, iter, join, median, melt,
ncols, new, nlargest, notna, nrows, nsmallest, null_counts, parse_join_on, pct_change,
percentile, percentiles, pivot_table, profile, quantile, quantiles, query, rename,
reorder_columns, rich_display, rolling, rolling_all, sample, select, set_column, set_value,
set_values_by_mask, set_values_by_range, shape, shift, slice, sort_by, sort_values, std,
tail, to_csv, to_json, to_pandas, to_parquet, to_type, union, validate_schema,
value_counts, var
```

### Nodes Table

- Native `#[pymethods]` plus dynamic fallback; retain existing statics and add wrappers for delegated behavior

```
__getattr__, __getitem__, __iter__, __len__, __repr__, __str__, _get_display_data,
base_table, drop_columns, filter, filter_by_attr, from_csv, from_dict, from_json,
from_parquet, group_by, head, interactive, interactive_embed, interactive_viz,
into_base_table, is_empty, iter, ncols, new, node_ids, nrows, rich_display, select,
shape, slice, sort_by, sort_values, tail, to_csv, to_json, to_pandas, to_parquet,
unique_attr_values, viz, with_attributes
```

### Edges Table

```
__getattr__, __getitem__, __iter__, __len__, __repr__, __str__, _get_display_data,
as_tuples, auto_assign_edge_ids, base_table, drop_columns, edge_ids, filter,
filter_by_attr, filter_by_sources, filter_by_targets, from_csv, from_dict, from_json,
from_parquet, group_by, head, interactive, interactive_embed, interactive_viz,
into_base_table, iter, ncols, new, nrows, rich_display, select, shape, slice,
sort_by, sort_values, sources, tail, targets, to_csv, to_json, to_pandas,
to_parquet, unique_attr_values, viz
```

## Table Arrays

### Generic TableArray (target of Nodes/EdgesTableArray delegation)

```
__getitem__, __iter__, __len__, __repr__, __str__, agg, all, iter_agg, max, mean, new, to_list
```

### NodesTableArray & EdgesTableArray Wrappers

```
__getattr__, __getitem__, __iter__, __len__, __repr__, __str__, to_list
```

> Notes: Delegated `agg`/`max`/`mean` currently come from `PyTableArray`; the plan is to surface table-array traits so grouped tables expose typed methods without dynamic fallback.

## Array Surface

- Dynamic entry point: `PyBaseArray.__getattr__` (element-wise method application)
- Static methods already defined:

```
__eq__, __ge__, __getattr__, __getitem__, __gt__, __le__, __len__, __lt__, __ne__, __repr__,
_repr_html_, append, append_element, apply, apply_to_each, contains, corr, count, cov,
cummax, cummin, cumsum, describe, drop_duplicates_elements, drop_elements, dropna,
dtype, expanding, extend, extend_elements, fillna, filter, get, get_percentile, has_nulls,
head, infer_numeric_type, insert, is_empty, is_numeric, isna, iter, len, map, max, mean,
median, min, new, notna, null_count, numeric_compatibility_info, nunique, pct_change,
percentile, percentiles, quantile, quantiles, remove, reverse, rolling, shift, sort, std,
sum, tail, to_list, to_num_array, to_table, to_table_with_name, to_table_with_prefix,
to_table_with_suffix, to_type, unique, value_counts, var
```

> Element-method delegation (`array.some_method()`) will be retired or gated; explicit trait-backed helpers (e.g., `apply_to_each`) remain.

## Accessor Surface

### NodesAccessor (already explicit)

```
__getitem__, __iter__, __len__, __setitem__, __str__, _get_node_attribute_column,
all, array, attribute_names, attributes, base, filter, get_meta_node, group_by, ids,
matrix, meta, set_attrs, subgraphs, table, viz
```

### EdgesAccessor (dynamic attribute columns)

```
__getattr__, __getitem__, __iter__, __len__, __setitem__, __str__, _get_edge_attribute_column,
all, array, attribute_names, attributes, base, filter, group_by, ids, matrix, meta,
set_attrs, sources, table, targets, viz, weight_matrix
```

> The `__getattr__` path will be replaced with explicit column accessors (e.g., `column(name)`)
and/or trait-backed attribute-selection helpers.

## Summary

The classes above represent the dynamic delegation surface targeted by the stabilization plan. As traits and adapters are expanded, this catalogue should be kept in sync (append any newly discovered dynamic entry points) to ensure complete coverage before cutover.
