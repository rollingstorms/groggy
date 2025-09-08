# Groggy Method Signatures Report

Generated: 2025-09-07 10:38:34

Complete method signatures for all major Groggy objects, including delegated methods.


## Graph Method Signatures

**Direct Methods (71)**:

- `__len__()`
- `__repr__()`
- `__str__()`
- `add_edge(source, target, uid_key=None, **kwargs)`
- `add_edges(edges, node_mapping=None, _uid_key=None, warm_cache=None)`
- `add_graph(other)`
- `add_node(**kwargs)`
- `add_nodes(data, uid_key=None)`
- `adjacency()`
- `adjacency_matrix()`
- `aggregate(attribute, operation, target=None, _node_ids=None)`
- `all_edge_attribute_names()`
- `all_node_attribute_names()`
- `bfs(start, max_depth=None, inplace=None, attr_name=None)`
- `branches()`
- `checkout_branch(branch_name)`
- `commit(message, author)`
- `commit_history()`
- `contains_edge(edge)`
- `contains_node(node)`
- `create_branch(branch_name)`
- `dense_adjacency_matrix()`
- `density()`
- `dfs(start, max_depth=None, inplace=None, attr_name=None)`
- `edge_attribute_keys(edge_id)`
- `edge_count()`
- `edge_endpoints(edge)`
- `edge_ids` → property
- `edges` → property
- `filter_edges(filter)`
- `filter_nodes(filter)`
- `get_edge_attr(edge, attr, default=None)`
- `get_edge_attrs(edges, attrs)`
- `get_node_attr(node, attr, default=None)`
- `get_node_attrs(nodes, attrs)`
- `get_node_mapping(uid_key, return_inverse=False)`
- `group_by(attribute, aggregation_attr, operation)`
- `group_nodes_by_attribute(attribute, aggregation_attr, operation)`
- `has_edge(edge_id)`
- `has_edge_attribute(edge_id, attr_name)`
- `has_node(node_id)`
- `has_node_attribute(node_id, attr_name)`
- `has_uncommitted_changes()`
- `historical_view(commit_id)`
- `is_connected()`
- `is_directed` → property
- `is_undirected` → property
- `laplacian_matrix(normalized=None)`
- `neighborhood(center_nodes, radius=None, max_nodes=None)`
- `neighborhood_statistics()`
- `neighbors(nodes=None)`
- `node_attribute_keys(node_id)`
- `node_count()`
- `node_ids` → property
- `nodes` → property
- `remove_edge(edge)`
- `remove_edges(edges)`
- `remove_node(node)`
- `remove_nodes(nodes)`
- `resolve_string_id_to_node(string_id, uid_key)`
- `set_edge_attr(edge, attr, value)`
- `set_edge_attrs(attrs_dict)`
- `set_node_attr(node, attr, value)`
- `set_node_attrs(attrs_dict)`
- `shortest_path(source, target, weight_attribute=None, inplace=None, attr_name=None)`
- `sparse_adjacency_matrix()`
- `table()`
- `to_networkx(directed: bool = False, include_attributes: bool = True)`
- `transition_matrix()`
- `view()`
- `weighted_adjacency_matrix(weight_attr)`

**Delegation via table() → GraphTable**

**Hidden Methods via table (2)**:
- `merge_with(_other)`
- `to_graph()`

**Summary**: 73 total methods (71 direct + 2 delegated)

## NodesTable Method Signatures

**Direct Methods (25)**:

- `__getitem__(key, /)`
- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `base_table()`
- `drop_columns(columns)`
- `filter(predicate)`
- `filter_by_attr(attr_name, value)`
- `group_by(columns)`
- `head(n=5)`
- `into_base_table()`
- `iter()`
- `ncols()`
- `node_ids()`
- `nrows()`
- `rich_display(config=None)`
- `select(columns)`
- `shape()`
- `slice(start, end)`
- `sort_by(column, ascending=True)`
- `tail(n=5)`
- `to_pandas()`
- `unique_attr_values(attr_name)`
- `with_attributes(attr_name, attributes)`

**Delegation via base_table() → BaseTable**

**Delegation via into_base_table() → BaseTable**

**Hidden Methods via base_table (2)**:
- `column_names()`
- `has_column(name)`

**Hidden Methods via into_base_table (2)**:
- `column_names()`
- `has_column(name)`

**Summary**: 27 total methods (25 direct + 2 delegated)

## EdgesTable Method Signatures

**Direct Methods (29)**:

- `__getitem__(key, /)`
- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `as_tuples()`
- `base_table()`
- `drop_columns(columns)`
- `edge_ids()`
- `filter(predicate)`
- `filter_by_attr(attr_name, value)`
- `filter_by_sources(source_nodes)`
- `filter_by_targets(target_nodes)`
- `group_by(columns)`
- `head(n=5)`
- `into_base_table()`
- `iter()`
- `ncols()`
- `nrows()`
- `rich_display(config=None)`
- `select(columns)`
- `shape()`
- `slice(start, end)`
- `sort_by(column, ascending=True)`
- `sources()`
- `tail(n=5)`
- `targets()`
- `to_pandas()`
- `unique_attr_values(attr_name)`

**Delegation via base_table() → BaseTable**

**Delegation via into_base_table() → BaseTable**

**Hidden Methods via base_table (2)**:
- `column_names()`
- `has_column(name)`

**Hidden Methods via into_base_table (2)**:
- `column_names()`
- `has_column(name)`

**Summary**: 31 total methods (29 direct + 2 delegated)

## GraphArray Method Signatures

**Direct Methods (31)**:

- `__getitem__(key, /)`
- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `count()`
- `describe()`
- `drop_na()`
- `fill_na(fill_value)`
- `has_null()`
- `is_sparse` → property
- `items()`
- `max()`
- `mean()`
- `median()`
- `min()`
- `null_count()`
- `percentile(p)`
- `preview(limit=None)`
- `quantile(q)`
- `rich_display(config=None)`
- `std()`
- `summary()`
- `to_list()`
- `to_numpy()`
- `to_pandas()`
- `to_scipy_sparse()`
- `true_indices()`
- `unique()`
- `value_counts()`
- `values` → property

**Summary**: 31 total methods (31 direct + 0 delegated)

## GraphMatrix Method Signatures

**Direct Methods (16)**:

- `__getitem__(...)`
- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `clear(...)`
- `copy(...)`
- `fromkeys(iterable, value=None, /)`
- `get(key, default=None, /)`
- `items(...)`
- `keys(...)`
- `pop(...)`
- `popitem()`
- `setdefault(key, default=None, /)`
- `update(...)`
- `values(...)`

**Summary**: 16 total methods (16 direct + 0 delegated)

# Summary

**Total Methods with Signatures**: 178
- Direct methods: 172
- Delegated methods: 6

**By Object Type**:
- **Graph**: 73 methods (71 direct + 2 delegated)
- **NodesTable**: 27 methods (25 direct + 2 delegated)
- **EdgesTable**: 31 methods (29 direct + 2 delegated)
- **GraphArray**: 31 methods (31 direct + 0 delegated)
- **GraphMatrix**: 16 methods (16 direct + 0 delegated)

---
*Method signatures discovered through intelligent analysis*
