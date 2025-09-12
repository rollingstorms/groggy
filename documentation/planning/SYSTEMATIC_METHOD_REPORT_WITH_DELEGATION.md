# Groggy Systematic Method Testing Report

Generated: 2025-09-07 10:27:44

This report systematically tests every method of every major object type in Groggy using Python introspection.
It discovers both direct methods (from `dir()`) and delegated methods (available through `__getattr__`).

## Legend
- 🔵 Direct method (available in `dir()`)
- 🔵📄 Direct property
- 🟡 Delegated method (available through `__getattr__`)
- 🟡📄 Delegated property
- ✅ Method works correctly
- ❌ Method fails with error


## Graph Methods

| Method | Type | Status | Error |
|--------|------|--------|-------|
| `__len__` | 🔵 | ✅ Working | - |
| `__repr__` | 🔵 | ✅ Working | - |
| `__str__` | 🔵 | ✅ Working | - |
| `add_edge` | 🔵 | ✅ Working | - |
| `add_edges` | 🔵 | ✅ Working | - |
| `add_graph` | 🔵 | ❌ Failing | Graph.add_graph() missing 1 required positional argument: 'o... |
| `add_node` | 🔵 | ✅ Working | - |
| `add_nodes` | 🔵 | ✅ Working | - |
| `adjacency` | 🔵 | ✅ Working | - |
| `adjacency_matrix` | 🔵 | ✅ Working | - |
| `aggregate` | 🔵 | ❌ Failing | Graph.aggregate() missing 2 required positional arguments: '... |
| `all_edge_attribute_names` | 🔵 | ✅ Working | - |
| `all_node_attribute_names` | 🔵 | ✅ Working | - |
| `bfs` | 🔵 | ✅ Working | - |
| `branches` | 🔵 | ✅ Working | - |
| `checkout_branch` | 🔵 | ❌ Failing | Graph.checkout_branch() missing 1 required positional argume... |
| `clustering_coefficient` | 🟡 | ❌ Failing | Clustering coefficient not yet implemented in core - coming ... |
| `commit` | 🔵 | ❌ Failing | Graph.commit() missing 2 required positional arguments: 'mes... |
| `commit_history` | 🔵 | ✅ Working | - |
| `contains_edge` | 🔵 | ✅ Working | - |
| `contains_node` | 🔵 | ✅ Working | - |
| `create_branch` | 🔵 | ❌ Failing | Graph.create_branch() missing 1 required positional argument... |
| `dense_adjacency_matrix` | 🔵 | ✅ Working | - |
| `density` | 🔵 | ✅ Working | - |
| `dfs` | 🔵 | ✅ Working | - |
| `edge_attribute_keys` | 🔵 | ❌ Failing | Graph.edge_attribute_keys() missing 1 required positional ar... |
| `edge_count` | 🔵 | ✅ Working | - |
| `edge_endpoints` | 🔵 | ✅ Working | - |
| `edge_ids` | 🔵📄 | ✅ Working | - |
| `edges` | 🔵📄 | ✅ Working | - |
| `filter_edges` | 🔵 | ❌ Failing | Graph.filter_edges() missing 1 required positional argument:... |
| `filter_nodes` | 🔵 | ❌ Failing | Graph.filter_nodes() missing 1 required positional argument:... |
| `get_edge_attr` | 🔵 | ❌ Failing | Graph.get_edge_attr() missing 1 required positional argument... |
| `get_edge_attrs` | 🔵 | ✅ Working | - |
| `get_node_attr` | 🔵 | ✅ Working | - |
| `get_node_attrs` | 🔵 | ✅ Working | - |
| `get_node_mapping` | 🔵 | ❌ Failing | Graph.get_node_mapping() missing 1 required positional argum... |
| `group_by` | 🔵 | ❌ Failing | Graph.group_by() missing 3 required positional arguments: 'a... |
| `group_nodes_by_attribute` | 🔵 | ❌ Failing | Graph.group_nodes_by_attribute() missing 3 required position... |
| `has_edge` | 🔵 | ✅ Working | - |
| `has_edge_attribute` | 🔵 | ❌ Failing | Graph.has_edge_attribute() missing 2 required positional arg... |
| `has_node` | 🔵 | ✅ Working | - |
| `has_node_attribute` | 🔵 | ✅ Working | - |
| `has_uncommitted_changes` | 🔵 | ✅ Working | - |
| `historical_view` | 🔵 | ❌ Failing | Graph.historical_view() missing 1 required positional argume... |
| `is_connected` | 🔵 | ✅ Working | - |
| `is_directed` | 🔵📄 | ✅ Working | - |
| `is_undirected` | 🔵📄 | ✅ Working | - |
| `laplacian_matrix` | 🔵 | ✅ Working | - |
| `neighborhood` | 🔵 | ❌ Failing | Graph.neighborhood() missing 1 required positional argument:... |
| `neighborhood_statistics` | 🔵 | ✅ Working | - |
| `neighbors` | 🔵 | ✅ Working | - |
| `node_attribute_keys` | 🔵 | ❌ Failing | Graph.node_attribute_keys() missing 1 required positional ar... |
| `node_count` | 🔵 | ✅ Working | - |
| `node_ids` | 🔵📄 | ✅ Working | - |
| `nodes` | 🔵📄 | ✅ Working | - |
| `remove_edge` | 🔵 | ✅ Working | - |
| `remove_edges` | 🔵 | ❌ Failing | Graph.remove_edges() missing 1 required positional argument:... |
| `remove_node` | 🔵 | ✅ Working | - |
| `remove_nodes` | 🔵 | ❌ Failing | Graph.remove_nodes() missing 1 required positional argument:... |
| `resolve_string_id_to_node` | 🔵 | ❌ Failing | Graph.resolve_string_id_to_node() missing 2 required positio... |
| `set_edge_attr` | 🔵 | ❌ Failing | Graph.set_edge_attr() missing 2 required positional argument... |
| `set_edge_attrs` | 🔵 | ❌ Failing | 'str' object cannot be interpreted as an integer |
| `set_node_attr` | 🔵 | ❌ Failing | Graph.set_node_attr() missing 1 required positional argument... |
| `set_node_attrs` | 🔵 | ❌ Failing | 'str' object cannot be interpreted as an integer |
| `shortest_path` | 🔵 | ✅ Working | - |
| `sparse_adjacency_matrix` | 🔵 | ✅ Working | - |
| `table` | 🔵 | ✅ Working | - |
| `to_networkx` | 🔵 | ✅ Working | - |
| `transition_matrix` | 🔵 | ❌ Failing | transition_matrix needs to be implemented in core first |
| `transitivity` | 🟡 | ❌ Failing | Transitivity not yet implemented in core - coming in future ... |
| `view` | 🔵 | ✅ Working | - |
| `weighted_adjacency_matrix` | 🔵 | ✅ Working | - |

**Summary**: 47 working, 26 failing out of 73 total methods
- Direct methods: 71
- Delegated methods: 2


## NodesTable Methods

| Method | Type | Status | Error |
|--------|------|--------|-------|
| `__getitem__` | 🔵 | ❌ Failing | expected 1 argument, got 2 |
| `__iter__` | 🔵 | ✅ Working | - |
| `__len__` | 🔵 | ✅ Working | - |
| `__repr__` | 🔵 | ✅ Working | - |
| `__str__` | 🔵 | ✅ Working | - |
| `base_table` | 🔵 | ✅ Working | - |
| `column_names` | 🟡 | ✅ Working | - |
| `drop_columns` | 🔵 | ❌ Failing | NodesTable.drop_columns() missing 1 required positional argu... |
| `filter` | 🔵 | ❌ Failing | NodesTable.filter() missing 1 required positional argument: ... |
| `filter_by_attr` | 🔵 | ❌ Failing | NodesTable.filter_by_attr() missing 2 required positional ar... |
| `group_by` | 🔵 | ❌ Failing | NodesTable.group_by() missing 1 required positional argument... |
| `head` | 🔵 | ✅ Working | - |
| `into_base_table` | 🔵 | ✅ Working | - |
| `iter` | 🔵 | ✅ Working | - |
| `ncols` | 🔵 | ✅ Working | - |
| `node_ids` | 🔵 | ✅ Working | - |
| `nrows` | 🔵 | ✅ Working | - |
| `rich_display` | 🔵 | ✅ Working | - |
| `select` | 🔵 | ✅ Working | - |
| `shape` | 🔵 | ✅ Working | - |
| `slice` | 🔵 | ❌ Failing | NodesTable.slice() missing 2 required positional arguments: ... |
| `sort_by` | 🔵 | ✅ Working | - |
| `tail` | 🔵 | ✅ Working | - |
| `to_pandas` | 🔵 | ✅ Working | - |
| `unique_attr_values` | 🔵 | ❌ Failing | NodesTable.unique_attr_values() missing 1 required positiona... |
| `with_attributes` | 🔵 | ❌ Failing | NodesTable.with_attributes() missing 2 required positional a... |

**Summary**: 18 working, 8 failing out of 26 total methods
- Direct methods: 25
- Delegated methods: 1


## EdgesTable Methods

| Method | Type | Status | Error |
|--------|------|--------|-------|
| `__getitem__` | 🔵 | ❌ Failing | expected 1 argument, got 2 |
| `__iter__` | 🔵 | ✅ Working | - |
| `__len__` | 🔵 | ✅ Working | - |
| `__repr__` | 🔵 | ✅ Working | - |
| `__str__` | 🔵 | ✅ Working | - |
| `as_tuples` | 🔵 | ✅ Working | - |
| `base_table` | 🔵 | ✅ Working | - |
| `column_names` | 🟡 | ✅ Working | - |
| `drop_columns` | 🔵 | ❌ Failing | EdgesTable.drop_columns() missing 1 required positional argu... |
| `edge_ids` | 🔵 | ✅ Working | - |
| `filter` | 🔵 | ❌ Failing | EdgesTable.filter() missing 1 required positional argument: ... |
| `filter_by_attr` | 🔵 | ❌ Failing | EdgesTable.filter_by_attr() missing 2 required positional ar... |
| `filter_by_sources` | 🔵 | ❌ Failing | EdgesTable.filter_by_sources() missing 1 required positional... |
| `filter_by_targets` | 🔵 | ❌ Failing | EdgesTable.filter_by_targets() missing 1 required positional... |
| `group_by` | 🔵 | ❌ Failing | EdgesTable.group_by() missing 1 required positional argument... |
| `head` | 🔵 | ✅ Working | - |
| `into_base_table` | 🔵 | ✅ Working | - |
| `iter` | 🔵 | ✅ Working | - |
| `ncols` | 🔵 | ✅ Working | - |
| `nrows` | 🔵 | ✅ Working | - |
| `rich_display` | 🔵 | ✅ Working | - |
| `select` | 🔵 | ✅ Working | - |
| `shape` | 🔵 | ✅ Working | - |
| `slice` | 🔵 | ❌ Failing | EdgesTable.slice() missing 2 required positional arguments: ... |
| `sort_by` | 🔵 | ✅ Working | - |
| `sources` | 🔵 | ✅ Working | - |
| `tail` | 🔵 | ✅ Working | - |
| `targets` | 🔵 | ✅ Working | - |
| `to_pandas` | 🔵 | ✅ Working | - |
| `unique_attr_values` | 🔵 | ❌ Failing | EdgesTable.unique_attr_values() missing 1 required positiona... |

**Summary**: 21 working, 9 failing out of 30 total methods
- Direct methods: 29
- Delegated methods: 1


## GraphArray Methods

| Method | Type | Status | Error |
|--------|------|--------|-------|
| `__getitem__` | 🔵 | ✅ Working | - |
| `__iter__` | 🔵 | ✅ Working | - |
| `__len__` | 🔵 | ✅ Working | - |
| `__repr__` | 🔵 | ✅ Working | - |
| `__str__` | 🔵 | ✅ Working | - |
| `count` | 🔵 | ✅ Working | - |
| `describe` | 🔵 | ✅ Working | - |
| `drop_na` | 🔵 | ✅ Working | - |
| `fill_na` | 🔵 | ❌ Failing | GraphArray.fill_na() missing 1 required positional argument:... |
| `has_null` | 🔵 | ✅ Working | - |
| `is_sparse` | 🔵📄 | ✅ Working | - |
| `items` | 🔵 | ✅ Working | - |
| `max` | 🔵 | ✅ Working | - |
| `mean` | 🔵 | ✅ Working | - |
| `median` | 🔵 | ✅ Working | - |
| `min` | 🔵 | ✅ Working | - |
| `null_count` | 🔵 | ✅ Working | - |
| `percentile` | 🔵 | ❌ Failing | GraphArray.percentile() missing 1 required positional argume... |
| `preview` | 🔵 | ✅ Working | - |
| `quantile` | 🔵 | ❌ Failing | GraphArray.quantile() missing 1 required positional argument... |
| `rich_display` | 🔵 | ✅ Working | - |
| `std` | 🔵 | ✅ Working | - |
| `summary` | 🔵 | ✅ Working | - |
| `to_list` | 🔵 | ✅ Working | - |
| `to_numpy` | 🔵 | ✅ Working | - |
| `to_pandas` | 🔵 | ✅ Working | - |
| `to_scipy_sparse` | 🔵 | ✅ Working | - |
| `true_indices` | 🔵 | ✅ Working | - |
| `unique` | 🔵 | ✅ Working | - |
| `value_counts` | 🔵 | ✅ Working | - |
| `values` | 🔵📄 | ✅ Working | - |

**Summary**: 28 working, 3 failing out of 31 total methods
- Direct methods: 31
- Delegated methods: 0


## GraphMatrix Methods

| Method | Type | Status | Error |
|--------|------|--------|-------|
| `__getitem__` | 🔵 | ❌ Failing | dict.__getitem__() takes exactly one argument (0 given) |
| `__iter__` | 🔵 | ✅ Working | - |
| `__len__` | 🔵 | ✅ Working | - |
| `__repr__` | 🔵 | ✅ Working | - |
| `__str__` | 🔵 | ✅ Working | - |
| `clear` | 🔵 | ✅ Working | - |
| `copy` | 🔵 | ✅ Working | - |
| `fromkeys` | 🔵 | ❌ Failing | fromkeys expected at least 1 argument, got 0 |
| `get` | 🔵 | ❌ Failing | get expected at least 1 argument, got 0 |
| `items` | 🔵 | ✅ Working | - |
| `keys` | 🔵 | ✅ Working | - |
| `pop` | 🔵 | ❌ Failing | pop expected at least 1 argument, got 0 |
| `popitem` | 🔵 | ❌ Failing | 'popitem(): dictionary is empty' |
| `setdefault` | 🔵 | ❌ Failing | setdefault expected at least 1 argument, got 0 |
| `update` | 🔵 | ✅ Working | - |
| `values` | 🔵 | ✅ Working | - |

**Summary**: 10 working, 6 failing out of 16 total methods
- Direct methods: 16
- Delegated methods: 0

## Subgraph Testing Failed

Error: Attribute 'subgraph' not found. Available node attributes: ["age", "name", "team", "salary", "active"], Available edge attributes: ["weight", "type"]

# Overall Summary

- **Total Working Methods**: 124
- **Total Failing Methods**: 52
- **Total Methods Tested**: 176
- **Success Rate**: 70.5%

## Methods Needing Work

These methods currently fail and need implementation or bug fixes:

- **Graph.add_graph**: Graph.add_graph() missing 1 required positional argument: 'other'
- **Graph.aggregate**: Graph.aggregate() missing 2 required positional arguments: 'attribute' and 'operation'
- **Graph.checkout_branch**: Graph.checkout_branch() missing 1 required positional argument: 'branch_name'
- **Graph.clustering_coefficient**: Clustering coefficient not yet implemented in core - coming in future version
- **Graph.commit**: Graph.commit() missing 2 required positional arguments: 'message' and 'author'
- **Graph.create_branch**: Graph.create_branch() missing 1 required positional argument: 'branch_name'
- **Graph.edge_attribute_keys**: Graph.edge_attribute_keys() missing 1 required positional argument: 'edge_id'
- **Graph.filter_edges**: Graph.filter_edges() missing 1 required positional argument: 'filter'
- **Graph.filter_nodes**: Graph.filter_nodes() missing 1 required positional argument: 'filter'
- **Graph.get_edge_attr**: Graph.get_edge_attr() missing 1 required positional argument: 'attr'
- **Graph.get_node_mapping**: Graph.get_node_mapping() missing 1 required positional argument: 'uid_key'
- **Graph.group_by**: Graph.group_by() missing 3 required positional arguments: 'attribute', 'aggregation_attr', and 'operation'
- **Graph.group_nodes_by_attribute**: Graph.group_nodes_by_attribute() missing 3 required positional arguments: 'attribute', 'aggregation_attr', and 'operation'
- **Graph.has_edge_attribute**: Graph.has_edge_attribute() missing 2 required positional arguments: 'edge_id' and 'attr_name'
- **Graph.historical_view**: Graph.historical_view() missing 1 required positional argument: 'commit_id'
- **Graph.neighborhood**: Graph.neighborhood() missing 1 required positional argument: 'center_nodes'
- **Graph.node_attribute_keys**: Graph.node_attribute_keys() missing 1 required positional argument: 'node_id'
- **Graph.remove_edges**: Graph.remove_edges() missing 1 required positional argument: 'edges'
- **Graph.remove_nodes**: Graph.remove_nodes() missing 1 required positional argument: 'nodes'
- **Graph.resolve_string_id_to_node**: Graph.resolve_string_id_to_node() missing 2 required positional arguments: 'string_id' and 'uid_key'
- **Graph.set_edge_attr**: Graph.set_edge_attr() missing 2 required positional arguments: 'attr' and 'value'
- **Graph.set_edge_attrs**: 'str' object cannot be interpreted as an integer
- **Graph.set_node_attr**: Graph.set_node_attr() missing 1 required positional argument: 'value'
- **Graph.set_node_attrs**: 'str' object cannot be interpreted as an integer
- **Graph.transition_matrix**: transition_matrix needs to be implemented in core first
- **Graph.transitivity**: Transitivity not yet implemented in core - coming in future version
- **NodesTable.__getitem__**: expected 1 argument, got 2
- **NodesTable.drop_columns**: NodesTable.drop_columns() missing 1 required positional argument: 'columns'
- **NodesTable.filter**: NodesTable.filter() missing 1 required positional argument: 'predicate'
- **NodesTable.filter_by_attr**: NodesTable.filter_by_attr() missing 2 required positional arguments: 'attr_name' and 'value'
- **NodesTable.group_by**: NodesTable.group_by() missing 1 required positional argument: 'columns'
- **NodesTable.slice**: NodesTable.slice() missing 2 required positional arguments: 'start' and 'end'
- **NodesTable.unique_attr_values**: NodesTable.unique_attr_values() missing 1 required positional argument: 'attr_name'
- **NodesTable.with_attributes**: NodesTable.with_attributes() missing 2 required positional arguments: 'attr_name' and 'attributes'
- **EdgesTable.__getitem__**: expected 1 argument, got 2
- **EdgesTable.drop_columns**: EdgesTable.drop_columns() missing 1 required positional argument: 'columns'
- **EdgesTable.filter**: EdgesTable.filter() missing 1 required positional argument: 'predicate'
- **EdgesTable.filter_by_attr**: EdgesTable.filter_by_attr() missing 2 required positional arguments: 'attr_name' and 'value'
- **EdgesTable.filter_by_sources**: EdgesTable.filter_by_sources() missing 1 required positional argument: 'source_nodes'
- **EdgesTable.filter_by_targets**: EdgesTable.filter_by_targets() missing 1 required positional argument: 'target_nodes'
- **EdgesTable.group_by**: EdgesTable.group_by() missing 1 required positional argument: 'columns'
- **EdgesTable.slice**: EdgesTable.slice() missing 2 required positional arguments: 'start' and 'end'
- **EdgesTable.unique_attr_values**: EdgesTable.unique_attr_values() missing 1 required positional argument: 'attr_name'
- **GraphArray.fill_na**: GraphArray.fill_na() missing 1 required positional argument: 'fill_value'
- **GraphArray.percentile**: GraphArray.percentile() missing 1 required positional argument: 'p'
- **GraphArray.quantile**: GraphArray.quantile() missing 1 required positional argument: 'q'
- **GraphMatrix.__getitem__**: dict.__getitem__() takes exactly one argument (0 given)
- **GraphMatrix.fromkeys**: fromkeys expected at least 1 argument, got 0
- **GraphMatrix.get**: get expected at least 1 argument, got 0
- **GraphMatrix.pop**: pop expected at least 1 argument, got 0
- **GraphMatrix.popitem**: 'popitem(): dictionary is empty'
- **GraphMatrix.setdefault**: setdefault expected at least 1 argument, got 0

---
*Generated by systematic method testing using Python introspection*
