# Comprehensive Groggy API Test Results

Generated: 2025-09-07 13:21:21

Systematic testing of all methods on all Groggy objects using intelligent argument inference.

🏗️ **Creating comprehensive test objects...**

✅ Created 12 test objects:
  - Graph: Graph
  - NodesTable: NodesTable
  - EdgesTable: EdgesTable
  - GraphArray_node_ids: GraphArray
  - GraphArray_ages: GraphArray
  - GraphArray_names: GraphArray
  - GraphMatrix_adjacency: dict
  - GraphMatrix_laplacian: GraphMatrix
  - GraphTable: GraphTable
  - Subgraph: Subgraph
  - NeighborhoodResult: NeighborhoodResult
  - BaseTable: BaseTable

# Test Results

## Graph (Graph)

**Testing 71 methods:**

✅ `__len__()` → 5
✅ `__repr__()` → "Graph(nodes=5, edges=5)"
✅ `__str__()` → "Graph(nodes=5, edges=5)"
❌ `add_edge(source, target, uid_key=None, **kwargs)` → Missing arguments: Graph.add_edge() missing 2 required positional arguments: 'source' and 'target'
❌ `add_edges(edges, node_mapping=None, _uid_key=None, warm_cache=None)` → Missing arguments: Graph.add_edges() missing 1 required positional argument: 'edges'
✅ `add_graph(other)` → None
✅ `add_node(**kwargs)` → 6
✅ `add_nodes(data, uid_key=None)` → list[2]
✅ `adjacency()` → GraphMatrix
✅ `adjacency_matrix()` → dict[4]
✅ `aggregate(attribute, operation, target=None, _node_ids=None)` → AggregationResult
✅ `all_edge_attribute_names()` → list[3]
✅ `all_node_attribute_names()` → list[6]
❌ `bfs(start, max_depth=None, inplace=None, attr_name=None)` → Missing arguments: Graph.bfs() missing 1 required positional argument: 'start'
✅ `branches()` → list[1]
❌ `checkout_branch(branch_name)` → Not found: Branch 'test_branch' not found
✅ `commit(message, author)` → 1
✅ `commit_history()` → list[1]
❌ `contains_edge(edge)` → Missing arguments: Graph.contains_edge() missing 1 required positional argument: 'edge'
❌ `contains_node(node)` → Missing arguments: Graph.contains_node() missing 1 required positional argument: 'node'
✅ `create_branch(branch_name)` → None
✅ `dense_adjacency_matrix()` → GraphMatrix
✅ `density()` → 0.1388888888888889
❌ `dfs(start, max_depth=None, inplace=None, attr_name=None)` → Missing arguments: Graph.dfs() missing 1 required positional argument: 'start'
❌ `edge_attribute_keys(edge_id)` → Missing arguments: Graph.edge_attribute_keys() missing 1 required positional argument: 'edge_id'
✅ `edge_count()` → 5
❌ `edge_endpoints(edge)` → Missing arguments: Graph.edge_endpoints() missing 1 required positional argument: 'edge'
✅ `edge_ids(...)` → GraphArray[5]
✅ `edges(...)` → EdgesAccessor[5]
❌ `filter_edges(filter)` → Skipped - complex method
❌ `filter_nodes(filter)` → Skipped - complex method
❌ `get_edge_attr(edge, attr, default=None)` → Missing arguments: Graph.get_edge_attr() missing 2 required positional arguments: 'edge' and 'attr'
❌ `get_edge_attrs(edges, attrs)` → Missing arguments: Graph.get_edge_attrs() missing 2 required positional arguments: 'edges' and 'attrs'
❌ `get_node_attr(node, attr, default=None)` → Missing arguments: Graph.get_node_attr() missing 2 required positional arguments: 'node' and 'attr'
❌ `get_node_attrs(nodes, attrs)` → Missing arguments: Graph.get_node_attrs() missing 2 required positional arguments: 'nodes' and 'attrs'
✅ `get_node_mapping(uid_key, return_inverse=False)` → dict[9]
❌ `group_by(attribute, aggregation_attr, operation)` → Missing arguments: Graph.group_by() missing 2 required positional arguments: 'aggregation_attr' and 'operation'
✅ `group_nodes_by_attribute(attribute, aggregation_attr, operation)` → GroupedAggregationResult
❌ `has_edge(edge_id)` → Missing arguments: Graph.has_edge() missing 1 required positional argument: 'edge_id'
❌ `has_edge_attribute(edge_id, attr_name)` → Missing arguments: Graph.has_edge_attribute() missing 2 required positional arguments: 'edge_id' and 'attr_name'
❌ `has_node(node_id)` → Missing arguments: Graph.has_node() missing 1 required positional argument: 'node_id'
❌ `has_node_attribute(node_id, attr_name)` → Missing arguments: Graph.has_node_attribute() missing 2 required positional arguments: 'node_id' and 'attr_name'
✅ `has_uncommitted_changes()` → False
❌ `historical_view(commit_id)` → argument 'commit_id': 'str' object cannot be interpreted as an integer
✅ `is_connected()` → False
✅ `is_directed(...)` → False
✅ `is_undirected(...)` → True
✅ `laplacian_matrix(normalized=None)` → GraphMatrix
❌ `neighborhood(center_nodes, radius=None, max_nodes=None)` → Missing arguments: Graph.neighborhood() missing 1 required positional argument: 'center_nodes'
✅ `neighborhood_statistics()` → NeighborhoodStats
❌ `neighbors(nodes=None)` → nodes parameter is required
❌ `node_attribute_keys(node_id)` → Missing arguments: Graph.node_attribute_keys() missing 1 required positional argument: 'node_id'
✅ `node_count()` → 9
✅ `node_ids(...)` → GraphArray[5]
✅ `nodes(...)` → NodesAccessor[9]
❌ `remove_edge(edge)` → Missing arguments: Graph.remove_edge() missing 1 required positional argument: 'edge'
❌ `remove_edges(edges)` → Missing arguments: Graph.remove_edges() missing 1 required positional argument: 'edges'
❌ `remove_node(node)` → Missing arguments: Graph.remove_node() missing 1 required positional argument: 'node'
❌ `remove_nodes(nodes)` → Missing arguments: Graph.remove_nodes() missing 1 required positional argument: 'nodes'
✅ `resolve_string_id_to_node(string_id, uid_key)` → 0
❌ `set_edge_attr(edge, attr, value)` → Missing arguments: Graph.set_edge_attr() missing 3 required positional arguments: 'edge', 'attr', and 'value'
❌ `set_edge_attrs(attrs_dict)` → Unsupported attribute format. Expected: dict {id: value}, dict {"nodes": [...], "values": [...]}, GraphArray, or GraphTable. Got: str
❌ `set_node_attr(node, attr, value)` → Missing arguments: Graph.set_node_attr() missing 3 required positional arguments: 'node', 'attr', and 'value'
❌ `set_node_attrs(attrs_dict)` → Unsupported attribute format. Expected: dict {id: value}, dict {"nodes": [...], "values": [...]}, GraphArray, or GraphTable. Got: str
❌ `shortest_path(source, target, weight_attribute=None, inplace=None, attr_name=None)` → Missing arguments: Graph.shortest_path() missing 2 required positional arguments: 'source' and 'target'
✅ `sparse_adjacency_matrix()` → dict[4]
✅ `table()` → GraphTable[14]
✅ `to_networkx(directed: bool = False, include_attributes: bool = True)` → Graph[9]
❌ `transition_matrix()` → transition_matrix needs to be implemented in core first
✅ `view()` → Subgraph[9]
✅ `weighted_adjacency_matrix(weight_attr)` → GraphMatrix

**Graph Results**: 37/71 methods working (52.1%)

## NodesTable (NodesTable)

**Testing 25 methods:**

✅ `__getitem__(key, /)` → GraphArray[5]
✅ `__iter__()` → NodesTableRowIterator
✅ `__len__()` → 5
✅ `__repr__()` → "NodesTable[5 x 7]"
✅ `__str__()` → "NodesTable[5 x 7]
BaseTable[5 x 7]
|       name | ..."
✅ `base_table()` → BaseTable[5]
✅ `drop_columns(columns)` → NodesTable[5]
❌ `filter(predicate)` → Skipped - complex method
❌ `filter_by_attr(attr_name, value)` → Skipped - complex method
❌ `group_by(columns)` → NotImplemented { feature: "group_by for BaseTable", tracking_issue: None }
✅ `head(n=5)` → NodesTable[3]
✅ `into_base_table()` → BaseTable[5]
✅ `iter()` → NodesTableIterator
✅ `ncols()` → 7
✅ `node_ids()` → GraphArray[5]
✅ `nrows()` → 5
✅ `rich_display(config=None)` → "⊖⊖ gr.table
╭─────────┬───────────────┬───────────..."
✅ `select(columns)` → NodesTable[5]
✅ `shape()` → tuple[2]
✅ `slice(start, end)` → NodesTable[3]
✅ `sort_by(column, ascending=True)` → NodesTable[5]
✅ `tail(n=5)` → NodesTable[3]
✅ `to_pandas()` → DataFrame[5]
✅ `unique_attr_values(attr_name)` → list[5]
❌ `with_attributes(attr_name, attributes)` → Missing arguments: NodesTable.with_attributes() missing 2 required positional arguments: 'attr_name' and 'attributes'

**NodesTable Results**: 21/25 methods working (84.0%)

## EdgesTable (EdgesTable)

**Testing 29 methods:**

✅ `__getitem__(key, /)` → GraphArray[5]
✅ `__iter__()` → EdgesTableRowIterator
✅ `__len__()` → 5
✅ `__repr__()` → "EdgesTable[5 x 6]"
✅ `__str__()` → "EdgesTable[5 x 6]
BaseTable[5 x 6]
|     source | ..."
✅ `as_tuples()` → list[5]
✅ `base_table()` → BaseTable[5]
❌ `drop_columns(columns)` → Cannot drop 'source' column from EdgesTable
✅ `edge_ids()` → GraphArray[5]
❌ `filter(predicate)` → Skipped - complex method
❌ `filter_by_attr(attr_name, value)` → Skipped - complex method
❌ `filter_by_sources(source_nodes)` → Skipped - complex method
❌ `filter_by_targets(target_nodes)` → Skipped - complex method
❌ `group_by(columns)` → NotImplemented { feature: "group_by for BaseTable", tracking_issue: None }
✅ `head(n=5)` → EdgesTable[3]
✅ `into_base_table()` → BaseTable[5]
✅ `iter()` → EdgesTableIterator
✅ `ncols()` → 6
✅ `nrows()` → 5
✅ `rich_display(config=None)` → "⊖⊖ gr.table
╭───────────┬────────────────┬────────..."
✅ `select(columns)` → EdgesTable[5]
✅ `shape()` → tuple[2]
✅ `slice(start, end)` → EdgesTable[3]
✅ `sort_by(column, ascending=True)` → EdgesTable[5]
✅ `sources()` → GraphArray[5]
✅ `tail(n=5)` → EdgesTable[3]
✅ `targets()` → GraphArray[5]
✅ `to_pandas()` → DataFrame[5]
✅ `unique_attr_values(attr_name)` → list[3]

**EdgesTable Results**: 23/29 methods working (79.3%)

## GraphArray_node_ids (GraphArray)

**Testing 31 methods:**

✅ `__getitem__(key, /)` → 0
✅ `__iter__()` → GraphArrayIterator
✅ `__len__()` → 5
✅ `__repr__()` → "GraphArray(len=5, dtype=int64)"
✅ `__str__()` → "GraphArray(len=5, dtype=int64)"
✅ `count()` → 5
✅ `describe()` → StatsSummary
✅ `drop_na()` → GraphArray[5]
✅ `fill_na(fill_value)` → GraphArray[5]
✅ `has_null()` → False
✅ `is_sparse(...)` → False
✅ `items()` → list[5]
✅ `max()` → 4
✅ `mean()` → 2.0
✅ `median()` → 2.0
✅ `min()` → 0
✅ `null_count()` → 0
✅ `percentile(p)` → 0.0
✅ `preview(limit=None)` → list[5]
✅ `quantile(q)` → 2.0
✅ `rich_display(config=None)` → "╭───┬───────╮
│ # │ value │
│   │ obj   │
├───┼───..."
✅ `std()` → 1.5811388300841898
✅ `summary()` → "GraphArray('unnamed', length=5, dtype=Int, sparse=..."
✅ `to_list()` → list[5]
✅ `to_numpy()` → ndarray[5]
✅ `to_pandas()` → Series[5]
✅ `to_scipy_sparse()` → csr_matrix
✅ `true_indices()` → list[0]
✅ `unique()` → GraphArray[5]
✅ `value_counts()` → dict[5]
✅ `values(...)` → list[5]

**GraphArray_node_ids Results**: 31/31 methods working (100.0%)

## GraphArray_ages (GraphArray)

**Testing 31 methods:**

✅ `__getitem__(key, /)` → 25
✅ `__iter__()` → GraphArrayIterator
✅ `__len__()` → 5
✅ `__repr__()` → "GraphArray(len=5, dtype=int64)"
✅ `__str__()` → "GraphArray(len=5, dtype=int64)"
✅ `count()` → 5
✅ `describe()` → StatsSummary
✅ `drop_na()` → GraphArray[5]
✅ `fill_na(fill_value)` → GraphArray[5]
✅ `has_null()` → False
✅ `is_sparse(...)` → False
✅ `items()` → list[5]
✅ `max()` → 35
✅ `mean()` → 30.0
✅ `median()` → 30.0
✅ `min()` → 25
✅ `null_count()` → 0
✅ `percentile(p)` → 25.0
✅ `preview(limit=None)` → list[5]
✅ `quantile(q)` → 30.0
✅ `rich_display(config=None)` → "╭───┬───────╮
│ # │ value │
│   │ obj   │
├───┼───..."
✅ `std()` → 3.8078865529319543
✅ `summary()` → "GraphArray('unnamed', length=5, dtype=Int, sparse=..."
✅ `to_list()` → list[5]
✅ `to_numpy()` → ndarray[5]
✅ `to_pandas()` → Series[5]
✅ `to_scipy_sparse()` → csr_matrix
✅ `true_indices()` → list[0]
✅ `unique()` → GraphArray[5]
✅ `value_counts()` → dict[5]
✅ `values(...)` → list[5]

**GraphArray_ages Results**: 31/31 methods working (100.0%)

## GraphArray_names (GraphArray)

**Testing 31 methods:**

✅ `__getitem__(key, /)` → "Alice"
✅ `__iter__()` → GraphArrayIterator
✅ `__len__()` → 5
✅ `__repr__()` → "GraphArray(len=5, dtype=str)"
✅ `__str__()` → "GraphArray(len=5, dtype=str)"
✅ `count()` → 5
✅ `describe()` → StatsSummary
✅ `drop_na()` → GraphArray[5]
✅ `fill_na(fill_value)` → GraphArray[5]
✅ `has_null()` → False
✅ `is_sparse(...)` → False
✅ `items()` → list[5]
✅ `max()` → "Eve"
✅ `mean()` → None
✅ `median()` → None
✅ `min()` → "Alice"
✅ `null_count()` → 0
✅ `percentile(p)` → None
✅ `preview(limit=None)` → list[5]
✅ `quantile(q)` → None
✅ `rich_display(config=None)` → "╭───┬─────────╮
│ # │ value   │
│   │ obj     │
├─..."
✅ `std()` → None
✅ `summary()` → "GraphArray('unnamed', length=5, dtype=Text, sparse..."
✅ `to_list()` → list[5]
✅ `to_numpy()` → ndarray[5]
✅ `to_pandas()` → Series[5]
❌ `to_scipy_sparse()` → unsupported data types in input
✅ `true_indices()` → list[0]
✅ `unique()` → GraphArray[5]
✅ `value_counts()` → dict[5]
✅ `values(...)` → list[5]

**GraphArray_names Results**: 30/31 methods working (96.8%)

## GraphMatrix_adjacency (dict)

**Testing 16 methods:**

❌ `__getitem__(...)` → dict.__getitem__() takes exactly one argument (0 given)
✅ `__iter__()` → dict_keyiterator
✅ `__len__()` → 4
✅ `__repr__()` → "{'size': 5, 'is_sparse': True, 'type': 'adjacency_..."
✅ `__str__()` → "{'size': 5, 'is_sparse': True, 'type': 'adjacency_..."
✅ `clear(...)` → None
✅ `copy(...)` → dict[0]
❌ `fromkeys(iterable, value=None, /)` → fromkeys expected at least 1 argument, got 0
❌ `get(key, default=None, /)` → get expected at least 1 argument, got 0
✅ `items(...)` → dict_items[0]
✅ `keys(...)` → dict_keys[0]
❌ `pop(...)` → pop expected at least 1 argument, got 0
❌ `popitem()` → 'popitem(): dictionary is empty'
❌ `setdefault(key, default=None, /)` → setdefault expected at least 1 argument, got 0
✅ `update(...)` → None
✅ `values(...)` → dict_values[0]

**GraphMatrix_adjacency Results**: 10/16 methods working (62.5%)

## GraphMatrix_laplacian (GraphMatrix)

**Testing 36 methods:**

❌ `__getitem__(key, /)` → expected 1 argument, got 0
❌ `__iter__()` → Matrix iteration temporarily disabled during Phase 3 - use matrix[i] for row access
✅ `__repr__()` → "GraphMatrix(5 x 5, dtype=Float)"
✅ `__str__()` → "GraphMatrix(5 x 5, dtype=Float)"
✅ `columns(...)` → list[5]
✅ `data(...)` → list[5]
✅ `dense()` → GraphMatrix
❌ `determinant()` → Determinant calculation will be implemented in Phase 5
✅ `dtype(...)` → "Float"
❌ `elementwise_multiply(other)` → Missing arguments: GraphMatrix.elementwise_multiply() missing 1 required positional argument: 'other'
❌ `from_graph_attributes(_graph, _attrs, _entities)` → argument '_entities': Can't extract `str` to `Vec`
✅ `get_cell(row, col)` → 2.0
✅ `get_column(col)` → GraphArray[5]
❌ `get_column_by_name(name)` → Missing arguments: GraphMatrix.get_column_by_name() missing 1 required positional argument: 'name'
✅ `get_row(row)` → GraphArray[5]
✅ `identity(size)` → GraphMatrix
❌ `inverse()` → Matrix inverse will be implemented in Phase 5
✅ `is_numeric(...)` → True
✅ `is_sparse(...)` → False
✅ `is_square(...)` → True
✅ `is_symmetric(...)` → False
✅ `iter_columns()` → list[5]
✅ `iter_rows()` → list[5]
✅ `mean_axis(axis)` → GraphArray[5]
❌ `multiply(other)` → Missing arguments: GraphMatrix.multiply() missing 1 required positional argument: 'other'
✅ `power(n)` → GraphMatrix
✅ `preview(row_limit=None, col_limit=None)` → list[5]
✅ `rich_display(config=None)` → "╭───────┬───────┬───────┬───────┬───────╮
│ col_0 ..."
✅ `shape(...)` → tuple[2]
✅ `std_axis(axis)` → GraphArray[5]
✅ `sum_axis(axis)` → GraphArray[5]
✅ `summary()` → "GraphMatrix(shape=(5, 5), dtype=Float, sparse=fals..."
✅ `to_numpy()` → ndarray[5]
✅ `to_pandas()` → DataFrame[5]
✅ `transpose()` → GraphMatrix
✅ `zeros(rows, cols, dtype=None)` → GraphMatrix

**GraphMatrix_laplacian Results**: 28/36 methods working (77.8%)

## GraphTable (GraphTable)

**Testing 20 methods:**

❌ `__getitem__(key, /)` → GraphTable indices must be strings (column names)
✅ `__len__()` → 10
✅ `__repr__()` → "GraphTable[
  NodesTable: 5 rows × 7 cols
  EdgesT..."
✅ `__str__()` → "GraphTable[5 nodes, 5 edges]
Validation Policy: St..."
✅ `edges(...)` → EdgesTable[5]
❌ `from_federated_bundles(bundle_paths, domain_names=None)` → argument 'bundle_paths': Can't extract `str` to `Vec`
✅ `head(n=5)` → GraphTable[8]
❌ `load_bundle(path)` → InvalidInput("Bundle path does not exist: test_path.bundle")
❌ `merge(tables)` → InvalidInput("Cannot merge empty list of tables")
❌ `merge_with(other, strategy)` → Missing arguments: GraphTable.merge_with() missing 2 required positional arguments: 'other' and 'strategy'
❌ `merge_with_strategy(tables, strategy)` → Unknown conflict resolution strategy: union
✅ `ncols()` → 13
✅ `nodes(...)` → NodesTable[5]
✅ `nrows()` → 10
✅ `save_bundle(path)` → None
✅ `shape()` → tuple[2]
✅ `stats()` → dict[8]
✅ `tail(n=5)` → GraphTable[8]
✅ `to_graph()` → Graph[5]
✅ `validate()` → "ValidationReport { errors: [], warnings: [], info:..."

**GraphTable Results**: 14/20 methods working (70.0%)

## Subgraph (Subgraph)

**Testing 57 methods:**

❌ `__getitem__(key, /)` → Subgraph indexing only supports string attribute names. Example: subgraph['community']
✅ `__len__()` → 5
✅ `__repr__()` → "Subgraph(nodes=5, edges=5)"
✅ `__str__()` → "Subgraph with 5 nodes and 5 edges

Edges:
  ID    ..."
❌ `bfs(start, max_depth=None)` → Missing arguments: Subgraph.bfs() missing 1 required positional argument: 'start'
✅ `calculate_similarity(other, metric='jaccard')` → 1.0
✅ `child_meta_nodes()` → list[0]
❌ `clustering_coefficient(_node_id=None)` → Clustering coefficient not yet implemented in core - coming in future version
✅ `collapse(node_aggs=None, edge_aggs=None, edge_strategy='aggregate', node_strategy='extract', preset=None, include_edge_count=True, mark_entity_type=True, entity_type='meta', allow_missing_attributes=True)` → MetaNode
❌ `collapse_to_node(agg_functions)` → Missing arguments: Subgraph.collapse_to_node() missing 1 required positional argument: 'agg_functions'
❌ `collapse_to_node_with_defaults(agg_functions, defaults=None)` → Missing arguments: Subgraph.collapse_to_node_with_defaults() missing 1 required positional argument: 'agg_functions'
✅ `connected_components()` → ComponentsArray[1]
❌ `contains_edge(edge_id)` → Missing arguments: Subgraph.contains_edge() missing 1 required positional argument: 'edge_id'
❌ `contains_node(node_id)` → Missing arguments: Subgraph.contains_node() missing 1 required positional argument: 'node_id'
✅ `degree(nodes=None, *, full_graph=False)` → GraphArray[5]
✅ `density()` → 0.5
❌ `dfs(start, max_depth=None)` → Missing arguments: Subgraph.dfs() missing 1 required positional argument: 'start'
✅ `edge_count()` → 5
❌ `edge_endpoints(edge_id)` → Missing arguments: Subgraph.edge_endpoints() missing 1 required positional argument: 'edge_id'
✅ `edge_ids(...)` → GraphArray[5]
✅ `edges(...)` → EdgesAccessor[5]
✅ `edges_table()` → EdgesTable[5]
✅ `entity_type()` → "Subgraph"
❌ `filter_edges(filter)` → Skipped - complex method
❌ `filter_nodes(filter)` → Skipped - complex method
❌ `get_edge_attribute(edge_id, attr_name)` → Missing arguments: Subgraph.get_edge_attribute() missing 2 required positional arguments: 'edge_id' and 'attr_name'
❌ `get_node_attribute(node_id, attr_name)` → Missing arguments: Subgraph.get_node_attribute() missing 2 required positional arguments: 'node_id' and 'attr_name'
❌ `has_edge(edge_id)` → Missing arguments: Subgraph.has_edge() missing 1 required positional argument: 'edge_id'
❌ `has_edge_between(source, target)` → Missing arguments: Subgraph.has_edge_between() missing 2 required positional arguments: 'source' and 'target'
✅ `has_meta_nodes()` → False
❌ `has_node(node_id)` → Missing arguments: Subgraph.has_node() missing 1 required positional argument: 'node_id'
❌ `has_path(node1_id, node2_id)` → Missing arguments: Subgraph.has_path() missing 2 required positional arguments: 'node1_id' and 'node2_id'
✅ `hierarchy_level(...)` → 0
✅ `in_degree(nodes=None, full_graph=False)` → GraphArray[5]
❌ `induced_subgraph(nodes)` → Missing arguments: Subgraph.induced_subgraph() missing 1 required positional argument: 'nodes'
❌ `intersect_with(_other)` → Subgraph set operations not yet implemented - requires subgraph algebra in core
✅ `is_connected()` → True
✅ `is_empty()` → False
❌ `merge_with(_other)` → Subgraph set operations not yet implemented - requires subgraph algebra in core
✅ `meta_nodes()` → list[0]
❌ `neighborhood(central_nodes, hops)` → Missing arguments: Subgraph.neighborhood() missing 2 required positional arguments: 'central_nodes' and 'hops'
❌ `neighbors(node_id)` → Missing arguments: Subgraph.neighbors() missing 1 required positional argument: 'node_id'
✅ `node_count()` → 5
✅ `node_ids(...)` → GraphArray[5]
✅ `nodes(...)` → NodesAccessor[5]
✅ `out_degree(nodes=None, full_graph=False)` → GraphArray[5]
✅ `parent_meta_node()` → None
❌ `set_edge_attrs(attrs_dict)` → Missing arguments: Subgraph.set_edge_attrs() missing 1 required positional argument: 'attrs_dict'
❌ `set_node_attrs(attrs_dict)` → Missing arguments: Subgraph.set_node_attrs() missing 1 required positional argument: 'attrs_dict'
❌ `shortest_path_subgraph(source, target)` → Missing arguments: Subgraph.shortest_path_subgraph() missing 2 required positional arguments: 'source' and 'target'
❌ `subgraph_from_edges(edges)` → Missing arguments: Subgraph.subgraph_from_edges() missing 1 required positional argument: 'edges'
❌ `subtract_from(_other)` → Subgraph set operations not yet implemented - requires subgraph algebra in core
✅ `summary()` → "Subgraph: 5 nodes, 5 edges, density: 0.500"
✅ `table()` → NodesTable[5]
✅ `to_graph()` → Graph[0]
✅ `to_networkx()` → Graph[5]
❌ `transitivity()` → Transitivity not yet implemented in core - coming in future version

**Subgraph Results**: 29/57 methods working (50.9%)

## NeighborhoodResult (NeighborhoodResult)

**Testing 9 methods:**

✅ `__getitem__(key, /)` → NeighborhoodSubgraph
✅ `__iter__()` → PyNeighborhoodResultIterator
✅ `__len__()` → 1
✅ `__repr__()` → "NeighborhoodResult(1 neighborhoods, largest_size=5..."
✅ `__str__()` → "NeighborhoodResult(1 neighborhoods, largest_size=5..."
✅ `execution_time_ms(...)` → 0.0
✅ `largest_neighborhood_size(...)` → 5
✅ `neighborhoods(...)` → list[1]
✅ `total_neighborhoods(...)` → 1

**NeighborhoodResult Results**: 9/9 methods working (100.0%)

## BaseTable (BaseTable)

**Testing 21 methods:**

✅ `__getitem__(key, /)` → GraphArray[5]
✅ `__iter__()` → BaseTableRowIterator
✅ `__len__()` → 5
✅ `__repr__()` → "BaseTable[5 x 7]"
✅ `__str__()` → "BaseTable[5 x 7]"
✅ `column_names()` → list[7]
✅ `drop_columns(columns)` → BaseTable[5]
❌ `filter(predicate)` → Skipped - complex method
❌ `group_by(columns)` → NotImplemented { feature: "group_by for BaseTable", tracking_issue: None }
✅ `has_column(name)` → True
✅ `head(n=5)` → BaseTable[3]
✅ `iter()` → BaseTableIterator
✅ `ncols()` → 7
✅ `nrows()` → 5
✅ `rich_display(config=None)` → "⊖⊖ gr.table
╭─────────┬───────────────┬───────────..."
✅ `select(columns)` → BaseTable[5]
✅ `shape()` → tuple[2]
✅ `slice(start, end)` → BaseTable[3]
✅ `sort_by(column, ascending=True)` → BaseTable[5]
✅ `tail(n=5)` → BaseTable[3]
✅ `to_pandas()` → DataFrame[5]

**BaseTable Results**: 19/21 methods working (90.5%)

# Test Summary

**Overall Results**: 282/377 methods working (74.8%)
---
*Comprehensive API testing complete*

# Generating CSV output...
✅ CSV saved to: groggy_comprehensive_api_test_results.csv
📊 DataFrame shape: (377, 10)
📋 Columns: ['object_name', 'object_type', 'method_name', 'method_signature', 'test_status', 'success', 'return_type', 'return_preview', 'error_message', 'execution_time_ms']

## Sample Results (first 5 rows):

  object_name object_type method_name                                            method_signature test_status  success return_type             return_preview                                                                                       error_message  execution_time_ms
0       Graph       Graph     __len__                                                          ()        PASS     True         int                          5                                                                                                None              0.002
1       Graph       Graph    __repr__                                                          ()        PASS     True         str  "Graph(nodes=5, edges=5)"                                                                                                None              0.022
2       Graph       Graph     __str__                                                          ()        PASS     True         str  "Graph(nodes=5, edges=5)"                                                                                                None              0.002
3       Graph       Graph    add_edge                    (source, target, uid_key=None, **kwargs)        FAIL    False        None                       None  Missing arguments: Graph.add_edge() missing 2 required positional arguments: 'source' and 'target'              0.000
4       Graph       Graph   add_edges  (edges, node_mapping=None, _uid_key=None, warm_cache=None)        FAIL    False        None                       None                Missing arguments: Graph.add_edges() missing 1 required positional argument: 'edges'              0.000
