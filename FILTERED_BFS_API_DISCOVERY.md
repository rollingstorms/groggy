# Complete Groggy Python API Reference (BFS Discovery)

Generated: 2025-09-07 10:53:14

Breadth-first search discovery of ALL Groggy objects and their methods.

🔍 **Discovering all Groggy objects...**

✓ NodesTable: NodesTable
✓ EdgesTable: EdgesTable
✓ GraphTable: GraphTable
✓ GraphArray: GraphArray
✓ GraphArray (age): GraphArray
✓ GraphMatrix: dict
✓ Laplacian Matrix: GraphMatrix
✗ Subgraph failed: Attribute 'subgraph' not found. Available node attributes: ["age", "salary", "name", "active", "team"], Available edge attributes: ["type", "weight"]
✓ Neighborhood: NeighborhoodResult
✓ GraphView: Subgraph
✗ NodeFilter failed: No constructor defined
✗ EdgeFilter failed: No constructor defined

🔍 **Checking groggy module contents...**
✗ AttributeFilter: Could not instantiate
→ display: module (non-callable)
→ enhanced_query: module (non-callable)
→ errors: module (non-callable)
→ generators: module (non-callable)
→ networkx_compat: module (non-callable)
→ table_extensions: module (non-callable)
→ types: module (non-callable)

**Total objects discovered: 10**

🔍 **Starting BFS Method Discovery (Groggy objects only)...**

🌱 Starting with Graph: Graph
🌱 Starting with NodesTable: NodesTable
🌱 Starting with EdgesTable: EdgesTable
🌱 Starting with GraphTable: GraphTable
🌱 Starting with GraphArray (node_id): GraphArray
🌱 Starting with GraphArray (age): GraphArray
🌱 Starting with GraphMatrix (adjacency): dict
🌱 Starting with GraphMatrix (laplacian): GraphMatrix
🌱 Starting with Neighborhood: NeighborhoodResult
🌱 Starting with GraphView: Subgraph
🔄 Processing Graph (depth 0): Graph

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
  ⚠️  Could not test neighborhood(): 'builtins.GraphArray' object is not callable

**Delegation via table() → GraphTable**

**Delegation via view() → Subgraph**

**Hidden Methods via table (2)**:
- `merge_with(_other)`
- `to_graph()`

**Hidden Methods via view (30)**:
- `calculate_similarity(other, metric='jaccard')`
- `child_meta_nodes()`
- `clustering_coefficient(_node_id=None)`
- `collapse(node_aggs=None, edge_aggs=None, edge_strategy='aggregate', node_strategy='extract', preset=None, include_edge_count=True, mark_entity_type=True, entity_type='meta', allow_missing_attributes=True)`
- `collapse_to_node(agg_functions)`
- `collapse_to_node_with_defaults(agg_functions, defaults=None)`
- `connected_components()`
- `degree(nodes=None, *, full_graph=False)`
- `edges_table()`
- `entity_type()`
- `get_edge_attribute(edge_id, attr_name)`
- `get_node_attribute(node_id, attr_name)`
- `has_edge_between(source, target)`
- `has_meta_nodes()`
- `has_path(node1_id, node2_id)`
- `hierarchy_level` → property
- `in_degree(nodes=None, full_graph=False)`
- `induced_subgraph(nodes)`
- `intersect_with(_other)`
- `is_empty()`
- `merge_with(_other)`
- `meta_nodes()`
- `out_degree(nodes=None, full_graph=False)`
- `parent_meta_node()`
- `shortest_path_subgraph(source, target)`
- `subgraph_from_edges(edges)`
- `subtract_from(_other)`
- `summary()`
- `to_graph()`
- `transitivity()`

**Summary**: 101 total methods (71 direct + 30 delegated)
  🔗 Found: Graph.adjacency() -> GraphMatrix
  🔗 Found: Graph.dense_adjacency_matrix() -> GraphMatrix
  🔗 Found: Graph.laplacian_matrix() -> GraphMatrix
  🔗 Found: Graph.neighborhood_statistics() -> NeighborhoodStats
  🔗 Found: Graph.table() -> GraphTable
  🔗 Found: Graph.to_networkx() -> Graph
  🔗 Found: Graph.view() -> Subgraph
  🔗 Found: Graph.to_graph() -> Graph
  🔗 Found: Graph.collapse() -> MetaNode
  🔗 Found: Graph.connected_components() -> ComponentsArray
  🔗 Found: Graph.degree() -> GraphArray
  🔗 Found: Graph.edges_table() -> EdgesTable
  🔗 Found: Graph.in_degree() -> GraphArray
  🔗 Found: Graph.out_degree() -> GraphArray
🔄 Processing NodesTable (depth 0): NodesTable

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
  🔗 Found: NodesTable.__iter__() -> NodesTableRowIterator
  🔗 Found: NodesTable.base_table() -> BaseTable
  🔗 Found: NodesTable.head() -> NodesTable
  🔗 Found: NodesTable.into_base_table() -> BaseTable
  🔗 Found: NodesTable.iter() -> NodesTableIterator
  🔗 Found: NodesTable.node_ids() -> GraphArray
  🔗 Found: NodesTable.select() -> NodesTable
  🔗 Found: NodesTable.sort_by() -> NodesTable
  🔗 Found: NodesTable.tail() -> NodesTable
  🔗 Found: NodesTable.to_pandas() -> DataFrame
🔄 Processing EdgesTable (depth 0): EdgesTable

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
  🔗 Found: EdgesTable.__iter__() -> EdgesTableRowIterator
  🔗 Found: EdgesTable.base_table() -> BaseTable
  🔗 Found: EdgesTable.edge_ids() -> GraphArray
  🔗 Found: EdgesTable.head() -> EdgesTable
  🔗 Found: EdgesTable.into_base_table() -> BaseTable
  🔗 Found: EdgesTable.iter() -> EdgesTableIterator
  🔗 Found: EdgesTable.select() -> EdgesTable
  🔗 Found: EdgesTable.sort_by() -> EdgesTable
  🔗 Found: EdgesTable.sources() -> GraphArray
  🔗 Found: EdgesTable.tail() -> EdgesTable
  🔗 Found: EdgesTable.targets() -> GraphArray
  🔗 Found: EdgesTable.to_pandas() -> DataFrame
🔄 Processing GraphTable (depth 0): GraphTable

## GraphTable Method Signatures

**Direct Methods (20)**:

- `__getitem__(key, /)`
- `__len__()`
- `__repr__()`
- `__str__()`
- `edges` → property
- `from_federated_bundles(bundle_paths, domain_names=None)`
- `head(n=5)`
- `load_bundle(path)`
- `merge(tables)`
- `merge_with(other, strategy)`
- `merge_with_strategy(tables, strategy)`
- `ncols()`
- `nodes` → property
- `nrows()`
- `save_bundle(path)`
- `shape()`
- `stats()`
- `tail(n=5)`
- `to_graph()`
- `validate()`

**Summary**: 20 total methods (20 direct + 0 delegated)
  🔗 Found: GraphTable.head() -> GraphTable
  🔗 Found: GraphTable.tail() -> GraphTable
  🔗 Found: GraphTable.to_graph() -> Graph
🔄 Processing GraphArray (node_id) (depth 0): GraphArray

## GraphArray (node_id) Method Signatures

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
  🔗 Found: GraphArray (node_id).__iter__() -> GraphArrayIterator
  🔗 Found: GraphArray (node_id).describe() -> StatsSummary
  🔗 Found: GraphArray (node_id).drop_na() -> GraphArray
  🔗 Found: GraphArray (node_id).to_numpy() -> ndarray
  🔗 Found: GraphArray (node_id).to_pandas() -> Series
  🔗 Found: GraphArray (node_id).to_scipy_sparse() -> csr_matrix
  🔗 Found: GraphArray (node_id).unique() -> GraphArray
🔄 Processing GraphArray (age) (depth 0): GraphArray

## GraphArray (age) Method Signatures

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
  🔗 Found: GraphArray (age).__iter__() -> GraphArrayIterator
  🔗 Found: GraphArray (age).describe() -> StatsSummary
  🔗 Found: GraphArray (age).drop_na() -> GraphArray
  🔗 Found: GraphArray (age).to_numpy() -> ndarray
  🔗 Found: GraphArray (age).to_pandas() -> Series
  🔗 Found: GraphArray (age).to_scipy_sparse() -> csr_matrix
  🔗 Found: GraphArray (age).unique() -> GraphArray
🔄 Processing GraphMatrix (adjacency) (depth 0): dict

## GraphMatrix (adjacency) Method Signatures

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
  🔗 Found: GraphMatrix (adjacency).__iter__() -> dict_keyiterator
  🔗 Found: GraphMatrix (adjacency).items() -> dict_items
  🔗 Found: GraphMatrix (adjacency).keys() -> dict_keys
  🔗 Found: GraphMatrix (adjacency).values() -> dict_values
🔄 Processing GraphMatrix (laplacian) (depth 0): GraphMatrix

## GraphMatrix (laplacian) Method Signatures

**Direct Methods (36)**:

- `__getitem__(key, /)`
- `__iter__()`
- `__repr__()`
- `__str__()`
- `columns` → property
- `data` → property
- `dense()`
- `determinant()`
- `dtype` → property
- `elementwise_multiply(other)`
- `from_graph_attributes(_graph, _attrs, _entities)`
- `get_cell(row, col)`
- `get_column(col)`
- `get_column_by_name(name)`
- `get_row(row)`
- `identity(size)`
- `inverse()`
- `is_numeric` → property
- `is_sparse` → property
- `is_square` → property
- `is_symmetric` → property
- `iter_columns()`
- `iter_rows()`
- `mean_axis(axis)`
- `multiply(other)`
- `power(n)`
- `preview(row_limit=None, col_limit=None)`
- `rich_display(config=None)`
- `shape` → property
- `std_axis(axis)`
- `sum_axis(axis)`
- `summary()`
- `to_numpy()`
- `to_pandas()`
- `transpose()`
- `zeros(rows, cols, dtype=None)`

**Summary**: 36 total methods (36 direct + 0 delegated)
  🔗 Found: GraphMatrix (laplacian).dense() -> GraphMatrix
  🔗 Found: GraphMatrix (laplacian).to_numpy() -> ndarray
  🔗 Found: GraphMatrix (laplacian).to_pandas() -> DataFrame
  🔗 Found: GraphMatrix (laplacian).transpose() -> GraphMatrix
🔄 Processing Neighborhood (depth 0): NeighborhoodResult

## Neighborhood Method Signatures

**Direct Methods (9)**:

- `__getitem__(key, /)`
- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `execution_time_ms` → property
- `largest_neighborhood_size` → property
- `neighborhoods` → property
- `total_neighborhoods` → property

**Summary**: 9 total methods (9 direct + 0 delegated)
  🔗 Found: Neighborhood.__getitem__() -> NeighborhoodSubgraph
  🔗 Found: Neighborhood.__iter__() -> PyNeighborhoodResultIterator
🔄 Processing GraphView (depth 0): Subgraph

## GraphView Method Signatures

**Direct Methods (57)**:

- `__getitem__(key, /)`
- `__len__()`
- `__repr__()`
- `__str__()`
- `bfs(start, max_depth=None)`
- `calculate_similarity(other, metric='jaccard')`
- `child_meta_nodes()`
- `clustering_coefficient(_node_id=None)`
- `collapse(node_aggs=None, edge_aggs=None, edge_strategy='aggregate', node_strategy='extract', preset=None, include_edge_count=True, mark_entity_type=True, entity_type='meta', allow_missing_attributes=True)`
- `collapse_to_node(agg_functions)`
- `collapse_to_node_with_defaults(agg_functions, defaults=None)`
- `connected_components()`
- `contains_edge(edge_id)`
- `contains_node(node_id)`
- `degree(nodes=None, *, full_graph=False)`
- `density()`
- `dfs(start, max_depth=None)`
- `edge_count()`
- `edge_endpoints(edge_id)`
- `edge_ids` → property
- `edges` → property
- `edges_table()`
- `entity_type()`
- `filter_edges(filter)`
- `filter_nodes(filter)`
- `get_edge_attribute(edge_id, attr_name)`
- `get_node_attribute(node_id, attr_name)`
- `has_edge(edge_id)`
- `has_edge_between(source, target)`
- `has_meta_nodes()`
- `has_node(node_id)`
- `has_path(node1_id, node2_id)`
- `hierarchy_level` → property
- `in_degree(nodes=None, full_graph=False)`
- `induced_subgraph(nodes)`
- `intersect_with(_other)`
- `is_connected()`
- `is_empty()`
- `merge_with(_other)`
- `meta_nodes()`
- `neighborhood(central_nodes, hops)`
- `neighbors(node_id)`
- `node_count()`
- `node_ids` → property
- `nodes` → property
- `out_degree(nodes=None, full_graph=False)`
- `parent_meta_node()`
- `set_edge_attrs(attrs_dict)`
- `set_node_attrs(attrs_dict)`
- `shortest_path_subgraph(source, target)`
- `subgraph_from_edges(edges)`
- `subtract_from(_other)`
- `summary()`
- `table()`
- `to_graph()`
- `to_networkx()`
- `transitivity()`
  ⚠️  Could not test neighborhood(): 'builtins.GraphArray' object is not callable

**Delegation via table() → NodesTable**

**Summary**: 57 total methods (57 direct + 0 delegated)
  🔗 Found: GraphView.collapse() -> MetaNode
  🔗 Found: GraphView.connected_components() -> ComponentsArray
  🔗 Found: GraphView.degree() -> GraphArray
  🔗 Found: GraphView.edges_table() -> EdgesTable
  🔗 Found: GraphView.in_degree() -> GraphArray
  🔗 Found: GraphView.out_degree() -> GraphArray
  🔗 Found: GraphView.table() -> NodesTable
  🔗 Found: GraphView.to_graph() -> Graph
  🔗 Found: GraphView.to_networkx() -> Graph
  🔄 Processing Graph.adjacency() (depth 1): GraphMatrix
  ⚡ Skipping GraphMatrix - already analyzed this type
  🔄 Processing Graph.dense_adjacency_matrix() (depth 1): GraphMatrix
  ⚡ Skipping GraphMatrix - already analyzed this type
  🔄 Processing Graph.laplacian_matrix() (depth 1): GraphMatrix
  ⚡ Skipping GraphMatrix - already analyzed this type
  🔄 Processing Graph.neighborhood_statistics() (depth 1): NeighborhoodStats

## Graph.neighborhood_statistics() Method Signatures

**Direct Methods (8)**:

- `__repr__()`
- `__str__()`
- `avg_nodes_per_neighborhood()`
- `avg_time_per_neighborhood_ms()`
- `operation_counts` → property
- `total_neighborhoods` → property
- `total_nodes_sampled` → property
- `total_time_ms` → property

**Summary**: 8 total methods (8 direct + 0 delegated)
  🔄 Processing Graph.table() (depth 1): GraphTable
  ⚡ Skipping GraphTable - already analyzed this type
  🔄 Processing Graph.to_networkx() (depth 1): Graph
  ⚡ Skipping Graph - already analyzed this type
  🔄 Processing Graph.to_graph() (depth 1): Graph
  ⚡ Skipping Graph - already analyzed this type
  🔄 Processing Graph.collapse() (depth 1): MetaNode

## Graph.collapse() Method Signatures

**Direct Methods (17)**:

- `__getitem__(key, /)`
- `__repr__()`
- `__str__()`
- `degree` → property
- `entity_type` → property
- `expand()`
- `has_subgraph` → property
- `id` → property
- `is_active` → property
- `keys()`
- `meta_edges` → property
- `neighbors` → property
- `re_aggregate(agg_functions)`
- `subgraph` → property
- `subgraph_id` → property
- `summary()`
- `values()`

**Summary**: 17 total methods (17 direct + 0 delegated)
    🔗 Found: Graph.collapse().expand() -> Subgraph
  🔄 Processing Graph.connected_components() (depth 1): ComponentsArray

## Graph.connected_components() Method Signatures

**Direct Methods (9)**:

- `__getitem__(key, /)`
- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `iter()`
- `largest_component()`
- `sizes()`
- `to_list()`

**Summary**: 9 total methods (9 direct + 0 delegated)
    🔗 Found: Graph.connected_components().__getitem__() -> Subgraph
    🔗 Found: Graph.connected_components().__iter__() -> PyComponentsArrayIterator
    🔗 Found: Graph.connected_components().iter() -> ComponentsIterator
    🔗 Found: Graph.connected_components().largest_component() -> Subgraph
  🔄 Processing Graph.degree() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing Graph.edges_table() (depth 1): EdgesTable
  ⚡ Skipping EdgesTable - already analyzed this type
  🔄 Processing Graph.in_degree() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing Graph.out_degree() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing NodesTable.__iter__() (depth 1): NodesTableRowIterator

## NodesTable.__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
  🔄 Processing NodesTable.base_table() (depth 1): BaseTable

## NodesTable.base_table() Method Signatures

**Direct Methods (21)**:

- `__getitem__(key, /)`
- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `column_names()`
- `drop_columns(columns)`
- `filter(predicate)`
- `group_by(columns)`
- `has_column(name)`
- `head(n=5)`
- `iter()`
- `ncols()`
- `nrows()`
- `rich_display(config=None)`
- `select(columns)`
- `shape()`
- `slice(start, end)`
- `sort_by(column, ascending=True)`
- `tail(n=5)`
- `to_pandas()`

**Summary**: 21 total methods (21 direct + 0 delegated)
    🔗 Found: NodesTable.base_table().__iter__() -> BaseTableRowIterator
    🔗 Found: NodesTable.base_table().head() -> BaseTable
    🔗 Found: NodesTable.base_table().iter() -> BaseTableIterator
    🔗 Found: NodesTable.base_table().select() -> BaseTable
    🔗 Found: NodesTable.base_table().sort_by() -> BaseTable
    🔗 Found: NodesTable.base_table().tail() -> BaseTable
    🔗 Found: NodesTable.base_table().to_pandas() -> DataFrame
  🔄 Processing NodesTable.head() (depth 1): NodesTable
  ⚡ Skipping NodesTable - already analyzed this type
  🔄 Processing NodesTable.into_base_table() (depth 1): BaseTable
  ⚡ Skipping BaseTable - already analyzed this type
  🔄 Processing NodesTable.iter() (depth 1): NodesTableIterator

## NodesTable.iter() Method Signatures

**Direct Methods (3)**:

- `__repr__()`
- `__str__()`
- `collect()`

**Summary**: 3 total methods (3 direct + 0 delegated)
    🔗 Found: NodesTable.iter().collect() -> NodesTable
  🔄 Processing NodesTable.node_ids() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing NodesTable.select() (depth 1): NodesTable
  ⚡ Skipping NodesTable - already analyzed this type
  🔄 Processing NodesTable.sort_by() (depth 1): NodesTable
  ⚡ Skipping NodesTable - already analyzed this type
  🔄 Processing NodesTable.tail() (depth 1): NodesTable
  ⚡ Skipping NodesTable - already analyzed this type
  🔄 Processing EdgesTable.__iter__() (depth 1): EdgesTableRowIterator

## EdgesTable.__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
  🔄 Processing EdgesTable.base_table() (depth 1): BaseTable
  ⚡ Skipping BaseTable - already analyzed this type
  🔄 Processing EdgesTable.edge_ids() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing EdgesTable.head() (depth 1): EdgesTable
  ⚡ Skipping EdgesTable - already analyzed this type
  🔄 Processing EdgesTable.into_base_table() (depth 1): BaseTable
  ⚡ Skipping BaseTable - already analyzed this type
  🔄 Processing EdgesTable.iter() (depth 1): EdgesTableIterator

## EdgesTable.iter() Method Signatures

**Direct Methods (3)**:

- `__repr__()`
- `__str__()`
- `collect()`

**Summary**: 3 total methods (3 direct + 0 delegated)
    🔗 Found: EdgesTable.iter().collect() -> EdgesTable
  🔄 Processing EdgesTable.select() (depth 1): EdgesTable
  ⚡ Skipping EdgesTable - already analyzed this type
  🔄 Processing EdgesTable.sort_by() (depth 1): EdgesTable
  ⚡ Skipping EdgesTable - already analyzed this type
  🔄 Processing EdgesTable.sources() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing EdgesTable.tail() (depth 1): EdgesTable
  ⚡ Skipping EdgesTable - already analyzed this type
  🔄 Processing EdgesTable.targets() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing GraphTable.head() (depth 1): GraphTable
  ⚡ Skipping GraphTable - already analyzed this type
  🔄 Processing GraphTable.tail() (depth 1): GraphTable
  ⚡ Skipping GraphTable - already analyzed this type
  🔄 Processing GraphTable.to_graph() (depth 1): Graph
  ⚡ Skipping Graph - already analyzed this type
  🔄 Processing GraphArray (node_id).__iter__() (depth 1): GraphArrayIterator

## GraphArray (node_id).__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
  🔄 Processing GraphArray (node_id).describe() (depth 1): StatsSummary

## GraphArray (node_id).describe() Method Signatures

**Direct Methods (10)**:

- `__repr__()`
- `__str__()`
- `count` → property
- `max` → property
- `mean` → property
- `median` → property
- `min` → property
- `q25` → property
- `q75` → property
- `std` → property

**Summary**: 10 total methods (10 direct + 0 delegated)
  🔄 Processing GraphArray (node_id).drop_na() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing GraphArray (node_id).unique() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing GraphArray (age).__iter__() (depth 1): GraphArrayIterator
  ⚡ Skipping GraphArrayIterator - already analyzed this type
  🔄 Processing GraphArray (age).describe() (depth 1): StatsSummary
  ⚡ Skipping StatsSummary - already analyzed this type
  🔄 Processing GraphArray (age).drop_na() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing GraphArray (age).unique() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing GraphMatrix (adjacency).__iter__() (depth 1): dict_keyiterator

## GraphMatrix (adjacency).__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
  🔄 Processing GraphMatrix (adjacency).items() (depth 1): dict_items

## GraphMatrix (adjacency).items() Method Signatures

**Direct Methods (5)**:

- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `isdisjoint(...)`

**Summary**: 5 total methods (5 direct + 0 delegated)
    🔗 Found: GraphMatrix (adjacency).items().__iter__() -> dict_itemiterator
  🔄 Processing GraphMatrix (adjacency).keys() (depth 1): dict_keys

## GraphMatrix (adjacency).keys() Method Signatures

**Direct Methods (5)**:

- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`
- `isdisjoint(...)`

**Summary**: 5 total methods (5 direct + 0 delegated)
    🔗 Found: GraphMatrix (adjacency).keys().__iter__() -> dict_keyiterator
  🔄 Processing GraphMatrix (adjacency).values() (depth 1): dict_values

## GraphMatrix (adjacency).values() Method Signatures

**Direct Methods (4)**:

- `__iter__()`
- `__len__()`
- `__repr__()`
- `__str__()`

**Summary**: 4 total methods (4 direct + 0 delegated)
    🔗 Found: GraphMatrix (adjacency).values().__iter__() -> dict_valueiterator
  🔄 Processing GraphMatrix (laplacian).dense() (depth 1): GraphMatrix
  ⚡ Skipping GraphMatrix - already analyzed this type
  🔄 Processing GraphMatrix (laplacian).transpose() (depth 1): GraphMatrix
  ⚡ Skipping GraphMatrix - already analyzed this type
  🔄 Processing Neighborhood.__getitem__() (depth 1): NeighborhoodSubgraph

## Neighborhood.__getitem__() Method Signatures

**Direct Methods (6)**:

- `__repr__()`
- `__str__()`
- `central_nodes` → property
- `hops` → property
- `is_central_node(node_id)`
- `subgraph()`
  ⚠️  Could not test subgraph(): 'builtins.GraphArray' object is not callable

**Summary**: 6 total methods (6 direct + 0 delegated)
    🔗 Found: Neighborhood.__getitem__().subgraph() -> Subgraph
  🔄 Processing Neighborhood.__iter__() (depth 1): PyNeighborhoodResultIterator

## Neighborhood.__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
  🔄 Processing GraphView.collapse() (depth 1): MetaNode
  ⚡ Skipping MetaNode - already analyzed this type
  🔄 Processing GraphView.connected_components() (depth 1): ComponentsArray
  ⚡ Skipping ComponentsArray - already analyzed this type
  🔄 Processing GraphView.degree() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing GraphView.edges_table() (depth 1): EdgesTable
  ⚡ Skipping EdgesTable - already analyzed this type
  🔄 Processing GraphView.in_degree() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing GraphView.out_degree() (depth 1): GraphArray
  ⚡ Skipping GraphArray - already analyzed this type
  🔄 Processing GraphView.table() (depth 1): NodesTable
  ⚡ Skipping NodesTable - already analyzed this type
  🔄 Processing GraphView.to_graph() (depth 1): Graph
  ⚡ Skipping Graph - already analyzed this type
  🔄 Processing GraphView.to_networkx() (depth 1): Graph
  ⚡ Skipping Graph - already analyzed this type
    🔄 Processing Graph.collapse().expand() (depth 2): Subgraph
    ⚡ Skipping Subgraph - already analyzed this type
    🔄 Processing Graph.connected_components().__getitem__() (depth 2): Subgraph
    ⚡ Skipping Subgraph - already analyzed this type
    🔄 Processing Graph.connected_components().__iter__() (depth 2): PyComponentsArrayIterator

## Graph.connected_components().__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
    🔄 Processing Graph.connected_components().iter() (depth 2): ComponentsIterator

## Graph.connected_components().iter() Method Signatures

**Direct Methods (6)**:

- `__repr__()`
- `__str__()`
- `collapse(aggs)`
- `collect()`
- `filter_edges(query)`
- `filter_nodes(query)`

**Summary**: 6 total methods (6 direct + 0 delegated)
    🔄 Processing Graph.connected_components().largest_component() (depth 2): Subgraph
    ⚡ Skipping Subgraph - already analyzed this type
    🔄 Processing NodesTable.base_table().__iter__() (depth 2): BaseTableRowIterator

## NodesTable.base_table().__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
    🔄 Processing NodesTable.base_table().head() (depth 2): BaseTable
    ⚡ Skipping BaseTable - already analyzed this type
    🔄 Processing NodesTable.base_table().iter() (depth 2): BaseTableIterator

## NodesTable.base_table().iter() Method Signatures

**Direct Methods (3)**:

- `__repr__()`
- `__str__()`
- `collect()`

**Summary**: 3 total methods (3 direct + 0 delegated)
    🔄 Processing NodesTable.base_table().select() (depth 2): BaseTable
    ⚡ Skipping BaseTable - already analyzed this type
    🔄 Processing NodesTable.base_table().sort_by() (depth 2): BaseTable
    ⚡ Skipping BaseTable - already analyzed this type
    🔄 Processing NodesTable.base_table().tail() (depth 2): BaseTable
    ⚡ Skipping BaseTable - already analyzed this type
    🔄 Processing NodesTable.iter().collect() (depth 2): NodesTable
    ⚡ Skipping NodesTable - already analyzed this type
    🔄 Processing EdgesTable.iter().collect() (depth 2): EdgesTable
    ⚡ Skipping EdgesTable - already analyzed this type
    🔄 Processing GraphMatrix (adjacency).items().__iter__() (depth 2): dict_itemiterator

## GraphMatrix (adjacency).items().__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
    🔄 Processing GraphMatrix (adjacency).keys().__iter__() (depth 2): dict_keyiterator
    ⚡ Skipping dict_keyiterator - already analyzed this type
    🔄 Processing GraphMatrix (adjacency).values().__iter__() (depth 2): dict_valueiterator

## GraphMatrix (adjacency).values().__iter__() Method Signatures

**Direct Methods (3)**:

- `__iter__()`
- `__repr__()`
- `__str__()`

**Summary**: 3 total methods (3 direct + 0 delegated)
    🔄 Processing Neighborhood.__getitem__().subgraph() (depth 2): Subgraph
    ⚡ Skipping Subgraph - already analyzed this type

✅ **BFS Discovery Complete**: Found 32 unique Groggy objects

# Complete API Reference

## Depth 0 Objects (10 objects)

### Graph
**Type**: `Graph`
**Methods**: 101 (71 direct + 30 delegated)
**Key Methods**: `add_edge`, `add_edges`, `add_graph`, `add_node`, `add_nodes`, `adjacency`, `adjacency_matrix`, `aggregate`, `all_edge_attribute_names`, `all_node_attribute_names`
... and 91 more

### NodesTable
**Type**: `NodesTable`
**Methods**: 27 (25 direct + 2 delegated)
**Key Methods**: `base_table`, `drop_columns`, `filter`, `filter_by_attr`, `group_by`, `head`, `into_base_table`, `iter`, `ncols`, `node_ids`
... and 17 more

### EdgesTable
**Type**: `EdgesTable`
**Methods**: 31 (29 direct + 2 delegated)
**Key Methods**: `as_tuples`, `base_table`, `drop_columns`, `edge_ids`, `filter`, `filter_by_attr`, `filter_by_sources`, `filter_by_targets`, `group_by`, `head`
... and 21 more

### GraphTable
**Type**: `GraphTable`
**Methods**: 20 (20 direct + 0 delegated)
**Key Methods**: `edges`, `from_federated_bundles`, `head`, `load_bundle`, `merge`, `merge_with`, `merge_with_strategy`, `ncols`, `nodes`, `nrows`
... and 10 more

### GraphArray (node_id)
**Type**: `GraphArray`
**Methods**: 31 (31 direct + 0 delegated)
**Key Methods**: `count`, `describe`, `drop_na`, `fill_na`, `has_null`, `is_sparse`, `items`, `max`, `mean`, `median`
... and 21 more

### GraphArray (age)
**Type**: `GraphArray`
**Methods**: 31 (31 direct + 0 delegated)
**Key Methods**: `count`, `describe`, `drop_na`, `fill_na`, `has_null`, `is_sparse`, `items`, `max`, `mean`, `median`
... and 21 more

### GraphMatrix (adjacency)
**Type**: `dict`
**Methods**: 16 (16 direct + 0 delegated)
**Key Methods**: `clear`, `copy`, `fromkeys`, `get`, `items`, `keys`, `pop`, `popitem`, `setdefault`, `update`
... and 6 more

### GraphMatrix (laplacian)
**Type**: `GraphMatrix`
**Methods**: 36 (36 direct + 0 delegated)
**Key Methods**: `columns`, `data`, `dense`, `determinant`, `dtype`, `elementwise_multiply`, `from_graph_attributes`, `get_cell`, `get_column`, `get_column_by_name`
... and 26 more

### Neighborhood
**Type**: `NeighborhoodResult`
**Methods**: 9 (9 direct + 0 delegated)
**Key Methods**: `execution_time_ms`, `largest_neighborhood_size`, `neighborhoods`, `total_neighborhoods`

### GraphView
**Type**: `Subgraph`
**Methods**: 57 (57 direct + 0 delegated)
**Key Methods**: `bfs`, `calculate_similarity`, `child_meta_nodes`, `clustering_coefficient`, `collapse`, `collapse_to_node`, `collapse_to_node_with_defaults`, `connected_components`, `contains_edge`, `contains_node`
... and 47 more

## Depth 1 Objects (16 objects)

### Graph.neighborhood_statistics()
**Type**: `NeighborhoodStats`
**Methods**: 8 (8 direct + 0 delegated)
**Key Methods**: `avg_nodes_per_neighborhood`, `avg_time_per_neighborhood_ms`, `operation_counts`, `total_neighborhoods`, `total_nodes_sampled`, `total_time_ms`

### Graph.collapse()
**Type**: `MetaNode`
**Methods**: 17 (17 direct + 0 delegated)
**Key Methods**: `degree`, `entity_type`, `expand`, `has_subgraph`, `id`, `is_active`, `keys`, `meta_edges`, `neighbors`, `re_aggregate`
... and 7 more

### Graph.connected_components()
**Type**: `ComponentsArray`
**Methods**: 9 (9 direct + 0 delegated)
**Key Methods**: `iter`, `largest_component`, `sizes`, `to_list`

### NodesTable.__iter__()
**Type**: `NodesTableRowIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

### NodesTable.base_table()
**Type**: `BaseTable`
**Methods**: 21 (21 direct + 0 delegated)
**Key Methods**: `column_names`, `drop_columns`, `filter`, `group_by`, `has_column`, `head`, `iter`, `ncols`, `nrows`, `rich_display`
... and 11 more

### NodesTable.iter()
**Type**: `NodesTableIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: `collect`

### EdgesTable.__iter__()
**Type**: `EdgesTableRowIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

### EdgesTable.iter()
**Type**: `EdgesTableIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: `collect`

### GraphArray (node_id).__iter__()
**Type**: `GraphArrayIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

### GraphArray (node_id).describe()
**Type**: `StatsSummary`
**Methods**: 10 (10 direct + 0 delegated)
**Key Methods**: `count`, `max`, `mean`, `median`, `min`, `q25`, `q75`, `std`

### GraphMatrix (adjacency).__iter__()
**Type**: `dict_keyiterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

### GraphMatrix (adjacency).items()
**Type**: `dict_items`
**Methods**: 5 (5 direct + 0 delegated)
**Key Methods**: `isdisjoint`

### GraphMatrix (adjacency).keys()
**Type**: `dict_keys`
**Methods**: 5 (5 direct + 0 delegated)
**Key Methods**: `isdisjoint`

### GraphMatrix (adjacency).values()
**Type**: `dict_values`
**Methods**: 4 (4 direct + 0 delegated)
**Key Methods**: 

### Neighborhood.__getitem__()
**Type**: `NeighborhoodSubgraph`
**Methods**: 6 (6 direct + 0 delegated)
**Key Methods**: `central_nodes`, `hops`, `is_central_node`, `subgraph`

### Neighborhood.__iter__()
**Type**: `PyNeighborhoodResultIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

## Depth 2 Objects (6 objects)

### Graph.connected_components().__iter__()
**Type**: `PyComponentsArrayIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

### Graph.connected_components().iter()
**Type**: `ComponentsIterator`
**Methods**: 6 (6 direct + 0 delegated)
**Key Methods**: `collapse`, `collect`, `filter_edges`, `filter_nodes`

### NodesTable.base_table().__iter__()
**Type**: `BaseTableRowIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

### NodesTable.base_table().iter()
**Type**: `BaseTableIterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: `collect`

### GraphMatrix (adjacency).items().__iter__()
**Type**: `dict_itemiterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

### GraphMatrix (adjacency).values().__iter__()
**Type**: `dict_valueiterator`
**Methods**: 3 (3 direct + 0 delegated)
**Key Methods**: 

# BFS Discovery Summary

**Total Objects Discovered**: 32
**Total Methods Discovered**: 486
**Average Methods per Object**: 15.2

**Discovery by Depth**:
- **Depth 0**: 10 objects, 359 methods
- **Depth 1**: 16 objects, 106 methods
- **Depth 2**: 6 objects, 21 methods

---
*Complete API surface discovered through BFS traversal*
