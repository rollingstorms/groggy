# Graph-to-matrix construction

Adjacency matrices are owned numerical snapshots computed from graph membership
and property references at call time. A Subgraph still reads its live parent;
this does not make the source view a frozen historical snapshot.

- Full graphs use ascending node IDs on both axes. Explicit node lists retain
  caller order. Labels are `node_ID` on both axes.
- Every edge instance contributes. Parallel edge weights add; an unweighted
  edge contributes one. Undirected edges contribute in both directions except
  self-loops, which contribute once. A matrix aggregates facts and is not a
  lossless container for individual edge identities.
- Unknown endpoints, duplicate node IDs, duplicate selected edge IDs, stale
  explicit IDs and invalid weight references produce errors.
- Finite floating contributions accumulate accurately before rounding; integer
  accumulation and result narrowing are checked. Nonfinite floating weights
  remain represented; downstream operators apply their own domain constraints.
- Missing weights default to one. Explicit null retains the legacy attribute
  conversion policy of zero. Invalid text is an error instead of a unit weight.
- Native `sparse_adjacency_matrix()` and `MatrixFormat::Sparse` produce canonical
  CSR without a dense intermediate. Construction uses O(nodes + edges) scratch
  and sorts contributions in O(edges log edges). Dense output explicitly converts
  the resulting operator. Raw-ID coordinates (`compact_indexing=false`) are
  rejected; compact coordinates carry original IDs as labels.

Python exports live on graph views:

```python
matrix = graph.view().adjacency_matrix(sparse=True, weight_attr="weight")
# adj(...) is an alias, with the same optional keywords.
```

A view export respects its selected edge IDs, rather than adding all parent
edges between its nodes. Active membership is checked against the parent at
compute time. Mutating the graph later does not change the returned matrix.

For generic rectangular `from_row_major_data(..., nodes=...)`, IDs describe
rows and must match the row count; non-square columns retain `col_N` names.
