# Canonical arrays and shared numeric primitives

`storage::array::BaseArray<T>` owns the shared column representation. The older
`storage::array::base::BaseArray` remains a compatibility metadata adapter with
its existing constructors, declared dtype and optional name. It stores a
canonical BaseArray<AttrValue>, rather than a separate Vec implementation.

`from_column`, `as_column` and `into_column` bridge the legacy adapter without
copying values. Clones may share immutable values; explicit `set`/`data_mut` use
copy-on-write so existing owned snapshots retain their contents. NumArray and
BoolArray keep their typed roles. This does not make graph-connected tables or
Subgraph values state-pinned; those ownership contracts remain AR4/26 work.

Compatible selection/reduction/ID operations delegate to canonical/shared native
implementations. Selection rejects every invalid position instead of silently
shortening output; boolean masks must match length. Canonical slice(start,length)
clamps safely without overflow; the legacy slice(start,end) keeps its end-index
convention through a thin adapter. Legacy text/bool min/max and metadata semantics
remain explicit compatibility behavior.

ID construction has checked try_from_node_ids/try_from_edge_ids variants and
rejects values outside signed attribute storage range. Infallible legacy wrappers
retain signatures but fail explicitly instead of wrapping IDs. Reading IDs rejects
negative/noninteger values; filtered reads omit invalid rows consistently.

`storage::numeric` owns scalar types, conversion policies, comparisons, norms,
typed reductions and accurate/checked accumulation. These primitives depend on
neither array nor matrix ownership. Matrix adapters preserve MatrixError while
using the shared kernels. Array/matrix numerical policy and output types remain
unchanged by relocation.

Existing `matrix::numeric_type`, `advanced_matrix::numeric_type`,
`array::numeric_arithmetic` scalar exports and `array::numeric_reductions` paths
remain compatibility imports of the same implementation/types. Native compatibility
fixtures and the numerical Python/regression/release corpus cover the migration.
