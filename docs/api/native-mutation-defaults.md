# Native mutation defaults

Rust `Graph::new()`, `Default`, directional constructors and `with_config` now
create a **publishing** graph, matching ordinary Python `Graph()`. Node insertion
is fallible: `add_node() -> GraphResult<NodeId>` and
`add_nodes(count) -> GraphResult<Vec<NodeId>>`. Custom-configuration constructors
also return `GraphResult<Graph>` and validate configuration before construction.
Propagate errors with `?`.

```rust
use groggy::{Graph, GraphResult};

fn example() -> GraphResult<()> {
    let mut graph = Graph::new();
    let nodes = graph.add_nodes(2)?; // one version, not two scalar versions
    let first = graph.current_state_id();
    graph.add_edge(nodes[0], nodes[1])?;
    assert_eq!(graph.current_state_id(), first + 1);
    assert!(!graph.has_uncommitted_changes());
    graph.commit("named checkpoint".into(), "example".into())?;
    Ok(())
}
```

This deliberately changes the old Rust node-insertion signatures and constructor
policy, as authorized in A03. It does not discard errors, return sentinel IDs or
convert publication failures to panics. The raw node allocator is private to the
prepared storage executor. Bulk remains one operation/version. Zero nodes is a
no-op; impossible ID/vector extents reject before allocation in both policies.

## Explicit manual construction

Use `Graph::new_manual()`, `new_directed_manual()`, `new_undirected_manual()`,
`new_with_type_manual(type)` or `with_config_manual(config)` for deferred assembly.
Their `add_node/add_nodes` methods still return `GraphResult`; ordinary mutations
remain pending until the explicit manual acceptance workflow. Policy changes
require a clean graph. `try_add_node/try_add_nodes` and `writer()` always request
checked publication and reject dirty manual state.

```rust
use groggy::{Graph, GraphResult};

fn assemble() -> GraphResult<Graph> {
    let mut graph = Graph::new_manual();
    graph.add_nodes(3)?;
    assert_eq!(graph.current_state_id(), 0);
    assert!(graph.has_uncommitted_changes());
    graph.commit("accept assembled graph".into(), "example".into())?;
    graph.set_automatic_publication(true)?;
    Ok(graph)
}
```

Existing private fixture, visualization and legacy selection/import assembly now
spells out manual construction. The tests that exercise deliberate pending state
switch policy explicitly; infallible insertion no longer creates a hidden dirty
publishing graph. `new_publishing()` remains a compatibility alias for `new()`.
An immutable selection's documented manual materialization contract is unchanged.

## Python node accessor creation

The previously unregistered append/extend helpers are now public thin adapters.
They decode all rows before borrowing or mutating the graph, then call the native
bulk constructor once. Unsupported values, invalid rows/keys, reserved properties
and invalid semantic labels reject without allocating any node from the input.

```python
from groggy import Graph

graph = Graph()
assert graph.nodes.append({"name": "Ada", "score": 1.25}) == 0
assert graph.current_state_id == 1
assert graph.nodes.extend([{"name": "Grace"}, {"name": "Katherine"}]) == [1, 2]
assert graph.current_state_id == 2
assert graph.nodes.extend([]) == []
assert not graph.has_uncommitted_changes()
```

These are explicit owner mutations; existing immutable graph/table selections do
not expand or change their captured values. `Graph(manual=True)` retains deferred
acceptance. Requested managed external IDs use the existing explicit identity APIs.
Raw mutable storage and remaining uncertified/topology multi-stage execution are
separate remaining native-boundary scope, not implied to be complete here.
