# Groggy Rust: graph/

Main graph logic, collection management, algorithms, and graph-level APIs.

- **fast_graph.rs**: FastGraph struct, main entry point
- **node_collection.rs / edge_collection.rs**: Node/edge storage and ops
- **attribute_manager.rs**: SIMD-optimized attribute access
- **algorithms/**: Core graph algorithms

## Example
```rust
let mut g = FastGraph::new();
g.add_node("n1");
g.add_edge("n1", "n2");
```
