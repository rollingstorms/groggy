# Builder DSL API Reference

The builder module is exposed at `groggy.builder` and provides the building blocks for composing
custom step pipelines.

## `builder(name: str) -> AlgorithmBuilder`

Factory function that returns a new `AlgorithmBuilder` configured with the given name. The resulting
algorithm is registered under `custom.{name}` when `build()` is called.

## `AlgorithmBuilder`

Methods:

- `init_nodes(default=0.0) -> VarHandle`
- `node_degrees(nodes: VarHandle) -> VarHandle`
- `normalize(values: VarHandle, method="sum") -> VarHandle`
- `attach_as(attr_name: str, values: VarHandle) -> None`
- `build() -> BuiltAlgorithm`

Attributes:

- `steps` — list of step dictionaries
- `variables` — mapping of variable names to `VarHandle`

## `VarHandle`

Lightweight reference returned from builder methods. Useful for debugging and for passing into
subsequent builder calls.

## `BuiltAlgorithm`

Implements the `AlgorithmHandle` protocol. The object can be passed to:

- `Subgraph.apply(custom_algo)`
- `groggy.pipeline.apply(subgraph, custom_algo)`
- `Pipeline([... custom_algo ...])`

Under the hood, `to_spec()` serialises the recorded steps to the `builder.step_pipeline` algorithm
implemented in Rust.

### Related Documentation

- [Builder User Guide](../guide/builder.md)
- [Pipeline API](pipeline.md)
- [Architecture Appendix](../appendices/graph-maintenance-rollup.md)
