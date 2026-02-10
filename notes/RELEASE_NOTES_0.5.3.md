# Release Notes 0.5.3 (Draft)

## New Features

- Sampler pipelines via the Builder DSL (`build_sampler`) that return `SubgraphArray` outputs.
- New sampling steps: iterate nodes/edges, expand neighborhoods, emit subgraphs, and map sub-pipelines over subgraph arrays.
- SubgraphArray mapping syntax: `nbh.map(...)` and builder-level `each(...)` helper.
- New sampler execution path in FFI to run compiled sampler pipelines from Python.

## Docs

- Added sampler pipeline examples to the Builder DSL guide.
- Documented sampler-generated SubgraphArrays in the SubgraphArray guide.

## Tests

- Added builder sampler tests for unified sampling and per-item mapped sampling.
