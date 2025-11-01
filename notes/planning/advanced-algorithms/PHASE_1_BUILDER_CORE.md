## Phase 1 – Builder Core Extensions

**Timeline**: 4-6 weeks  
**Dependencies**: None (can run in parallel with temporal work)
**Status**: ✅ Complete – 48+ primitives implemented across 15 modules (see `PHASE_1_IMPLEMENTATION_STATUS.md`)

### Objectives

Expand the builder step primitive catalog to support advanced algorithm composition without
requiring Rust implementation. Each primitive must be columnar, composable, and validated
for performance (targeting O(1) amortized overhead).

**Achievement**: Phase 1 delivered comprehensive step primitive coverage enabling complex graph computations through composable building blocks. All primitives follow the performance-first design established in the STYLE_ALGO refactoring.

### Relationship to STYLE_ALGO

While Phase 1 focused on **compositional step primitives** for the builder DSL, the concurrent performance refactoring established **STYLE_ALGO** for full algorithms. Key differences:

- **Step Primitives** (Phase 1): Small, reusable operations composed into custom algorithms
- **Full Algorithms** (STYLE_ALGO): Optimized end-to-end implementations with CSR caching

Both follow similar principles:
- Columnar operations on `Vec<AttrValue>`
- Pre-allocated buffers, no inner-loop allocations  
- Profiling instrumentation
- Deterministic ordering via `ordered_nodes()`/`ordered_edges()`

Where applicable, step primitives use CSR (e.g., `structural.rs` degree computations) following the same caching patterns as full algorithms.

### Module Organization

The step primitives are organized into focused modules in `src/algorithms/steps/`:

**Current Modules (Implemented):**
- **`core.rs`** – Core traits (`Step`, `StepScope`, `StepVariables`, `StepRegistry`)
- **`init.rs`** – Initialization primitives (2 steps)
- **`attributes.rs`** – Attribute loading and persistence (4 steps)
- **`arithmetic.rs`** – Binary arithmetic operations (4 operations)
- **`transformations.rs`** – Mapping and transformation (1 step)
- **`aggregations.rs`** – Reductions and statistics (7 steps)
- **`structural.rs`** – Graph structure computations (5 steps)
- **`normalization.rs`** – Value normalization (4 methods)
- **`temporal.rs`** – Temporal operations (8 steps)
- **`filtering.rs`** – Sorting, filtering, top-k operations (4 steps)
- **`sampling.rs`** – Random sampling operations (3 steps)
- **`pathfinding.rs`** – Path utilities (3 steps)
- **`community.rs`** – Community detection helpers (3 steps)
- **`flow.rs`** – Network flow primitives (2 steps)
- **`registry.rs`** – Step registration and discovery
- **`mod.rs`** – Module orchestration and tests

**Phase 1 Complete**: All 10 sections (1.1-1.10) implemented with 48+ step primitives across 15 focused modules.

### Step Primitive Categories

#### 1.1 Arithmetic & Attribute Operations
Enable basic math on node/edge attributes:

- ✅ `core.add(left, right, target)` – Element-wise addition → **`arithmetic.rs`**
- ✅ `core.sub(left, right, target)` – Element-wise subtraction → **`arithmetic.rs`**
- ✅ `core.mul(left, right, target)` – Element-wise multiplication → **`arithmetic.rs`**
- ✅ `core.div(left, right, target)` – Element-wise division with zero-check → **`arithmetic.rs`**
- ✅ `core.load_attr(attr_name, output)` – Load node attribute into variable → **`attributes.rs`**
- ✅ `core.load_node_attr(attr, target, default)` – Load node attribute → **`attributes.rs`**
- ✅ `core.load_edge_attr(attr, target, default)` – Load edge attribute → **`attributes.rs`**

**Implementation**: `BinaryArithmeticStep` in `arithmetic.rs`, attribute steps in `attributes.rs`

**Implementation Notes:**
- Operate on columnar `Vec<AttrValue>` for bulk efficiency
- Type coercion rules (int/float promotion, string concatenation)
- Validate operand shapes match (node count)
- Error on type mismatches with clear messages

#### 1.2 Degree & Structural Primitives
Common graph structure computations:

- ✅ `core.node_degree(target)` – Unweighted degree → **`structural.rs`**
- ✅ `core.weighted_degree(weight_attr, target)` – Degree with edge weights → **`structural.rs`**
- ✅ `core.k_core_mark(k, target)` – Mark nodes in k-core → **`structural.rs`**
- ✅ `core.triangle_count(target)` – Count triangles per node → **`structural.rs`**
- ✅ `core.edge_weight_sum(weight_attr, target)` – Sum edge weights → **`structural.rs`**
- ✅ `core.edge_weight_scale(attr, factor, target)` – Scale edge weights → **`attributes.rs`** (edge operations)

**Implemented**: All 6 steps complete in `structural.rs` (5 steps) and `attributes.rs` (1 step)
- ✅ `NodeDegreeStep`
- ✅ `WeightedDegreeStep`
- ✅ `KCoreMarkStep`
- ✅ `TriangleCountStep`
- ✅ `EdgeWeightSumStep`
- ✅ `EdgeWeightScaleStep` (in `attributes.rs`)

**Implementation Notes:**
- Leverage existing neighbor bulk operations
- Cache degree computations when possible
- Triangle counting uses edge-iterator + neighbor intersection
- k-core implemented via iterative pruning with convergence check

#### 1.3 Normalization & Scaling
Feature engineering for downstream ML:

- ✅ `core.normalize_node_values(source, target, method, epsilon)` – Normalize using sum/max/minmax (backward compat) → **`normalization.rs`**
- ✅ `core.normalize_values(source, target, method, epsilon)` – Generic normalize (nodes or edges) → **`normalization.rs`**
- ✅ `core.standardize(source, target, epsilon)` – Z-score normalization (mean=0, std=1) → **`normalization.rs`**
- ✅ `core.clip(source, min, max, target)` – Clamp values to range → **`normalization.rs`**

**Implemented**: All 4 operations complete in `normalization.rs` (now 397 lines, up from 148)
- ✅ `NormalizeValuesStep` (generic, works on node or edge maps)
- ✅ `NormalizeNodeValuesStep` (backward-compatible wrapper)
- ✅ `StandardizeStep` (generic, computes Z-scores)
- ✅ `ClipValuesStep` (generic, clamps to [min, max])

**Key Design**: All normalization operations are now **generic** and work on both node and edge maps automatically by detecting the variable type at runtime.

**Implementation Notes:**
- Single-pass computation where possible (e.g., find min/max)
- Handle edge cases (all zeros, single value)
- Preserve NaN/infinity semantics clearly
- Document numerical stability considerations

#### 1.4 Temporal Selectors (from Phase 0)
Time-aware operations:

- ✅ `temporal.diff_nodes(before, after, output_prefix)` – Node set differences → **`temporal.rs`** (uses real `TemporalSnapshot` delta computation)
- ✅ `temporal.diff_edges(before, after, output_prefix)` – Edge set differences → **`temporal.rs`** (emits per-edge add/remove maps)
- ✅ `temporal.window_aggregate(attr, function, output, index_var)` – Aggregate over time → **`temporal.rs`** (streams values from `TemporalIndex`, falls back to snapshots, resolves placeholder indices)
- ✅ `temporal.filter(predicate, output)` – Filter by temporal properties → **`temporal.rs`** (supports created/modified/existed predicates via lifetime metadata)
- ✅ `temporal.mark_changed_nodes(output, change_type)` – Mark changed nodes → **`temporal.rs`** (uses `Context::changed_entities` + change-type filter)
- ✅ `temporal.snapshot_at(commit|timestamp, output)` – Materialize snapshot handles → **`temporal.rs`** (stores real `TemporalSnapshot` in variables)
- ✅ `temporal.window(start, end, output)` – Mark nodes that existed within a window → **`temporal.rs`** (consults `TemporalIndex` lifetimes)
- ✅ `temporal.decay(attr, half_life, output)` – Time-based decay → **`temporal.rs`** (previously implemented)

**Status**: ✅ **Temporal step suite now fully implemented.** Highlights:
1. `StepValue` gained `Snapshot`/`TemporalIndex` variants so builders can cache real artifacts.
2. Diff steps automatically resolve `before`/`after` snapshots (from vars or temporal scope) and call `Context::delta`.
3. Aggregations read from `TemporalIndex` histories, resolve placeholder indices back into `AttrValue`s via `GraphPool`, and respect numeric-only reducers.
4. Filtering/mark-changed/window logic all leverage fresh helpers added to `TemporalIndex` (creation commit, lifetime ranges, range change scans).
5. Snapshot + window steps now operate on real history (`HistoryForest`), and `temporal.decay` remains available for time-based attenuation.

#### 1.5 Ordering & Filtering
Common data manipulation patterns:

- ✅ `core.sort_nodes_by_attr(attr, order, target)` – Sort node list by attribute → **`filtering.rs`**
- ✅ `core.filter_edges_by_attr(attr, predicate, target)` – Filter edge set → **`filtering.rs`**
- ✅ `core.filter_nodes_by_attr(attr, predicate, target)` – Filter node set → **`filtering.rs`**
- ✅ `core.top_k(source, k, target)` – Select top-k by value → **`filtering.rs`**

**Implemented**: All 4 filtering operations in `filtering.rs` (393 lines)

**Features:**
- `Predicate` enum: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `contains`, `range`
- Stable sorting with O(n log n) complexity
- Top-k returns ranked results
- Comparison works across int/float types
- String filtering with substring matching

**Test Coverage:** 4 predicate unit tests + 2 integration tests

#### 1.6 Sampling
Randomized selection for large graphs:

- ✅ `core.sample_nodes(fraction|count, seed, target)` – Random node sample → **`sampling.rs`**
- ✅ `core.sample_edges(fraction|count, seed, target)` – Random edge sample → **`sampling.rs`**
- ✅ `core.reservoir_sample(stream_var, k, seed, target)` – Reservoir sampling → **`sampling.rs`**

**Implemented**: All 3 sampling operations in `sampling.rs` (389 lines)

**Features:**
- `SampleSpec` enum: `Fraction { fraction }` or `Count { count }`
- Reproducible RNG with optional seed parameter
- Uniform random sampling using reservoir algorithm
- Works on subgraph nodes/edges or map variables
- O(n) complexity for reservoir sampling

**Test Coverage:** 5 unit tests (spec validation, reproducibility) + 4 integration tests

**Example Usage:**
```json
// Sample 10% of nodes with seed for reproducibility
{
  "id": "core.sample_nodes",
  "params": {
    "fraction": 0.1,
    "seed": 42,
    "target": "sampled_nodes"
  }
}

// Reservoir sample 100 elements from a map
{
  "id": "core.reservoir_sample",
  "params": {
    "source": "candidate_nodes",
    "k": 100,
    "seed": 123,
    "entity_type": "nodes",
    "target": "selected_nodes"
  }
}
```

#### 1.7 Aggregations & Reductions
Statistical summaries:

- ✅ `core.reduce_nodes(source, target, reducer)` – Aggregate with sum/min/max/mean → **`aggregations.rs`**
- ✅ `core.std(source, target)` – Standard deviation → **`aggregations.rs`**
- ✅ `core.median(source, target)` – Median value → **`aggregations.rs`**
- ✅ `core.mode(source, target)` – Most common value → **`aggregations.rs`**
- ✅ `core.quantile(source, q, target)` – q-th quantile → **`aggregations.rs`**
- ✅ `core.entropy(source, target)` – Shannon entropy → **`aggregations.rs`**
- ✅ `core.histogram(source, bins, target)` – Binned counts → **`aggregations.rs`**

**Implemented**: All 7 aggregation operations complete in `aggregations.rs` (635 lines)

**Features:**
- `ReduceNodeValuesStep` with `Reduction` enum (sum, min, max, mean)
- `StdDevStep`: Sample standard deviation (n-1 denominator)
- `MedianStep`: Handles both odd and even length sequences
- `ModeStep`: Finds most common value, preserves original type
- `QuantileStep`: Linear interpolation for fractional quantiles
- `EntropyStep`: Shannon entropy in bits (log2 base)
- `HistogramStep`: Equal-width binning, returns node map with bin counts

**Test Coverage:** 6 unit tests covering std dev, median, quantile, entropy, and histogram

**Implementation Notes:**
- Median/quantile use selection algorithm (O(n))
- Mode uses hash table counting
- Entropy computed over discrete distribution (bins or unique values)
- Histogram bins: equal-width by default, configurable

#### 1.8 Path Utilities
Pathfinding helpers:

- ✅ `core.shortest_path_map(source, output)` – SSSP to all nodes → **`pathfinding.rs`**
- ✅ `core.k_shortest_paths(source, target, k, output)` – K-shortest paths → **`pathfinding.rs`**
- ✅ `core.random_walk(start_nodes, length, output)` – Random walk sequences → **`pathfinding.rs`**

**Implemented**: All 3 pathfinding operations complete in `pathfinding.rs` (695 lines)

**Features:**
- `ShortestPathMapStep`: Uses BFS for unweighted graphs, Dijkstra for weighted graphs
- `KShortestPathsStep`: Yen's algorithm for finding k-shortest paths between two nodes
- `RandomWalkStep`: Supports restart probability, weighted edge transitions, and reproducible RNG with seed
- All steps support optional edge weight attributes
- Paths and walks serialized as JSON for downstream processing

**Implementation Notes:**
- Leverages existing `bfs_layers` and `dijkstra` utilities from `src/algorithms/pathfinding/utils.rs`
- K-shortest paths uses Yen's algorithm with edge exclusion for path diversity
- Random walks support teleport/restart with configurable probability (0.0-1.0)
- Weighted random walks use rejection sampling for neighbor selection
- All operations emit iteration metrics for monitoring

#### 1.9 Community Helpers
Building blocks for custom community detection:

- ✅ `core.community_seed(strategy, target)` – Initialize communities → **`community.rs`**
- ✅ `core.modularity_gain(partition, target)` – Compute modularity change → **`community.rs`**
- ✅ `core.label_propagate_step(labels, target)` – Single LPA iteration → **`community.rs`**

**Implemented**: All 3 community helper steps complete in `community.rs` (569 lines)
- ✅ `CommunitySeedStep` with `SeedStrategy` enum (singleton, degree_based, random)
- ✅ `ModularityGainStep` for computing modularity changes
- ✅ `LabelPropagateStep` for single-iteration label propagation
- All steps registered in the step registry

**Test Coverage:** 5 unit tests covering all three steps with various scenarios

**Implementation Notes:**
- Seeding strategies: singleton (each node in own community), degree-based (high-degree hubs get unique labels), random (k communities with optional seed)
- Modularity computation uses sparse edge weights and computes gain for potential moves
- LPA step uses mode aggregation over neighbors with deterministic tie-breaking
- Uses fastrand for reproducible random seeding

#### 1.10 Flow & Capacity
Network flow primitives:

- ✅ `core.flow_update(flow, delta, target)` – Update flow along path → **`flow.rs`**
- ✅ `core.residual_capacity(capacity, flow, target)` – Compute residual graph → **`flow.rs`**

**Implemented**: All 2 flow operations complete in `flow.rs` (301 lines)

**Features:**
- `FlowUpdateStep`: Updates flow values by adding deltas (supports both edge map and scalar deltas)
- `ResidualCapacityStep`: Computes residual capacity (capacity - flow), filtering saturated edges
- Type-safe arithmetic with int/float promotion
- Automatic handling of forward and backward residual edges
- Foundation for max-flow/min-cut algorithms (Ford-Fulkerson, Edmonds-Karp)

**Test Coverage:** 5 unit tests covering arithmetic operations, type promotion, and edge cases

**Implementation Notes:**
- Foundation for max-flow / min-cut algorithms
- Flow conservation validation is responsibility of algorithm builder
- Support both directed and undirected edges via residual capacity filtering
- Saturated edges (residual = 0) automatically excluded from result

### Infrastructure Components

Beyond individual steps, Phase 1 includes infrastructure for step management:

#### Pipeline Builder Enhancements

- [ ] **Step schema registry**: Each step declares inputs, outputs, parameters, types
- [ ] **Validation framework**: Type checking, data-flow analysis, cost estimation
- [ ] **Error reporting**: Structured errors pointing to problematic steps
- [ ] **Step composition helpers**: Macros for common multi-step patterns

#### FFI Runtime

- [ ] **Spec serialization**: JSON/TOML roundtrip for pipeline specs
- [ ] **Handle lifecycle**: Automatic cleanup, reference counting for shared state
- [ ] **Error translation**: Rust error types mapped to Python exceptions
- [ ] **GIL management**: Release during expensive operations, reacquire for callbacks

#### Python Builder DSL

- [ ] **Method chaining**: Fluent API for step composition
- [ ] **Variable scoping**: Auto-generated variable names, explicit output control
- [ ] **Type hints**: Full stub coverage for IDE support
- [ ] **Documentation**: Docstrings with examples for each step primitive

#### Validation & Testing

- [ ] **Unit tests**: Each step primitive with edge cases
- [ ] **Integration tests**: Multi-step pipelines exercising composition
- [ ] **Benchmark suite**: Performance regression tracking (`benches/steps/`)
- [ ] **Roundtrip tests**: Python spec → Rust execution → Python result

### Success Metrics

- 100% step primitive coverage with tests
- <1ms per-step overhead for simple operations (add, load_attr)
- <10ms for complex operations (k_core, triangle_count) on 10K nodes
- Zero FFI marshaling overhead for in-place operations
- Clear error messages (90%+ user comprehension without docs)

### Example: Custom PageRank with Builder

```python
from groggy.builder import AlgorithmBuilder

builder = AlgorithmBuilder("custom_pagerank")

# Input subgraph
sg = builder.input("subgraph")

# Initialize ranks
ranks = builder.var("ranks", builder.init_nodes(sg, fn="constant(1.0)"))

# Iteration loop
for _ in range(20):
    # Sum neighbor ranks
    neighbor_sums = builder.map_nodes(
        sg,
        fn="sum(ranks[neighbors(node)])",
        inputs={"ranks": ranks},
    )
    
    # Apply damping
    damped = builder.var("damped", builder.core.mul(neighbor_sums, 0.85))
    teleport = builder.var("teleport", builder.core.add(damped, 0.15))
    
    # Normalize
    ranks = builder.var("ranks", builder.core.normalize_sum(teleport))

# Attach as attribute
builder.attach_node_attr(sg, "pagerank", ranks)

# Compile to algorithm
pagerank_algo = builder.compile()
```

### Dependencies

**Rust Core:**
- Existing `src/algorithms/steps/` foundation
- Columnar neighbor/attribute operations in `GraphSpace`
- Algorithm trait and pipeline infrastructure

**FFI:**
- Parameter marshaling for complex types (predicates, functions)
- Error translation with source location tracking
- Step registry exposure for discovery

**Python:**
- Builder API foundation from Phase 5 (v0.5.0)
- Type stub generation
- Documentation generation from Rust metadata

### Risks & Mitigations

**Risk**: Step primitive explosion (too many variants)  
**Mitigation**: Keep core set minimal; add parameterization (e.g., one `aggregate` step with function parameter vs. separate `sum`, `mean`, `max`)

**Risk**: Performance regression from indirection  
**Mitigation**: Benchmark each primitive; inline hot paths; consider JIT compilation for step sequences

**Risk**: Type system complexity (heterogeneous attributes)  
**Mitigation**: Explicit coercion rules; runtime type checks with clear errors; document type compatibility matrix

**Risk**: User confusion about when to use steps vs pre-built algorithms  
**Mitigation**: Clear guidance in docs; builder examples for common patterns; pre-built algorithms preferred for standard use cases
