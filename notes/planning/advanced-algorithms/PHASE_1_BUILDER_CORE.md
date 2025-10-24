## Phase 1 – Builder Core Extensions

**Timeline**: 4-6 weeks  
**Dependencies**: None (can run in parallel with temporal work)

### Objectives

Expand the builder step primitive catalog to support advanced algorithm composition without
requiring Rust implementation. Each primitive must be columnar, composable, and validated
for performance (targeting O(1) amortized overhead).

### Step Primitive Categories

#### 1.1 Arithmetic & Attribute Operations
Enable basic math on node/edge attributes:

- [ ] `core.add(var1, var2, output)` – Element-wise addition
- [ ] `core.sub(var1, var2, output)` – Element-wise subtraction
- [ ] `core.mul(var1, var2, output)` – Element-wise multiplication
- [ ] `core.div(var1, var2, output)` – Element-wise division with zero-check
- [ ] `core.load_attr(attr_name, output)` – Load node attribute into variable
- [ ] `core.load_edge_attr(attr_name, output)` – Load edge attribute into variable

**Implementation Notes:**
- Operate on columnar `Vec<AttrValue>` for bulk efficiency
- Type coercion rules (int/float promotion, string concatenation)
- Validate operand shapes match (node count)
- Error on type mismatches with clear messages

#### 1.2 Degree & Structural Primitives
Common graph structure computations:

- [ ] `core.weighted_degree(weight_attr, output)` – Degree with edge weights
- [ ] `core.k_core_mark(k, output)` – Mark nodes in k-core
- [ ] `core.triangle_count(output)` – Count triangles per node
- [ ] `core.edge_weight_sum(source_attr, target_attr, output)` – Sum edge weights incident to nodes
- [ ] `core.edge_weight_scale(attr, factor, output)` – Scale edge weights

**Implementation Notes:**
- Leverage existing neighbor bulk operations
- Cache degree computations when possible
- Triangle counting uses edge-iterator + neighbor intersection
- k-core implemented via iterative pruning with convergence check

#### 1.3 Normalization & Scaling
Feature engineering for downstream ML:

- [ ] `core.normalize_sum(var, output)` – Normalize to sum = 1.0
- [ ] `core.normalize_max(var, output)` – Normalize to max = 1.0
- [ ] `core.normalize_minmax(var, output)` – Scale to [0, 1] range
- [ ] `core.standardize(var, output)` – Z-score normalization (mean=0, std=1)
- [ ] `core.clip(var, min, max, output)` – Clamp values to range

**Implementation Notes:**
- Single-pass computation where possible (e.g., find min/max)
- Handle edge cases (all zeros, single value)
- Preserve NaN/infinity semantics clearly
- Document numerical stability considerations

#### 1.4 Temporal Selectors (from Phase 0)
Time-aware operations:

- [ ] `core.snapshot_at(commit|timestamp, output)` – Create temporal snapshot
- [ ] `core.temporal_window(start, end, output)` – Filter to window
- [ ] `core.decay(attr, half_life, output)` – Time-based decay function

**Implementation Notes:**
- Integrate with TemporalSnapshot and TemporalIndex
- Validate commit/timestamp existence
- Decay uses exponential model by default, configurable

#### 1.5 Ordering & Filtering
Common data manipulation patterns:

- [ ] `core.sort_nodes_by_attr(attr, order, output)` – Sort node list by attribute
- [ ] `core.filter_edges_by_attr(attr, predicate, output)` – Filter edge set
- [ ] `core.filter_nodes_by_attr(attr, predicate, output)` – Filter node set
- [ ] `core.top_k(var, k, output)` – Select top-k by value

**Implementation Notes:**
- Predicates: `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `contains`, `matches` (regex)
- Return node/edge ID lists suitable for subgraph creation
- Sorting uses stable sort, document complexity (O(n log n))
- Top-k uses heap-based selection for efficiency

#### 1.6 Sampling
Randomized selection for large graphs:

- [ ] `core.sample_nodes(fraction|count, seed, output)` – Random node sample
- [ ] `core.sample_edges(fraction|count, seed, output)` – Random edge sample
- [ ] `core.reservoir_sample(stream_var, k, seed, output)` – Reservoir sampling

**Implementation Notes:**
- Use reproducible RNG with seed parameter
- Fraction in [0.0, 1.0], count as integer
- Reservoir sampling for streaming contexts
- Document sampling guarantees (uniform, with/without replacement)

#### 1.7 Aggregations & Reductions
Statistical summaries:

- [ ] `core.mean(var, output)` – Arithmetic mean
- [ ] `core.std(var, output)` – Standard deviation
- [ ] `core.median(var, output)` – Median value
- [ ] `core.mode(var, output)` – Most common value
- [ ] `core.quantile(var, q, output)` – q-th quantile
- [ ] `core.entropy(var, output)` – Shannon entropy
- [ ] `core.histogram(var, bins, output)` – Binned counts

**Implementation Notes:**
- Median/quantile use selection algorithm (O(n))
- Mode uses hash table counting
- Entropy computed over discrete distribution (bins or unique values)
- Histogram bins: equal-width by default, configurable

#### 1.8 Path Utilities
Pathfinding helpers:

- [ ] `core.shortest_path_map(source, output)` – SSSP to all nodes
- [ ] `core.k_shortest_paths(source, target, k, output)` – K-shortest paths
- [ ] `core.random_walk(start_nodes, length, output)` – Random walk sequences

**Implementation Notes:**
- shortest_path_map uses Dijkstra/BFS depending on weights
- k-shortest uses Yen's algorithm
- random_walk supports restart probability, weighted transitions

#### 1.9 Community Helpers
Building blocks for custom community detection:

- [ ] `core.community_seed(strategy, output)` – Initialize communities
- [ ] `core.modularity_gain(partition, output)` – Compute modularity change
- [ ] `core.label_propagate_step(labels, output)` – Single LPA iteration

**Implementation Notes:**
- Seeding strategies: singleton, degree-based, random
- Modularity computation uses sparse edge weights
- LPA step uses mode aggregation over neighbors

#### 1.10 Flow & Capacity
Network flow primitives:

- [ ] `core.flow_update(flow, residual, output)` – Update flow along path
- [ ] `core.residual_capacity(capacity, flow, output)` – Compute residual graph

**Implementation Notes:**
- Foundation for max-flow / min-cut algorithms
- Validate flow conservation constraints
- Support both directed and undirected edges

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

