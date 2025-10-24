## Phase 5 – Builder / Pipeline Meta Infrastructure

**Timeline**: 3-4 weeks  
**Dependencies**: Phases 1-4 (algorithms using infrastructure)

### Objectives

Elevate the builder/pipeline system with higher-level tooling: DSL expression language,
manifest import/export, schema validation, and introspection APIs. Make algorithm composition
more discoverable and debuggable.

### 6.1 Builder DSL Expression Language

**Goal**: Higher-level macros and patterns for common multi-step compositions.

#### Features

- [ ] **Macro system** for common patterns
  - `with_normalized_degrees()` → init + normalize + attach steps
  - `with_temporal_diff(ref)` → snapshot + diff steps
  - `with_feature_scaling(attrs)` → multi-attribute scaling

- [ ] **Expression compiler** for inline computations
  - Parse expressions like `"rank * 0.85 + 0.15"` into step sequences
  - Type inference from context
  - Optimize trivial expressions (constant folding)

- [ ] **Control flow** (conditional, loops)
  - `builder.if_condition(predicate, then_steps, else_steps)`
  - `builder.iterate(max_iterations, convergence_check, body_steps)`

**Example**:
```python
builder = AlgorithmBuilder("custom_workflow")

# Macro usage
sg = builder.input("subgraph")
sg = builder.with_normalized_degrees(sg)

# Expression usage
ranks = builder.expr("pagerank * 0.5 + degree_centrality * 0.5")

# Conditional
builder.if_condition(
    lambda: ctx.graph_size > 10000,
    then_steps=[use_approximate_algorithm],
    else_steps=[use_exact_algorithm]
)
```

### 6.2 Pipeline Manifest Export/Import

**Goal**: Serialize pipelines for sharing, versioning, and cross-language interop.

#### Features

- [ ] **JSON export** (`pipeline.to_json()`)
  - Include algorithm IDs, parameters, step order
  - Metadata: creator, timestamp, description, version
  - Schema version for compatibility

- [ ] **TOML export** (`pipeline.to_toml()`)
  - Human-readable format for configuration files
  - Comments preserved for documentation

- [ ] **Import with validation** (`Pipeline.from_json()`, `.from_toml()`)
  - Validate schema version compatibility
  - Check algorithm availability (warn on missing)
  - Parameter type validation before instantiation

- [ ] **Manifest diffing** (`diff_manifests(m1, m2)`)
  - Show step additions, removals, parameter changes
  - Use for pipeline versioning and debugging

**Example**:
```python
# Export
pipeline = build_my_pipeline()
manifest = pipeline.to_json()
with open("my_pipeline.json", "w") as f:
    f.write(manifest)

# Import
with open("my_pipeline.json") as f:
    manifest = f.read()
pipeline = Pipeline.from_json(manifest)

# Diff
diff = diff_manifests(old_manifest, new_manifest)
print(diff.summary())  # "Changed parameter 'max_iter' from 20 to 50 in step 3"
```

### 6.3 Pipeline Registry Inspection CLI

**Goal**: Command-line tools for exploring algorithm catalog and debugging pipelines.

#### Commands

- [ ] `groggy algorithms list [--category=community]`
  - List available algorithms with descriptions

- [ ] `groggy algorithms info <algorithm_id>`
  - Show full metadata: params, cost hint, examples

- [ ] `groggy algorithms search <query>`
  - Search by name, description, category

- [ ] `groggy pipeline validate <manifest.json>`
  - Validate pipeline spec without running

- [ ] `groggy pipeline explain <manifest.json>`
  - Show execution plan, estimated cost, step dependencies

- [ ] `groggy pipeline run <manifest.json> <input_graph>`
  - Execute pipeline from CLI

**Example**:
```bash
$ groggy algorithms list --category=centrality
Available algorithms in 'centrality':
  - centrality.pagerank: PageRank centrality measure
  - centrality.betweenness: Betweenness centrality
  - centrality.eigenvector: Eigenvector centrality
  ...

$ groggy pipeline validate my_pipeline.json
✓ Pipeline valid
  Steps: 5
  Estimated cost: O(n²)
  Warnings: Step 3 (floyd_warshall) may be slow for large graphs
```

### 6.4 Parameter Schema Validation

**Goal**: Rich schema system for algorithm parameters with runtime validation.

#### Features

- [ ] **Schema definition language** (Rust-side)
  ```rust
  pub fn parameter_schema() -> ParameterSchema {
      ParameterSchema::new()
          .param("max_iter", ParamType::Int)
              .default(20)
              .range(1, 1000)
              .description("Maximum iterations")
          .param("tolerance", ParamType::Float)
              .default(0.001)
              .range(0.0, 1.0)
              .description("Convergence threshold")
          .param("seed", ParamType::Int)
              .optional()
              .description("Random seed for reproducibility")
  }
  ```

- [ ] **Python-side validation** before FFI call
  - Type checking
  - Range validation
  - Required vs optional parameters
  - Clear error messages pointing to invalid params

- [ ] **Schema introspection** (`algorithm.schema()`)
  - Query parameter requirements programmatically
  - Generate documentation automatically
  - Power IDE autocomplete / validation

**Example**:
```python
# Schema query
schema = algorithms.pagerank.schema()
print(schema.params["damping"])
# ParamInfo(name="damping", type="float", default=0.85, range=(0.0, 1.0), ...)

# Validation error
try:
    sg.apply(pagerank(damping=1.5))  # Invalid: > 1.0
except ValueError as e:
    print(e)  # "Parameter 'damping' out of range: 1.5 not in [0.0, 1.0]"
```

### 6.5 Subgraph Marshaller Enhancements

**Goal**: Optimize FFI marshalling for partial materialization and lazy evaluation.

#### Features

- [ ] **Partial materialization**
  - Only serialize node/edge IDs and changed attributes
  - Avoid full graph copy when pipeline modifies few nodes

- [ ] **Lazy attribute loading**
  - Load attributes on-demand during algorithm execution
  - Cache frequently accessed attributes

- [ ] **Attribute diff compression**
  - For temporal pipelines, send only deltas
  - Use bitmaps for existence changes

- [ ] **Zero-copy views** (where possible)
  - Share read-only subgraph data across FFI boundary
  - Use Arc/shared pointers for immutable data

**Example** (internal optimization, transparent to users):
```rust
// Instead of full subgraph copy:
let full_copy = subgraph.clone();  // Expensive!

// Use partial materialization:
let changes = subgraph.changed_since(last_checkpoint);
ffi_marshal_changes(changes);  // Only send deltas
```

### 6.6 Serde Bridge for Custom AttrValue Types

**Goal**: Support user-defined attribute types with custom serialization.

#### Features

- [ ] **AttrValue extension mechanism**
  - Register custom types (e.g., `AttrValue::Custom(Box<dyn CustomAttr>)`)
  - Provide serialization/deserialization hooks

- [ ] **Common custom types** (built-in)
  - `AttrValue::DateTime` (timestamps)
  - `AttrValue::Uuid` (identifiers)
  - `AttrValue::Json` (nested structures)

- [ ] **Python interop** for custom types
  - Automatic conversion from Python objects
  - Preserve type fidelity across FFI

**Example**:
```python
# Custom attribute type
from datetime import datetime

g.nodes[42].attr["created_at"] = datetime(2024, 1, 15)  # Stored as AttrValue::DateTime
result = sg.apply(temporal_filter(created_after="2024-01-01"))  # Type preserved
```

### 6.7 Integration Tests

**Goal**: Comprehensive tests covering builder-generated pipelines across all categories.

#### Test Coverage

- [ ] **Multi-step pipelines** (5+ steps)
  - Community detection → centrality → filtering
  - Temporal snapshot → diff → window aggregate
  - Decomposition → transform → feature engineering

- [ ] **Cross-category composition**
  - Pathfinding feeding into centrality
  - Community detection feeding into spectral analysis

- [ ] **Error handling**
  - Invalid parameters caught before execution
  - Graceful failures with rollback

- [ ] **Manifest roundtrip**
  - Export → import → execute produces same result

- [ ] **Performance regression**
  - Track execution time for standard pipelines
  - Alert on >10% slowdown

### Success Metrics

- 100% algorithm coverage with parameter schemas
- Manifest export/import roundtrip success rate: 100%
- Pipeline validation catches 95%+ of errors before execution
- CLI tools discoverable and documented
- <5% FFI overhead from marshaller enhancements

---

