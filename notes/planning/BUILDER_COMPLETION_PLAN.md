# Pipeline Builder Completion Plan

**Goal**: Enable the builder to construct PageRank and LPA algorithms as shown in the roadmap example  
**Status**: Planning Phase  
**Timeline**: 2-3 weeks

---

## Current State Analysis

### What Works Today ✅
- Basic builder DSL with 4 methods: `init_nodes`, `node_degrees`, `normalize`, `attach_as`
- Step registry with 48+ primitives in Rust
- Schema system, validation framework, and composition helpers (just completed)
- Pipeline execution via `builder.step_pipeline`
- Auto-variable generation (`_new_var`)

### Critical Gaps ❌

**1. DSL Primitives Missing**
- No `builder.input()` - can't reference input subgraph
- No `builder.var()` - can't explicitly create/reassign variables
- No `builder.map_nodes()` - can't express neighbor aggregations
- No `builder.core.*` namespace - can't access arithmetic/aggregation primitives
- No `builder.load_attr()` - can't load existing attributes

**2. Control Flow Missing**
- No loop constructs - can't iterate 20 times for PageRank
- No convergence checking - can't implement "iterate until stable"
- No conditional execution - can't express "if changed then..."

**3. Integration Gaps**
- Validation not connected to Python builder
- No type hints or IDE support
- No fluent chaining (can't do `builder.init().normalize().attach()`)
- Error messages are generic, not step-specific

**4. Testing Gaps**
- No Python builder tests
- No PageRank/LPA examples
- No roundtrip validation (Python → Rust → verify)

---

## Implementation Plan

### Phase 1: Core DSL Expansion (Week 1)

**Goal**: Add primitives needed for basic algorithms

#### Task 1.1: Add Input/Variable Management
**File**: `python-groggy/python/groggy/builder.py`

```python
class AlgorithmBuilder:
    def input(self, name: str = "subgraph") -> SubgraphHandle:
        """Reference to input subgraph."""
        # Returns handle that can be passed to operations
        
    def var(self, name: str, value: VarHandle) -> VarHandle:
        """Create or reassign a variable."""
        # Enables: ranks = builder.var("ranks", init_nodes(...))
        
    def auto_var(self, prefix: str = "var") -> str:
        """Generate unique variable name."""
        # Public version of _new_var for advanced users
```

**Rust Support**: No changes needed (input is implicit, var is Python-side tracking)

**Tests**:
- `test_builder_input_reference()`
- `test_builder_var_creation()`
- `test_builder_var_reassignment()`

---

#### Task 1.2: Add Attribute Operations
**File**: `python-groggy/python/groggy/builder.py`

```python
class AlgorithmBuilder:
    def load_attr(self, attr_name: str, default: Any = 0.0) -> VarHandle:
        """Load node attribute into variable."""
        # Maps to core.load_node_attr step
        
    def load_edge_attr(self, attr_name: str, default: Any = 0.0) -> VarHandle:
        """Load edge attribute into variable."""
        # Maps to core.load_edge_attr step
```

**Rust Support**: Steps already exist (`core.load_node_attr`, `core.load_edge_attr`)

**Tests**:
- `test_builder_load_attr()`
- `test_builder_load_edge_attr()`

---

#### Task 1.3: Add Core Namespace
**File**: `python-groggy/python/groggy/builder.py`

```python
class CoreOps:
    """Namespace for core step primitives."""
    def __init__(self, builder: 'AlgorithmBuilder'):
        self.builder = builder
    
    # Arithmetic
    def add(self, left: VarHandle, right: VarHandle) -> VarHandle:
        """Element-wise addition."""
    
    def sub(self, left: VarHandle, right: VarHandle) -> VarHandle:
        """Element-wise subtraction."""
    
    def mul(self, left: VarHandle, right: Union[VarHandle, float]) -> VarHandle:
        """Element-wise multiplication or scalar multiply."""
    
    def div(self, left: VarHandle, right: Union[VarHandle, float]) -> VarHandle:
        """Element-wise division or scalar divide."""
    
    # Aggregations
    def reduce(self, values: VarHandle, reducer: str = "sum") -> VarHandle:
        """Reduce to scalar (sum, mean, max, min)."""
    
    def normalize_sum(self, values: VarHandle) -> VarHandle:
        """Normalize so values sum to 1.0."""
    
    # Structural
    def node_degree(self, target: str) -> VarHandle:
        """Compute node degrees."""
    
    def weighted_degree(self, weight_attr: str) -> VarHandle:
        """Compute weighted node degrees."""

class AlgorithmBuilder:
    def __init__(self, name: str):
        # ...
        self.core = CoreOps(self)
```

**Rust Support**: All steps exist, just need Python wrappers

**Tests**:
- `test_core_arithmetic()`
- `test_core_aggregations()`
- `test_core_structural()`
- `test_core_scalar_multiply()`

---

#### Task 1.4: Add Map Operations
**File**: `python-groggy/python/groggy/builder.py`

```python
class AlgorithmBuilder:
    def map_nodes(self, 
                  fn: str,
                  inputs: Dict[str, VarHandle] = None,
                  **kwargs) -> VarHandle:
        """
        Map expression over nodes with access to neighbors.
        
        Args:
            fn: Expression string (e.g., "sum(ranks[neighbors(node)])")
            inputs: Variable context for the expression
            **kwargs: Additional parameters
            
        Example:
            neighbor_sums = builder.map_nodes(
                "sum(ranks[neighbors(node)])",
                inputs={"ranks": ranks}
            )
        """
        # Maps to core.map_nodes step with expression parsing
```

**Rust Support**: `core.map_nodes` step exists with expression support

**Tests**:
- `test_map_nodes_neighbor_sum()`
- `test_map_nodes_with_context()`
- `test_map_nodes_complex_expr()`

---

### Phase 2: Control Flow (Week 1-2)

**Goal**: Enable iterative algorithms

#### Task 2.1: Add Loop Construct
**File**: `python-groggy/python/groggy/builder.py`

```python
class LoopContext:
    """Context manager for loop body."""
    def __init__(self, builder: 'AlgorithmBuilder', iterations: int):
        self.builder = builder
        self.iterations = iterations
        self.start_step = len(builder.steps)
    
    def __enter__(self):
        return self
    
    def __exit__(self, *args):
        # Mark steps in loop body
        self.builder._finalize_loop(self.start_step, self.iterations)

class AlgorithmBuilder:
    def iterate(self, count: int) -> LoopContext:
        """
        Create a loop that repeats for `count` iterations.
        
        Example:
            with builder.iterate(20) as loop:
                # Steps here repeat 20 times
                neighbor_sums = builder.map_nodes(...)
                ranks = builder.var("ranks", ...)
        """
        return LoopContext(self, count)
```

**Rust Support**: Need loop handling in step interpreter

**Options**:
1. **Simple Unrolling**: Repeat steps N times in pipeline (easy, works for fixed iterations)
2. **Loop Step**: Add special `core.loop` step with body (more complex, better for large N)

**Recommendation**: Start with unrolling (simpler), add loop step later if needed

**Tests**:
- `test_loop_basic()`
- `test_loop_variable_persistence()`
- `test_loop_nested()` (future)

---

#### Task 2.2: Add Convergence Loop (Optional)
**File**: `python-groggy/python/groggy/builder.py`

```python
class AlgorithmBuilder:
    def while_converged(self, 
                       watch: VarHandle,
                       tolerance: float = 1e-6,
                       max_iterations: int = 100) -> LoopContext:
        """
        Loop until variable stabilizes or max iterations reached.
        
        Example:
            with builder.while_converged(ranks, tolerance=1e-6, max_iterations=100):
                # Update ranks
                ranks = builder.var("ranks", ...)
        """
```

**Rust Support**: Needs convergence checking in runtime

**Complexity**: High (need to track deltas, check convergence)

**Priority**: **Phase 3** (defer until basic loops work)

---

### Phase 3: Integration & Polish (Week 2)

**Goal**: Connect validation, improve ergonomics

#### Task 3.1: Integrate Validation
**File**: `python-groggy/python/groggy/builder.py`

```python
class AlgorithmBuilder:
    def build(self, validate: bool = True) -> 'BuiltAlgorithm':
        """
        Build the algorithm with optional validation.
        
        Args:
            validate: If True, validate pipeline before building
            
        Raises:
            ValidationError: If pipeline is invalid
        """
        if validate:
            report = self._validate()
            if not report.is_valid():
                raise ValidationError(report.format())
        
        return BuiltAlgorithm(self.name, self.steps)
    
    def _validate(self) -> ValidationReport:
        """Run Rust validation on pipeline."""
        # Call into _groggy.validate_pipeline()
```

**FFI Support**: Need to expose `validate_pipeline` to Python

**Tests**:
- `test_build_with_validation()`
- `test_validation_catches_errors()`
- `test_validation_warnings()`

---

#### Task 3.2: Add Fluent Chaining (Optional)
**File**: `python-groggy/python/groggy/builder.py`

```python
class VarHandle:
    def normalize(self, method: str = "sum") -> 'VarHandle':
        """Normalize this variable."""
        return self.builder.normalize(self, method)
    
    def attach(self, attr_name: str) -> 'VarHandle':
        """Attach as attribute and return self."""
        self.builder.attach_as(attr_name, self)
        return self

# Enables: builder.init_nodes().normalize().attach("result")
```

**Priority**: Low (nice-to-have, not required for PageRank/LPA)

---

#### Task 3.3: Add Type Hints & Stubs
**File**: `python-groggy/python/groggy/builder.pyi`

```python
from typing import Any, Dict, Union, Optional, overload

class VarHandle:
    name: str
    builder: AlgorithmBuilder
    def __init__(self, name: str, builder: AlgorithmBuilder) -> None: ...

class AlgorithmBuilder:
    name: str
    core: CoreOps
    
    def __init__(self, name: str) -> None: ...
    def var(self, name: str, value: VarHandle) -> VarHandle: ...
    def load_attr(self, attr_name: str, default: Any = 0.0) -> VarHandle: ...
    def map_nodes(self, fn: str, inputs: Optional[Dict[str, VarHandle]] = None) -> VarHandle: ...
    # ... etc
```

**Priority**: Medium (improves IDE experience)

---

### Phase 4: Testing & Examples (Week 2-3)

**Goal**: Comprehensive test coverage and working examples

#### Task 4.1: PageRank Builder Example
**File**: `tests/test_builder_pagerank.py`

```python
def test_builder_pagerank_basic():
    """Build PageRank using the builder DSL."""
    from groggy import Graph, Subgraph
    from groggy.builder import AlgorithmBuilder
    
    # Create test graph
    graph = Graph()
    a, b, c = graph.add_node(), graph.add_node(), graph.add_node()
    graph.add_edge(a, b)
    graph.add_edge(b, c)
    graph.add_edge(c, a)
    sg = Subgraph.from_graph(graph)
    
    # Build PageRank with builder
    builder = AlgorithmBuilder("custom_pagerank")
    
    # Initialize ranks
    ranks = builder.init_nodes(default=1.0)
    
    # Iterate 20 times
    with builder.iterate(20):
        # Sum neighbor ranks
        neighbor_sums = builder.map_nodes(
            "sum(ranks[neighbors(node)])",
            inputs={"ranks": ranks}
        )
        
        # Apply damping: 0.85 * neighbor_sums + 0.15
        damped = builder.core.mul(neighbor_sums, 0.85)
        ranks = builder.core.add(damped, 0.15)
        
        # Normalize
        ranks = builder.core.normalize_sum(ranks)
    
    # Attach result
    builder.attach_as("pagerank", ranks)
    
    # Execute
    algo = builder.build()
    result = sg.apply(algo)
    
    # Verify results match native PageRank
    from groggy.algorithms import pagerank
    expected = sg.apply(pagerank(iterations=20))
    
    for node in result.nodes():
        pr_builder = result.get_node_attr(node, "pagerank")
        pr_native = expected.get_node_attr(node, "pagerank")
        assert abs(pr_builder - pr_native) < 1e-6
```

**Tests**:
- `test_builder_pagerank_basic()`
- `test_builder_pagerank_matches_native()`
- `test_builder_pagerank_directed_vs_undirected()`

---

#### Task 4.2: LPA Builder Example
**File**: `tests/test_builder_lpa.py`

```python
def test_builder_lpa():
    """Build Label Propagation using the builder DSL."""
    builder = AlgorithmBuilder("custom_lpa")
    
    # Initialize labels to node IDs
    labels = builder.load_attr("_node_id", default=0)
    
    # Iterate until convergence (or 100 times max)
    with builder.iterate(100):
        # For each node, adopt most common neighbor label
        new_labels = builder.map_nodes(
            "mode(labels[neighbors(node)])",
            inputs={"labels": labels}
        )
        
        labels = builder.var("labels", new_labels)
    
    # Attach result
    builder.attach_as("community", labels)
    
    # Execute and verify
    algo = builder.build()
    result = sg.apply(algo)
    
    # Compare with native LPA
    from groggy.algorithms import lpa
    expected = sg.apply(lpa(iterations=100))
    
    # Communities should be similar (not exact due to randomness)
    assert_similar_communities(result, expected)
```

**Tests**:
- `test_builder_lpa_basic()`
- `test_builder_lpa_convergence()`
- `test_builder_lpa_matches_native_structure()`

---

#### Task 4.3: Builder Unit Tests
**File**: `tests/test_builder_core.py`

```python
def test_builder_var_tracking():
    """Verify variable tracking works correctly."""
    
def test_builder_step_encoding():
    """Verify steps encode to correct Rust specs."""
    
def test_builder_arithmetic_chain():
    """Verify chained arithmetic operations."""
    
def test_builder_map_nodes_expressions():
    """Test various map_nodes expression patterns."""
    
def test_builder_loop_unrolling():
    """Verify loop unrolling produces correct steps."""
```

**Coverage Target**: 90%+ of builder.py

---

### Phase 5: FFI Runtime Polish (Week 3)

**Goal**: Production-ready runtime

#### Task 5.1: Handle Lifecycle Management
**File**: `python-groggy/src/ffi/api/pipeline.rs`

```rust
// Currently: pipelines stored in static HashMap, never cleaned up
// Need: automatic cleanup when Python handle is dropped

struct PipelineHandle {
    id: Uuid,
    // Add reference counting or cleanup hook
}

impl Drop for PipelineHandle {
    fn drop(&mut self) {
        // Remove from global registry
        remove_pipeline(self.id);
    }
}
```

**Priority**: Medium (not blocking for examples)

---

#### Task 5.2: GIL Release
**File**: `python-groggy/src/ffi/api/pipeline.rs`

```rust
pub fn execute_pipeline(
    py: Python,
    pipeline_id: String,
    subgraph: &PySubgraph,
) -> PyResult<PySubgraph> {
    let sg = subgraph.inner.borrow().clone();
    
    // Release GIL for long-running execution
    let result = py.allow_threads(|| {
        execute_pipeline_inner(pipeline_id, sg)
    })?;
    
    Ok(PySubgraph::from(result))
}
```

**Priority**: High (enables parallelism)

---

#### Task 5.3: Rich Error Translation
**File**: `python-groggy/src/ffi/api/pipeline.rs`

```rust
// Current: anyhow::Error → generic PyRuntimeError
// Need: structured Python exceptions

impl From<ValidationError> for PyErr {
    fn from(err: ValidationError) -> Self {
        // Create PipelineValidationError with step context
        PyValueError::new_err(format!(
            "Step {} ({}): {}",
            err.step_index, err.step_id, err.message
        ))
    }
}
```

**Priority**: High (better debugging experience)

---

## Implementation Timeline

### Week 1: Core DSL + Basic Loops
- **Days 1-2**: Tasks 1.1-1.2 (input, var, load_attr)
- **Days 3-4**: Task 1.3 (core namespace)
- **Day 5**: Task 1.4 (map_nodes)
- **Weekend**: Task 2.1 (iterate loop)

**Milestone**: Can write basic PageRank skeleton

### Week 2: Integration + Examples  
- **Days 1-2**: Task 3.1 (validation integration)
- **Day 3**: Task 3.3 (type hints)
- **Days 4-5**: Task 4.1 (PageRank example)
- **Weekend**: Task 4.2 (LPA example)

**Milestone**: PageRank and LPA examples working

### Week 3: Testing + Polish
- **Days 1-2**: Task 4.3 (comprehensive tests)
- **Day 3**: Task 5.1 (handle lifecycle)
- **Days 4-5**: Tasks 5.2-5.3 (GIL release, error translation)

**Milestone**: Production-ready builder

---

## Success Criteria

### Must Have ✅
1. PageRank example builds and executes correctly
2. LPA example builds and executes correctly
3. Results match native algorithm implementations (within tolerance)
4. Validation catches common errors before execution
5. All tests passing (90%+ coverage)

### Should Have 🎯
1. Type hints for IDE support
2. GIL release for long pipelines
3. Rich error messages with step context
4. Handle lifecycle management

### Nice to Have 💡
1. Fluent chaining API
2. Convergence-based loops
3. Nested loops
4. Builder templates for common patterns

---

## Risk Mitigation

### Risk: Loop Unrolling Complexity
**Impact**: High  
**Mitigation**: Start with simple fixed iteration count, defer convergence loops

### Risk: Expression Parsing Complexity
**Impact**: Medium  
**Mitigation**: Leverage existing `core.map_nodes` expression support, add more functions as needed

### Risk: Python-Rust Type Mismatches
**Impact**: Medium  
**Mitigation**: Validation at build time, comprehensive tests

### Risk: Performance Overhead
**Impact**: Low  
**Mitigation**: Builder generates same Rust steps as hand-coded pipelines, should be equivalent

---

## Dependencies

### Completed (Phase 1) ✅
- Schema registry system
- Validation framework
- Composition helpers (Rust side)

### External Dependencies
- None (all Rust primitives exist)

### Blocking Issues
- None identified

---

## Next Steps

1. **Review & Approval**: Get stakeholder sign-off on plan
2. **Setup**: Create test infrastructure, example data
3. **Start Implementation**: Begin with Task 1.1 (input/var management)
4. **Iterate**: Weekly check-ins on progress
5. **Documentation**: Update roadmap as tasks complete

---

## Questions to Resolve

1. **Loop Implementation**: Unroll or add loop step primitive?
   - **Recommendation**: Unroll for simplicity
   
2. **Expression Functions**: What functions should `map_nodes` support?
   - **Recommendation**: Start with `sum()`, `mean()`, `mode()`, `neighbors()`
   
3. **Validation Integration**: Automatic or explicit?
   - **Recommendation**: Automatic by default, opt-out with `validate=False`
   
4. **Type Hints**: Generate or write manually?
   - **Recommendation**: Write manually for now, automate later

---

## Appendix: Example Usage Comparison

### Before (Current, Limited)
```python
builder = AlgorithmBuilder("simple")
nodes = builder.init_nodes(default=0.0)
degrees = builder.node_degrees(nodes)
normalized = builder.normalize(degrees)
builder.attach_as("result", normalized)
```

### After (Target, Full PageRank)
```python
builder = AlgorithmBuilder("pagerank")

ranks = builder.init_nodes(default=1.0)

with builder.iterate(20):
    neighbor_sums = builder.map_nodes(
        "sum(ranks[neighbors(node)])",
        inputs={"ranks": ranks}
    )
    damped = builder.core.mul(neighbor_sums, 0.85)
    ranks = builder.var("ranks", builder.core.add(damped, 0.15))
    ranks = builder.core.normalize_sum(ranks)

builder.attach_as("pagerank", ranks)
algo = builder.build()
```

**Difference**: From 4 simple operations to full iterative algorithm with neighbor aggregation!
