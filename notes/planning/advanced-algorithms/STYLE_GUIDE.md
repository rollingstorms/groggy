## 🎨 Implementation Style Guide

### Builder Core Extensions

**Naming Conventions:**
- Keep primitive names under `core.*` namespace
- Prefer verbs over nouns (`core.filter_edges`, not `core.edge_filter`)
- Use snake_case for step names
- Be explicit: `normalize_sum` over `normalize`

**Step Design:**
- Every step must declare: inputs, outputs, parameter schema, validation rules
- Steps should be composable and stateless
- Return data via variables, not by mutating shared state
- Provide parallel Python helper where it aids readability (e.g., `builder.filter_edges(...)`)
- Document each primitive in builder docs with accepted methods/options

**Example**:
```rust
pub struct NormalizeSumStep {
    input_var: String,
    output_var: String,
}

impl Step for NormalizeSumStep {
    fn id(&self) -> &'static str { "core.normalize_sum" }
    
    fn schema() -> StepSchema {
        StepSchema::new()
            .input("input_var", VarType::NodeData)
            .output("output_var", VarType::NodeData)
            .description("Normalize values to sum = 1.0")
    }
    
    fn execute(&self, ctx: &mut Context, sg: &Subgraph) -> GraphResult<StepOutput> {
        let values = ctx.get_var(&self.input_var)?;
        let sum: f64 = values.iter().sum();
        let normalized: Vec<f64> = values.iter().map(|v| v / sum).collect();
        Ok(StepOutput::NodeData(normalized))
    }
}
```

### Rust Algorithm Implementations

**Module Organization:**
- Place algorithms under `src/algorithms/<category>/`
- One file per algorithm: `src/algorithms/community/leiden.rs`
- Shared utilities in `<category>/mod.rs` or `<category>/utils.rs`

**Error Handling:**
- Return `GraphResult<T>` for all fallible operations
- Avoid `.unwrap()` in execution paths (use `?` operator)
- Provide context in errors: `context("Failed to compute modularity")?`
- Validate parameters in factory, not during execution

**Cancellation Support:**
- Check `ctx.is_cancelled()` in loops (every iteration or every N iterations)
- Return early with descriptive error: `Err(GraphError::Cancelled("LPA cancelled at iteration 42"))`

**Configuration:**
- Expose config via `AlgorithmParams` struct
- Use `expect_*` helpers for required parameters
- Provide sensible defaults
- Document parameter ranges and effects

**Testing:**
- Unit tests in same file (bottom) or `tests/` submodule
- Integration tests in `tests/` directory
- Benchmarks in `benches/<category>_algorithms.rs`

**Example**:
```rust
// src/algorithms/community/leiden.rs
use super::*;

pub struct Leiden {
    resolution: f64,
    iterations: usize,
    seed: Option<u64>,
}

impl Algorithm for Leiden {
    fn id(&self) -> &'static str { "community.leiden" }
    
    fn execute(&self, ctx: &mut Context, mut sg: Subgraph) -> GraphResult<Subgraph> {
        // Algorithm implementation
        for i in 0..self.iterations {
            if ctx.is_cancelled() {
                return Err(GraphError::cancelled("Leiden cancelled at iteration", i));
            }
            // ... iteration logic
        }
        Ok(sg)
    }
}

// Factory registration
pub fn register(registry: &mut Registry) {
    registry.register_factory("community.leiden", |params| {
        let resolution = params.get("resolution").unwrap_or(1.0);
        let iterations = params.get("iterations").unwrap_or(10);
        let seed = params.get("seed");
        
        Ok(Box::new(Leiden { resolution, iterations, seed }))
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_leiden_convergence() {
        // Test implementation
    }
}
```

### Python Algorithm Handles & Docs

**Factory Functions:**
- Expose via `groggy.algorithms.<category>` modules
- Use lowercase function names matching algorithm: `leiden()`, `eigenvector()`
- Return `RustAlgorithmHandle` or custom handle type

**Docstrings:**
- Follow NumPy docstring format
- Include: brief description, parameters (with types and defaults), return value, examples
- Document complexity where relevant
- Cross-reference related algorithms

**Parameter Handling:**
- Use keyword arguments with defaults
- Validate types early (in Python, before FFI call)
- Provide clear error messages for invalid parameters
- Support both dict and kwargs interfaces

**Example**:
```python
# python-groggy/python/groggy/algorithms/community.py
from .base import algorithm

def leiden(resolution: float = 1.0, iterations: int = 10, seed: Optional[int] = None):
    """
    Leiden algorithm for community detection.
    
    Leiden improves on Louvain by guaranteeing connected communities and
    faster convergence. Uses modularity optimization with quality function.
    
    Parameters
    ----------
    resolution : float, default=1.0
        Resolution parameter for modularity. Higher values result in smaller
        communities. Must be positive.
    iterations : int, default=10
        Maximum number of iterations. Algorithm may converge earlier.
        Must be in range [1, 1000].
    seed : int, optional
        Random seed for reproducibility. If None, uses system randomness.
    
    Returns
    -------
    Subgraph
        Input subgraph with 'community' attribute on nodes indicating
        community membership (integer labels).
    
    Examples
    --------
    >>> from groggy.algorithms.community import leiden
    >>> communities = sg.apply(leiden())
    >>> communities = sg.apply(leiden(resolution=1.5, iterations=20))
    
    See Also
    --------
    louvain : Similar algorithm without connectivity guarantee
    spectral : Spectral clustering alternative
    
    Notes
    -----
    Complexity: O(m) per iteration where m is number of edges.
    Typically converges in 10-20 iterations.
    """
    return algorithm(
        "community.leiden",
        defaults={"resolution": resolution, "iterations": iterations, "seed": seed},
    )
```

**Testing Python APIs:**
- Test factory function creates valid handle
- Test parameter validation (type errors, range errors)
- Test end-to-end execution on small graph
- Test docstring examples (doctests or explicit tests)

### Documentation Guidelines

**API Documentation:**
- All public items documented (modules, classes, functions, methods)
- Examples in every docstring
- Type hints in Python, type signatures in Rust
- Cross-references between related items

**Tutorials:**
- Start simple (single algorithm), build to complex (pipelines)
- Use realistic but small examples
- Include visualizations where helpful
- Provide complete, runnable code

**Performance Notes:**
- Document complexity (big-O) for all algorithms
- Note memory requirements for large graphs
- Provide tuning guidance (when to use what)
- Include benchmark results for reference

### Code Review Checklist

Before submitting:

**Rust:**
- [ ] `cargo fmt --all` passes
- [ ] `cargo clippy --all-targets -- -D warnings` passes
- [ ] `cargo test` passes
- [ ] New code has unit tests
- [ ] Complex algorithms have integration tests
- [ ] Benchmarks added for performance-critical code
- [ ] Docstrings complete with examples

**Python:**
- [ ] `black .` and `isort .` pass
- [ ] Type hints present and correct
- [ ] `pytest tests/` passes
- [ ] New APIs have test coverage
- [ ] Docstrings follow NumPy format
- [ ] Examples in docstrings are runnable

**Documentation:**
- [ ] README updated if needed
- [ ] API reference updated
- [ ] Migration notes if breaking changes
- [ ] CHANGELOG.md updated

**FFI:**
- [ ] No unsafe code without justification
- [ ] GIL released for expensive operations
- [ ] Error handling complete (no panics across FFI)
- [ ] Memory management correct (no leaks)

---

