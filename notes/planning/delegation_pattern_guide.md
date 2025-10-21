# Delegation Pattern Guide

## Architecture Overview

We use **explicit PyO3 methods backed by Rust traits** for all Python-facing functionality. This approach provides:

- **Discoverability**: All methods visible via `dir()`, IDEs, and type stubs
- **Maintainability**: Clear, readable code without macro magic
- **Single Source of Truth**: Logic lives in Rust traits with default implementations
- **Thin Wrappers**: PyO3 methods are simple delegators with error translation

## Core Pattern: Explicit Wrapper → Helper → Trait

```rust
// 1. Rust trait provides the logic (in src/traits/)
pub trait SubgraphOperations {
    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>>;
}

// 2. Helper standardizes access to the trait implementation
impl PyGraph {
    pub(crate) fn with_full_view<R, F>(
        graph_ref: PyRef<Self>,
        py: Python,
        f: F,
    ) -> PyResult<R>
    where
        F: for<'a> FnOnce(PyRef<'a, PySubgraph>, Python<'a>) -> PyResult<R>,
    {
        let view = Self::view(graph_ref, py)?;
        f(view.borrow(py), py)
    }
}

// 3. Explicit PyO3 method wraps the trait call
#[pymethods]
impl PyGraph {
    pub fn connected_components(
        slf: PyRef<Self>,
        py: Python,
    ) -> PyResult<PyComponentsArray> {
        Self::with_full_view(slf, py, |subgraph, _py| {
            // Access inner Rust type and call trait method
            let components = subgraph
                .inner
                .connected_components()
                .map_err(graph_error_to_py_err)?;
            
            // Convert to Python type
            Ok(PyComponentsArray::from_components(
                components,
                subgraph.inner.graph().clone()
            ))
        })
    }
}
```

## Adding a New Method

### Step 1: Ensure Trait Method Exists

Check `src/traits/` for the relevant trait:
- `SubgraphOperations` - graph structure operations
- `ComponentOperations` - connected component operations
- `NodeOperations` - node-specific operations
- `EdgeOperations` - edge-specific operations
- `FilterOperations` - filtering operations
- `MetaNodeOperations` / `MetaEdgeOperations` - hierarchy operations

If the method doesn't exist, add it to the appropriate trait with a default implementation.

### Step 2: Identify the Helper Pattern

Different classes use different helpers:

**PyGraph** → uses `with_full_view` to access full graph as subgraph:
```rust
pub fn method_name(slf: PyRef<Self>, py: Python, ...) -> PyResult<ReturnType> {
    Self::with_full_view(slf, py, |subgraph, _py| {
        // Call trait method on subgraph.inner
        // Convert result to Python type
    })
}
```

**PySubgraph** → directly accesses `self.inner`:
```rust
pub fn method_name(&self, py: Python, ...) -> PyResult<ReturnType> {
    let result = self.inner
        .trait_method(...)
        .map_err(graph_error_to_py_err)?;
    
    // Convert to Python type
    Ok(PyType::from_rust(result))
}
```

**PyNodesTable / PyEdgesTable** → access `self.base_table.inner`:
```rust
pub fn method_name(&self, ...) -> PyResult<ReturnType> {
    // Call trait method on inner table
    // Handle conversion
}
```

### Step 3: Write the Explicit Method

Add to the appropriate `#[pymethods]` block with:

1. **Clear signature** matching Python expectations
2. **Docstring** explaining purpose, params, returns, examples
3. **Signature macro** if using optional/keyword args: `#[pyo3(signature = (...))]`
4. **Error translation** via `map_err(graph_error_to_py_err)`
5. **GIL release** for long operations: wrap in `py.allow_threads(|| ...)`
6. **Type conversion** from Rust → Python types

### Step 4: Organize and Document

- Group methods logically (topology, analysis, conversion, etc.)
- Add section comments: `// === ANALYSIS OPERATIONS ===`
- Update `trait_delegation_system_plan.md` Progress Log
- Update `trait_delegation_surface_catalog.md` status

### Step 5: Test and Verify

```bash
# Rust compilation
cargo check --all-features
cargo clippy --all-targets -- -D warnings
cargo fmt --all

# Python build
maturin develop --release

# Verification
python -c "import groggy; print('method_name' in dir(groggy.Graph()))"
pytest tests -q
```

## Common Patterns

### Pattern: Simple Delegation

For methods that directly map to trait methods:

```rust
pub fn node_count(&self) -> usize {
    self.inner.node_count()  // Trait method
}
```

### Pattern: With Error Handling

```rust
pub fn has_path(
    slf: PyRef<Self>,
    py: Python,
    source: NodeId,
    target: NodeId,
) -> PyResult<bool> {
    Self::with_full_view(slf, py, |subgraph, _py| {
        subgraph
            .inner
            .bfs(source, None)
            .map(|bfs_subgraph| bfs_subgraph.contains_node(target))
            .map_err(graph_error_to_py_err)
    })
}
```

### Pattern: With Type Conversion

```rust
pub fn connected_components(
    slf: PyRef<Self>,
    py: Python,
) -> PyResult<PyComponentsArray> {
    Self::with_full_view(slf, py, |subgraph, _py| {
        // Call trait method
        let rust_components = subgraph
            .inner
            .connected_components()
            .map_err(graph_error_to_py_err)?;
        
        // Convert to Python type
        let py_components = PyComponentsArray::from_components(
            rust_components,
            subgraph.inner.graph().clone(),
        );
        
        Ok(py_components)
    })
}
```

### Pattern: With Optional Parameters

```rust
#[pyo3(signature = (node_id = None))]
pub fn clustering_coefficient(
    slf: PyRef<Self>,
    py: Python,
    node_id: Option<NodeId>,
) -> PyResult<f64> {
    Self::with_full_view(slf, py, |subgraph, _py| {
        match node_id {
            Some(id) => {
                // Per-node calculation
                Ok(calculate_node_clustering(&subgraph.inner, id))
            }
            None => {
                // Global calculation
                Ok(calculate_global_clustering(&subgraph.inner))
            }
        }
    })
}
```

### Pattern: With GIL Release

For computationally expensive operations:

```rust
pub fn expensive_operation(
    slf: PyRef<Self>,
    py: Python,
) -> PyResult<ReturnType> {
    // Release GIL for the expensive part
    let result = py.allow_threads(|| {
        Self::with_full_view(slf, py, |subgraph, _py| {
            subgraph
                .inner
                .expensive_trait_method()
                .map_err(graph_error_to_py_err)
        })
    })?;
    
    Ok(result)
}
```

## What NOT to Do

❌ **Don't use macros to generate methods** - Write them explicitly
❌ **Don't put logic in PyO3 methods** - Logic goes in traits
❌ **Don't duplicate trait implementations** - Use the trait!
❌ **Don't forget error handling** - Always use `map_err`
❌ **Don't skip documentation** - Every method needs a docstring
❌ **Don't forget to update the catalog** - Track progress in plan docs

## Checklist for Each New Method

- [ ] Trait method exists in `src/traits/`
- [ ] Explicit wrapper added to appropriate PyO3 class
- [ ] Docstring with description, parameters, returns, example
- [ ] Signature macro if needed for optional/keyword args
- [ ] Error translation via `map_err(graph_error_to_py_err)`
- [ ] GIL release via `py.allow_threads()` if long-running
- [ ] Type conversion from Rust → Python types
- [ ] Code formatted with `cargo fmt`
- [ ] Passes `cargo clippy -- -D warnings`
- [ ] Progress log updated in plan document
- [ ] Catalog status updated
- [ ] Tested manually: `python -c "import groggy; ..."`
- [ ] Integration tests pass: `pytest tests -q`

## Helper Functions Reference

### PyGraph Helpers

**`with_full_view(slf, py, f)`** - Access full graph as cached subgraph
- Use for: Methods that need full graph context
- Trait access: `subgraph.inner.trait_method()`

**`view(slf, py)`** - Get cached full-graph subgraph (used by with_full_view)
- Don't call directly in new methods; use `with_full_view` instead

### Future Helpers (To Be Added)

**`PySubgraph::with_trait_view`** - Standardize subgraph trait access
**`PyNodesTable::with_table_ops`** - Standardize table operations
**`PyBaseArray::with_array_ops`** - Standardize array operations

## Examples from Existing Code

See `python-groggy/src/ffi/api/graph.rs` for reference implementations:
- `connected_components()` - Complex type conversion
- `has_path()` - Simple boolean return
- `sample()` - Delegates to public PySubgraph method
- `clustering_coefficient()` - Optional parameters with match
- `transitivity()` - Simple numeric return

## Questions?

Refer to:
- `documentation/planning/trait_delegation_system_plan.md` - Overall architecture
- `documentation/planning/trait_delegation_surface_catalog.md` - Method inventory
- `src/traits/` - Trait definitions and defaults
- Existing implementations in `python-groggy/src/ffi/api/graph.rs`
