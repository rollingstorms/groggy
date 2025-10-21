# Experimental Prototyping Workflow

## Overview

The experimental delegation system allows rapid prototyping of new graph operations without committing to the stable API surface. This guide explains how to develop, test, and graduate experimental features.

## Quick Start

### Enable Experimental Features

**Option 1: Build Time (Recommended for Development)**
```bash
# Build with experimental features
maturin develop --features experimental-delegation --release

# Or with cargo
cargo build --features experimental-delegation
```

**Option 2: Runtime Environment Variable**
```bash
# Set environment variable
export GROGGY_EXPERIMENTAL=1

# Run your script
python my_script.py
```

### Using Experimental Methods

```python
import groggy

g = groggy.Graph()
# ... populate graph ...

# List available experimental methods
methods = g.experimental("list")
print(f"Available: {methods}")

# Get method description
desc = g.experimental("describe", "pagerank")
print(desc)

# Call experimental method
scores = g.experimental("pagerank", damping=0.85)
```

## Adding a New Experimental Method

### Step 1: Implement Core Algorithm (Optional)

If your method requires new Rust implementation:

```rust
// In src/algorithms/centrality/pagerank.rs (create if needed)

use crate::core::graph::Graph;
use crate::storage::table::NodesTable;
use crate::types::GraphResult;

pub fn pagerank(
    graph: &Graph,
    damping: f64,
    max_iterations: usize,
) -> GraphResult<NodesTable> {
    // Implementation here
    todo!("PageRank algorithm")
}
```

Add to trait if applicable:

```rust
// In src/traits/centrality.rs

pub trait CentralityOps {
    fn pagerank(&self, damping: f64) -> GraphResult<NodesTable> {
        pagerank_core(self.graph_ref(), damping, 100)
    }
}
```

### Step 2: Register in Experimental Registry

```rust
// In python-groggy/src/ffi/experimental.rs

impl ExperimentalRegistry {
    pub fn new() -> Self {
        let mut methods = HashMap::new();

        #[cfg(feature = "experimental-delegation")]
        {
            // Add your method here
            methods.insert(
                "pagerank".to_string(),
                ExperimentalMethod {
                    name: "pagerank".to_string(),
                    description: "Calculate PageRank centrality scores".to_string(),
                    handler: experimental_pagerank,
                },
            );
        }

        Self { methods }
    }
}

// Implement the handler
#[cfg(feature = "experimental-delegation")]
fn experimental_pagerank(
    obj: PyObject,
    py: Python,
    args: &PyTuple,
    kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    // Extract PyGraph from obj
    let graph: PyRef<crate::ffi::api::graph::PyGraph> = obj.extract(py)?;
    
    // Parse arguments
    let damping: f64 = if args.len() > 0 {
        args.get_item(0)?.extract()?
    } else if let Some(kw) = kwargs {
        kw.get_item("damping")
            .and_then(|item| item.map(|v| v.extract().ok()).flatten())
            .unwrap_or(0.85)
    } else {
        0.85
    };
    
    // Call core implementation
    let graph_ref = graph.inner.borrow();
    let result = groggy::algorithms::centrality::pagerank(
        &graph_ref,
        damping,
        100,  // max_iterations
    ).map_err(crate::ffi::utils::graph_error_to_py_err)?;
    
    // Convert to Python
    let py_table = crate::ffi::storage::table::PyNodesTable { table: result };
    Ok(py_table.into_py(py))
}
```

### Step 3: Test Your Implementation

```python
# tests/test_experimental.py

import groggy
import pytest
import os

# Skip if experimental features not enabled
@pytest.mark.skipif(
    "experimental-delegation" not in os.getenv("GROGGY_FEATURES", ""),
    reason="Experimental features not enabled"
)
def test_pagerank():
    g = groggy.Graph()
    
    # Create simple graph
    g.add_node(0)
    g.add_node(1)
    g.add_edge(0, 1)
    
    # Call experimental method
    scores = g.experimental("pagerank", damping=0.85)
    
    assert scores is not None
    assert len(scores) == 2
```

### Step 4: Iterate and Refine

1. **Test in notebooks** - Use experimental methods in Jupyter notebooks for exploration
2. **Gather feedback** - Share with team/users to refine interface
3. **Performance profile** - Use `cargo bench` to ensure acceptable performance
4. **Documentation** - Update experimental method description in registry

## Graduating to Stable API

Once an experimental method is stable:

### Step 1: Add Explicit PyO3 Method

```rust
// In python-groggy/src/ffi/api/graph.rs

impl PyGraph {
    /// Calculate PageRank centrality scores.
    ///
    /// Args:
    ///     damping: Damping factor (default 0.85)
    ///     max_iterations: Maximum iterations (default 100)
    ///
    /// Returns:
    ///     NodesTable with PageRank scores
    #[pyo3(signature = (damping = 0.85, max_iterations = 100))]
    pub fn pagerank(
        slf: PyRef<Self>,
        py: Python,
        damping: f64,
        max_iterations: usize,
    ) -> PyResult<crate::ffi::storage::table::PyNodesTable> {
        Self::with_full_view(slf, py, |subgraph, _py| {
            let result = subgraph
                .inner
                .pagerank(damping)  // Trait method
                .map_err(crate::ffi::utils::graph_error_to_py_err)?;
            Ok(crate::ffi::storage::table::PyNodesTable { table: result })
        })
    }
}
```

### Step 2: Remove from Experimental Registry

```rust
// In python-groggy/src/ffi/experimental.rs
// Remove the method registration and handler
```

### Step 3: Update Documentation

1. Add to API documentation
2. Update CHANGELOG
3. Add migration note in release notes
4. Update stubs: `python scripts/generate_stubs.py`

### Step 4: Deprecation Notice (Optional)

If method name or signature changed, add deprecation warning:

```rust
#[cfg(feature = "experimental-delegation")]
fn experimental_pagerank_old(
    obj: PyObject,
    py: Python,
    args: &PyTuple,
    kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    py.run(
        r#"
import warnings
warnings.warn(
    "graph.experimental('pagerank') is deprecated. Use graph.pagerank() instead.",
    DeprecationWarning,
    stacklevel=2
)
        "#,
        None,
        None,
    )?;
    
    // Delegate to stable implementation...
}
```

## Best Practices

### DO ✅

- **Start with placeholder** - Register method early, even if not implemented
- **Use feature flags** - Keep experimental code behind `#[cfg(feature = "...")]`
- **Document thoroughly** - Explain purpose, args, return type in registry
- **Test early** - Add tests even for incomplete implementations
- **Gather feedback** - Share experimental methods with users before stabilizing
- **Profile performance** - Ensure methods meet performance requirements

### DON'T ❌

- **Don't skip core implementation** - Experimental methods should still follow trait patterns
- **Don't ignore errors** - Proper error handling even in prototypes
- **Don't break existing code** - Experimental features should be additive
- **Don't stabilize prematurely** - Gather sufficient feedback before graduating
- **Don't forget migration** - Plan for moving users from experimental to stable

## CI/CD Integration

### GitHub Actions

```yaml
# .github/workflows/experimental.yml

name: Test Experimental Features

on: [push, pull_request]

jobs:
  test-experimental:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3
      
      - name: Install Rust
        uses: actions-rs/toolchain@v1
        with:
          toolchain: stable
      
      - name: Install maturin
        run: pip install maturin
      
      - name: Build with experimental features
        run: maturin develop --features experimental-delegation --release
      
      - name: Run experimental tests
        run: GROGGY_EXPERIMENTAL=1 pytest tests/ -m experimental
```

### Local Development

```bash
# Create development environment with experimental features
python -m venv venv-experimental
source venv-experimental/bin/activate
maturin develop --features experimental-delegation --release

# Run tests
GROGGY_EXPERIMENTAL=1 pytest tests/test_experimental.py -v
```

## Troubleshooting

### Feature Flag Not Working

**Problem**: Experimental methods not available despite setting flag

**Solution**:
1. Rebuild with feature flag: `maturin develop --features experimental-delegation --release`
2. Verify build: `python -c "import groggy; print(groggy.Graph().experimental('list'))"`
3. Check environment: `echo $GROGGY_EXPERIMENTAL`

### Method Not Found

**Problem**: `AttributeError: Experimental method 'xyz' not found`

**Solution**:
1. List available methods: `graph.experimental("list")`
2. Check registry in `experimental.rs`
3. Ensure method is registered inside `#[cfg(feature = "experimental-delegation")]`

### Type Conversion Errors

**Problem**: `TypeError` when calling experimental method

**Solution**:
1. Check argument extraction in handler
2. Verify Python type matches Rust type
3. Use `.extract::<Type>()?` with proper error handling
4. Add type hints to method description

## Examples

### Example 1: Simple Algorithm

```rust
#[cfg(feature = "experimental-delegation")]
fn experimental_graph_density(
    obj: PyObject,
    py: Python,
    _args: &PyTuple,
    _kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    let graph: PyRef<crate::ffi::api::graph::PyGraph> = obj.extract(py)?;
    let graph_ref = graph.inner.borrow();
    
    let node_count = graph_ref.node_count() as f64;
    let edge_count = graph_ref.edge_count() as f64;
    
    let density = if node_count > 1.0 {
        edge_count / (node_count * (node_count - 1.0))
    } else {
        0.0
    };
    
    Ok(density.to_object(py))
}
```

### Example 2: Complex Operation with Options

```rust
#[cfg(feature = "experimental-delegation")]
fn experimental_community_detection(
    obj: PyObject,
    py: Python,
    args: &PyTuple,
    kwargs: Option<&PyDict>,
) -> PyResult<PyObject> {
    let graph: PyRef<crate::ffi::api::graph::PyGraph> = obj.extract(py)?;
    
    // Parse algorithm option
    let algorithm: String = if let Some(kw) = kwargs {
        kw.get_item("algorithm")
            .and_then(|item| item.map(|v| v.extract().ok()).flatten())
            .unwrap_or_else(|| "louvain".to_string())
    } else {
        "louvain".to_string()
    };
    
    // Parse resolution parameter
    let resolution: f64 = if let Some(kw) = kwargs {
        kw.get_item("resolution")
            .and_then(|item| item.map(|v| v.extract().ok()).flatten())
            .unwrap_or(1.0)
    } else {
        1.0
    };
    
    // Call implementation
    match algorithm.as_str() {
        "louvain" => {
            // TODO: Implement Louvain
            Err(pyo3::exceptions::PyNotImplementedError::new_err(
                "Louvain not yet implemented"
            ))
        }
        _ => {
            Err(pyo3::exceptions::PyValueError::new_err(
                format!("Unknown algorithm: {}", algorithm)
            ))
        }
    }
}
```

## References

- [Cargo Features Documentation](https://doc.rust-lang.org/cargo/reference/features.html)
- [PyO3 Guide](https://pyo3.rs/)
- [Trait Delegation System Plan](./trait_delegation_system_plan.md)
- [API Design Guidelines](../../docs/api_design.md)

---

**Last Updated**: 2025-01-XX  
**Maintainer**: Bridge Persona (FFI Layer)
