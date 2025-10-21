# Explicit Delegation Design: Making Current System Discoverable

**Date:** October 14, 2025
**Goal:** Keep working delegation, make it explicit and type-safe
**Complexity:** 2-3 weeks (vs 4-5 weeks for full trait system)

---

## Current Working Pattern

Your system already works beautifully:

```rust
// Graph.__getattr__ (graph.rs:1827-1854)
fn __getattr__(&self, py: Python, name: String) -> PyResult<PyObject> {
    // ... attribute handling ...

    // Delegate to subgraph methods
    let concrete_subgraph = Subgraph::new(
        self.inner.clone(),
        all_nodes,
        all_edges,
        "full_graph_delegation"
    );

    let py_subgraph = PySubgraph::from_core_subgraph(concrete_subgraph)?;
    let subgraph_obj = Py::new(py, py_subgraph)?;
    subgraph_obj.getattr(py, name.as_str())  // Magic happens here
}
```

```rust
// NeighborhoodArray.__getattr__ (neighborhood.rs:84-88)
fn __getattr__(&self, name: &str, py: Python) -> PyResult<PyObject> {
    let subgraph = self.subgraph(py)?;
    let subgraph_obj = Py::new(py, subgraph)?;
    subgraph_obj.getattr(py, name)  // Delegates everything
}
```

**What works:**
- ✅ Graph delegates to PySubgraph (which has explicit methods)
- ✅ NeighborhoodArray delegates to PySubgraph
- ✅ PySubgraph has ~30+ explicit methods (degree, in_degree, connected_components, etc.)

**What's opaque:**
- ❌ No way to know Graph has degree() without reading __getattr__ code
- ❌ No IDE autocomplete for delegated methods
- ❌ No documentation showing what's delegated

---

## Design Solution: Explicit Delegation with Macros

Keep your working delegation pattern, but make it explicit using Rust macros to generate forwarding methods.

### Architecture Overview

```
┌─────────────────────────────────────────────┐
│ Core Types (have explicit methods)         │
│ - PySubgraph: degree(), in_degree(), etc.  │
│ - PyNodesTable: filter(), select(), etc.   │
│ - PyBaseArray: map(), filter(), etc.       │
└─────────────────────────────────────────────┘
                    ▲
                    │ delegates to
                    │
┌─────────────────────────────────────────────┐
│ Delegating Types (use macro)               │
│ - PyGraph → PySubgraph                     │
│ - PyNeighborhoodArray → PySubgraph         │
│ - PyNodesTable → PyBaseTable               │
│ - PyEdgesTable → PyBaseTable               │
└─────────────────────────────────────────────┘
```

### Key Insight

**You don't need traits in Rust core.** You just need to:
1. List what methods get delegated
2. Generate explicit forwarding methods
3. Keep `__getattr__` as fallback for edge cases

---

## Implementation Design

### Phase 1: Delegation Macro (Week 1)

Create a macro that generates explicit forwarding methods.

**Location:** `python-groggy/src/ffi/macros/delegation.rs`

```rust
/// Macro to generate explicit delegation methods for PyO3 classes
///
/// Usage:
/// ```
/// delegate_to_subgraph! {
///     target: self.to_subgraph(py),
///     methods: [
///         (degree, "Calculate node degrees", nodes: Option<&PyAny>, full_graph: bool = false),
///         (in_degree, "Calculate in-degrees", nodes: Option<&PyAny>, full_graph: bool = false),
///         (out_degree, "Calculate out-degrees", nodes: Option<&PyAny>, full_graph: bool = false),
///         (connected_components, "Find connected components"),
///         (table, "Convert to table"),
///     ]
/// }
/// ```
#[macro_export]
macro_rules! delegate_to_subgraph {
    // Pattern: Method with args
    (
        target: $target_expr:expr,
        methods: [
            $(
                ($method_name:ident, $doc:literal $(, $arg_name:ident: $arg_type:ty $(= $default:expr)? )* )
            ),* $(,)?
        ]
    ) => {
        $(
            #[doc = $doc]
            #[doc = ""]
            #[doc = "**Note:** This method is delegated to the underlying subgraph."]
            pub fn $method_name(
                &self,
                py: Python,
                $( $arg_name: $arg_type ),*
            ) -> PyResult<PyObject> {
                let target = $target_expr?;
                let target_obj = Py::new(py, target)?;

                // Call the method on the target
                let method = target_obj.getattr(py, stringify!($method_name))?;
                method.call1(py, ( $( $arg_name, )* ))
            }
        )*
    };
}

/// Simpler version for methods with no args
#[macro_export]
macro_rules! delegate_methods {
    (
        to: $delegate_method:ident,
        methods: [ $( $method_name:ident ),* $(,)? ]
    ) => {
        $(
            #[doc = concat!("Delegates to `", stringify!($method_name), "()` on the underlying object.")]
            pub fn $method_name(&self, py: Python) -> PyResult<PyObject> {
                let target = self.$delegate_method(py)?;
                let target_obj = Py::new(py, target)?;
                target_obj.call_method0(py, stringify!($method_name))
            }
        )*
    };
}

/// Generate both explicit methods AND __getattr__ fallback
#[macro_export]
macro_rules! explicit_delegation {
    (
        struct $struct_name:ident;
        delegate_to: $target_expr:expr;
        explicit_methods: {
            $( $method_name:ident ( $( $arg_name:ident : $arg_type:ty ),* $(,)? ) -> $ret_type:ty; )*
        }
        $(fallback_attrs: [ $( $fallback_attr:literal ),* $(,)? ] ;)?
    ) => {
        impl $struct_name {
            // Generate explicit forwarding methods
            $(
                pub fn $method_name(
                    &self,
                    py: Python,
                    $( $arg_name: $arg_type ),*
                ) -> PyResult<$ret_type> {
                    let target = $target_expr;
                    let target_obj = Py::new(py, target)?;
                    target_obj.call_method1(py, stringify!($method_name), ( $( $arg_name, )* ))
                }
            )*

            // Generate __getattr__ that delegates everything else
            pub fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
                // Skip Python internals
                if name.starts_with("_") {
                    return Err(PyAttributeError::new_err(format!(
                        "'{}' object has no attribute '{}'",
                        stringify!($struct_name),
                        name
                    )));
                }

                // Try delegation
                let target = $target_expr;
                let target_obj = Py::new(py, target)?;
                target_obj.getattr(py, name)
            }
        }
    };
}
```

### Phase 2: Apply to PyGraph (Week 1-2)

**Location:** `python-groggy/src/ffi/api/graph.rs`

```rust
use crate::delegate_to_subgraph;

#[pymethods]
impl PyGraph {
    // === Explicit delegated methods (generated by macro) ===

    delegate_to_subgraph! {
        target: self.to_full_graph_subgraph(py),
        methods: [
            (degree,
             "Calculate degree of nodes. If nodes is None, returns dict of all degrees.",
             nodes: Option<&PyAny>,
             full_graph: bool = false),

            (in_degree,
             "Calculate in-degree of nodes in directed graphs.",
             nodes: Option<&PyAny>,
             full_graph: bool = false),

            (out_degree,
             "Calculate out-degree of nodes in directed graphs.",
             nodes: Option<&PyAny>,
             full_graph: bool = false),

            (connected_components,
             "Find all connected components in the graph."),

            (bfs,
             "Breadth-first search from starting node.",
             start: NodeId,
             max_depth: Option<usize>),

            (dfs,
             "Depth-first search from starting node.",
             start: NodeId,
             max_depth: Option<usize>),

            (shortest_path,
             "Find shortest path between two nodes.",
             source: NodeId,
             target: NodeId),

            (diameter,
             "Calculate graph diameter."),

            (density,
             "Calculate graph density."),
        ]
    }

    // Helper method to create full-graph subgraph
    fn to_full_graph_subgraph(&self, py: Python) -> PyResult<PySubgraph> {
        let graph_ref = self.inner.borrow();
        let all_nodes = graph_ref.node_ids().into_iter().collect();
        let all_edges = graph_ref.edge_ids().into_iter().collect();
        drop(graph_ref);

        let subgraph = groggy::subgraphs::Subgraph::new(
            self.inner.clone(),
            all_nodes,
            all_edges,
            "full_graph_delegation".to_string(),
        );

        PySubgraph::from_core_subgraph(subgraph)
    }

    // Keep __getattr__ for attributes and fallback
    fn __getattr__(&self, py: Python, name: String) -> PyResult<PyObject> {
        // 1. Try node attributes
        let all_node_attrs = self.all_node_attribute_names();
        if all_node_attrs.contains(&name) {
            return self.get_node_attribute_dict(py, &name);
        }

        // 2. Try edge attributes
        let all_edge_attrs = self.all_edge_attribute_names();
        if all_edge_attrs.contains(&name) {
            return self.get_edge_attribute_dict(py, &name);
        }

        // 3. Delegate to subgraph (for methods not in macro)
        let subgraph = self.to_full_graph_subgraph(py)?;
        let subgraph_obj = Py::new(py, subgraph)?;
        subgraph_obj.getattr(py, name.as_str())
    }
}
```

**What this generates:**

```rust
// Explicit methods (autocomplete works, docs generate)
impl PyGraph {
    /// Calculate degree of nodes. If nodes is None, returns dict of all degrees.
    ///
    /// **Note:** This method is delegated to the underlying subgraph.
    pub fn degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        let target = self.to_full_graph_subgraph(py)?;
        let target_obj = Py::new(py, target)?;
        let method = target_obj.getattr(py, "degree")?;
        method.call1(py, (nodes, full_graph))
    }

    /// Calculate in-degree of nodes in directed graphs.
    ///
    /// **Note:** This method is delegated to the underlying subgraph.
    pub fn in_degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        let target = self.to_full_graph_subgraph(py)?;
        let target_obj = Py::new(py, target)?;
        let method = target_obj.getattr(py, "in_degree")?;
        method.call1(py, (nodes, full_graph))
    }

    // ... etc for all methods in macro
}
```

### Phase 3: Apply to Other Delegating Types (Week 2)

#### PyNeighborhoodArray

```rust
#[pymethods]
impl PyNeighborhoodArray {
    delegate_methods! {
        to: subgraph,
        methods: [
            degree,
            in_degree,
            out_degree,
            connected_components,
            table,
            density,
            diameter,
        ]
    }

    // Keep specialized methods explicit
    fn central_nodes(&self) -> Vec<NodeId> {
        self.inner.central_nodes().to_vec()
    }

    fn hops(&self) -> usize {
        self.inner.hops()
    }

    // Fallback for unknown methods
    fn __getattr__(&self, name: &str, py: Python) -> PyResult<PyObject> {
        let subgraph = self.subgraph(py)?;
        let subgraph_obj = Py::new(py, subgraph)?;
        subgraph_obj.getattr(py, name)
    }
}
```

#### PyNodesTable / PyEdgesTable

```rust
#[pymethods]
impl PyNodesTable {
    delegate_methods! {
        to: base_table,
        methods: [
            head,
            tail,
            select,
            filter,
            sort_by,
            group_by,
            agg,
            unique,
        ]
    }

    fn __getattr__(&self, py: Python, name: &str) -> PyResult<PyObject> {
        let base = self.base_table();
        let base_obj = base.into_py(py);
        base_obj.getattr(py, name)
    }
}
```

---

## Python Type Stubs (Auto-generation)

Generate `.pyi` stubs that expose delegated methods.

**Location:** `python-groggy/python/groggy/_groggy.pyi`

```python
# Auto-generated from delegation macros

class Graph:
    """Groggy graph with node/edge storage and algorithms."""

    # === Delegated subgraph operations ===

    def degree(
        self,
        nodes: Optional[Union[int, List[int]]] = None,
        full_graph: bool = False
    ) -> Union[int, Dict[int, int]]:
        """
        Calculate degree of nodes. If nodes is None, returns dict of all degrees.

        Note: This method is delegated to the underlying subgraph.

        Args:
            nodes: Single node ID, list of node IDs, or None for all nodes
            full_graph: If True, calculate degree in full graph (not just subgraph)

        Returns:
            Single degree (int) or dict mapping node IDs to degrees
        """
        ...

    def in_degree(
        self,
        nodes: Optional[Union[int, List[int]]] = None,
        full_graph: bool = False
    ) -> Union[int, Dict[int, int]]:
        """
        Calculate in-degree of nodes in directed graphs.

        Note: This method is delegated to the underlying subgraph.
        """
        ...

    def connected_components(self) -> SubgraphArray:
        """
        Find all connected components in the graph.

        Note: This method is delegated to the underlying subgraph.

        Returns:
            SubgraphArray containing one subgraph per component
        """
        ...

    # ... etc for all delegated methods

    # === Attribute access ===

    def __getattr__(self, name: str) -> Any:
        """
        Dynamic attribute access for node/edge attributes.

        - If name is a node attribute: returns Dict[NodeId, Value]
        - If name is an edge attribute: returns Dict[EdgeId, Value]
        - Otherwise: delegates to subgraph methods
        """
        ...
```

### Auto-generation Script

**Location:** `scripts/generate_delegation_stubs.py`

```python
#!/usr/bin/env python3
"""
Generate Python type stubs from Rust delegation macros.

Parses delegation macro invocations and generates .pyi files.
"""

import re
from pathlib import Path
from typing import List, Tuple

def parse_delegation_macro(rust_source: str) -> List[Tuple[str, str, List[str]]]:
    """
    Parse delegate_to_subgraph! macro invocations.

    Returns: List of (method_name, docstring, args)
    """
    pattern = r'delegate_to_subgraph!\s*\{[^}]+methods:\s*\[(.*?)\]'
    matches = re.findall(pattern, rust_source, re.DOTALL)

    methods = []
    for match in matches:
        # Parse each method entry
        method_pattern = r'\((\w+),\s*"([^"]+)"(?:,\s*(\w+):\s*([\w<>]+))*\)'
        method_matches = re.findall(method_pattern, match)

        for method_name, doc, arg_name, arg_type in method_matches:
            # Collect all args for this method
            args = []
            if arg_name:
                args.append(f"{arg_name}: {arg_type}")

            methods.append((method_name, doc, args))

    return methods

def generate_stub(class_name: str, methods: List[Tuple[str, str, List[str]]]) -> str:
    """Generate Python stub code for delegated methods."""
    lines = [f"class {class_name}:"]

    for method_name, doc, args in methods:
        lines.append(f"    def {method_name}(self, {', '.join(args)}) -> Any:")
        lines.append(f'        """')
        lines.append(f'        {doc}')
        lines.append(f'        ')
        lines.append(f'        Note: This method is delegated to the underlying subgraph.')
        lines.append(f'        """')
        lines.append(f'        ...')
        lines.append('')

    return '\n'.join(lines)

def main():
    # Parse Rust source files
    graph_rs = Path("python-groggy/src/ffi/api/graph.rs").read_text()
    methods = parse_delegation_macro(graph_rs)

    # Generate stub
    stub_code = generate_stub("Graph", methods)

    # Write to .pyi file
    stub_file = Path("python-groggy/python/groggy/_groggy.pyi")
    stub_file.write_text(stub_code)

    print(f"Generated {len(methods)} method stubs for Graph")

if __name__ == "__main__":
    main()
```

---

## Benefits Over Current System

| Aspect | Current `__getattr__` | Explicit Delegation | Full Trait System |
|--------|----------------------|---------------------|-------------------|
| **Discoverability** | ❌ Hidden | ✅ Explicit + docs | ✅ Explicit + docs |
| **IDE Autocomplete** | ❌ No | ✅ Yes | ✅ Yes |
| **Type Safety** | ⚠️ Runtime | ✅ Compile-time | ✅ Compile-time |
| **Documentation** | ❌ Manual | ✅ Auto-generated | ✅ Auto-generated |
| **Flexibility** | ✅ Very flexible | ✅ Flexible | ⚠️ Rigid |
| **Boilerplate** | ✅ Minimal | ✅ Macro-based | ❌ High |
| **Implementation Time** | ✅ Done | ⚠️ 2-3 weeks | ❌ 4-5 weeks |
| **Maintains Current Pattern** | ✅ Yes | ✅ Yes | ❌ Complete rewrite |

---

## Implementation Timeline

### Week 1: Macro Infrastructure
**Days 1-2:** Create delegation macros
- `delegate_to_subgraph!` macro with full arg support
- `delegate_methods!` macro for simple cases
- Unit tests for macro expansion

**Days 3-4:** Apply to PyGraph
- List all delegated methods in macro
- Test that all methods work
- Keep `__getattr__` for attributes

**Day 5:** Generate type stubs
- Create stub generation script
- Generate `.pyi` for PyGraph
- Test IDE autocomplete

### Week 2: Expand to Other Types
**Days 1-2:** PyNeighborhoodArray, PyComponentArray
- Apply delegation macros
- Test method delegation
- Generate stubs

**Days 3-4:** PyNodesTable, PyEdgesTable
- Apply delegation macros
- Test table operations
- Generate stubs

**Day 5:** PyBaseArray and specialized arrays
- Apply to array types
- Complete stub generation
- Integration testing

### Week 3: Polish & Documentation
**Days 1-2:** Comprehensive testing
- Test all delegated methods
- Verify IDE autocomplete
- Performance benchmarking

**Days 3-4:** Documentation
- Update user docs with delegation info
- Add "Delegated Methods" section to API docs
- Write migration guide

**Day 5:** Release & validation
- Final code review
- Merge to main
- Update changelog

---

## Example Usage (After Implementation)

```python
import groggy as gr

g = gr.Graph()
# ... add nodes/edges ...

# NOW DISCOVERABLE: IDE shows these methods
degrees = g.degree()  # Autocomplete works!
components = g.connected_components()  # Type hints work!
path = g.shortest_path(0, 5)  # Documentation shows up!

# Attribute access still works
ages = g.age  # Returns dict of node ages

# Delegation still works for unlisted methods
g.some_future_method()  # Falls back to __getattr__
```

**VS Code / PyCharm:**
```
g.deg<cursor>
  ↓ Shows autocomplete:
  - degree() -> Union[int, Dict[int, int]]
    Calculate degree of nodes...
    Note: Delegated to underlying subgraph
```

---

## Migration Path

### Option 1: Gradual (Recommended)
1. Implement macros for high-use methods first (degree, connected_components)
2. Keep `__getattr__` for everything else
3. Gradually expand macro coverage
4. Eventually remove `__getattr__` fallback

### Option 2: Comprehensive
1. List ALL methods that should be delegated
2. Apply macros to all delegating types at once
3. Keep `__getattr__` only for attribute access
4. Complete in 2-3 weeks

### Option 3: Hybrid Forever
1. Use macros for core API methods
2. Keep `__getattr__` for advanced/experimental features
3. Best of both worlds: discovery + flexibility

---

## Comparison to Full Trait System

**Explicit Delegation Pros:**
- ✅ 2-3 weeks vs 4-5 weeks
- ✅ Keeps working pattern (delegates to PySubgraph)
- ✅ No changes to Rust core needed
- ✅ Macro-based boilerplate (manageable)
- ✅ Can keep `__getattr__` fallback

**Explicit Delegation Cons:**
- ⚠️ Still some dynamic dispatch (creating subgraph)
- ⚠️ Not trait-based (less "proper" Rust)
- ⚠️ Macro debugging can be tricky

**When to choose this over traits:**
- Your delegation pattern already works perfectly
- 2-3 weeks is better than 4-5 weeks
- Don't want to refactor Rust core
- Value pragmatism over purity

**When to choose traits instead:**
- Want "proper" Rust architecture
- Planning major refactoring anyway
- Need trait objects for extensibility
- Have 4-5 weeks available

---

## Conclusion

This design gives you **90% of trait system benefits at 50% of the cost**:

✅ **Explicit**: Methods listed in macro, not hidden
✅ **Discoverable**: IDE autocomplete, type hints, docs
✅ **Type-safe**: Compile-time method checking
✅ **Fast**: 2-3 weeks implementation
✅ **Low-risk**: Keeps working delegation pattern
✅ **Flexible**: Can still use `__getattr__` for edge cases

It's the pragmatic middle ground between "opaque __getattr__" and "full trait rewrite".

**Next Step:** Would you like me to implement the delegation macro and apply it to PyGraph as a proof-of-concept?
