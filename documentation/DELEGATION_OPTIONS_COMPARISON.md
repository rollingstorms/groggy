# Delegation System Options: Clear Comparison

**Current Issue:** Graph delegates methods to Subgraph via `__getattr__`, but `.pyi` stubs don't show these methods, so IDE autocomplete doesn't work.

## Your Current State

**What works:**
```rust
// graph.rs:1827-1854
fn __getattr__(&self, py: Python, name: String) -> PyResult<PyObject> {
    // ... attribute handling ...

    // Create full-graph subgraph and delegate methods
    let subgraph = Subgraph::new(all_nodes, all_edges, ...);
    subgraph.getattr(py, name)  // Forwards to PySubgraph methods
}
```

**Result:**
```python
g = gr.Graph()
g.degree()  # ✅ Works at runtime via __getattr__
            # ❌ IDE doesn't know about it (not in .pyi stub)
            # ❌ No autocomplete
```

**Your `.pyi` stubs:**
- ✅ `Subgraph` has `degree()`, `connected_components()`, etc.
- ❌ `Graph` class doesn't list these delegated methods
- Result: Runtime works, but not discoverable

---

## Option 1: Macro Delegation (ON TOP OF CURRENT)

### What It Is
**Add explicit forwarding methods** to PyGraph using macros. **No refactoring needed.**

### Implementation

```rust
// python-groggy/src/ffi/api/graph.rs

#[pymethods]
impl PyGraph {
    // === Existing explicit methods (unchanged) ===
    fn add_node(&self, ...) { ... }
    fn add_edge(&self, ...) { ... }

    // === NEW: Explicit delegated methods (macro-generated) ===

    /// Calculate degree of nodes
    ///
    /// Delegates to underlying subgraph.
    fn degree(&self, py: Python, nodes: Option<&PyAny>, full_graph: bool) -> PyResult<PyObject> {
        let subgraph = self.to_full_graph_subgraph(py)?;
        let subgraph_obj = Py::new(py, subgraph)?;
        subgraph_obj.call_method1(py, "degree", (nodes, full_graph))
    }

    /// Find connected components
    ///
    /// Delegates to underlying subgraph.
    fn connected_components(&self, py: Python) -> PyResult<PyComponentsArray> {
        let subgraph = self.to_full_graph_subgraph(py)?;
        let subgraph_obj = Py::new(py, subgraph)?;
        subgraph_obj.call_method0(py, "connected_components")
    }

    // ... (repeat for ~15-20 delegated methods)

    // === Keep __getattr__ for attributes + fallback ===
    fn __getattr__(&self, py: Python, name: String) -> PyResult<PyObject> {
        // 1. Try node/edge attributes (unchanged)
        if all_node_attrs.contains(&name) { return ...; }
        if all_edge_attrs.contains(&name) { return ...; }

        // 2. Delegate to subgraph (unchanged - fallback for unlisted methods)
        let subgraph = self.to_full_graph_subgraph(py)?;
        subgraph.getattr(py, name)
    }
}
```

**Macro to generate this:**
```rust
delegate_to_subgraph! {
    methods: [
        degree(nodes: Option<&PyAny>, full_graph: bool = false),
        in_degree(nodes: Option<&PyAny>, full_graph: bool = false),
        out_degree(nodes: Option<&PyAny>, full_graph: bool = false),
        connected_components(),
        bfs(start: NodeId, max_depth: Option<usize>),
        dfs(start: NodeId, max_depth: Option<usize>),
        shortest_path(source: NodeId, target: NodeId),
        // ... list all ~15-20 delegated methods once
    ]
}
```

### What Changes

**In Rust:**
- ✅ Add macro that generates forwarding methods
- ✅ Apply macro to PyGraph (list methods once)
- ✅ Keep existing `__getattr__` unchanged
- ✅ No changes to Rust core
- ✅ No changes to Subgraph implementation

**In Python stubs:**
- ✅ Re-run stub generator
- ✅ Graph class now shows delegated methods
- ✅ IDE autocomplete works

**Result:**
```python
g = gr.Graph()
g.degree()  # ✅ Works at runtime (same as before)
            # ✅ IDE knows about it (in .pyi stub)
            # ✅ Autocomplete works!
```

### Complexity: **2-3 weeks**

**Week 1:** Macro + apply to PyGraph
- Days 1-2: Write delegation macro
- Days 3-4: Apply to PyGraph, test
- Day 5: Update stub generator, regenerate stubs

**Week 2:** Apply to other types
- PyNeighborhoodArray → PySubgraph
- PyNodesTable → PyBaseTable
- PyEdgesTable → PyBaseTable

**Week 3:** Polish & docs
- Test all delegated methods
- Update documentation
- Performance validation

---

## Option 2: Full Trait System (REPLACE CURRENT)

### What It Is
**Complete architectural rewrite** using Rust traits in core.

### Implementation

```rust
// === 1. CREATE TRAITS IN RUST CORE ===
// src/core/traits/subgraph_operations.rs

pub trait SubgraphOperations {
    fn node_ids(&self) -> Vec<NodeId>;
    fn edge_ids(&self) -> Vec<EdgeId>;
    fn degree(&self, node_id: NodeId) -> GraphResult<usize>;
    fn neighbors(&self, node_id: NodeId) -> GraphResult<Vec<NodeId>>;
    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>>;
    fn bfs_subgraph(&self, start: NodeId) -> GraphResult<Box<dyn SubgraphOperations>>;
    // ... ~40 trait methods
}

pub trait TableOps {
    fn agg(&self, spec: &AggSpec) -> BaseTable;
    fn filter(&self, expr: &str) -> Self;
    fn group_by(&self, columns: &[&str]) -> GroupedTable;
    // ... ~15 trait methods
}

// === 2. IMPLEMENT ON CORE TYPES ===
// src/core/subgraph.rs

impl SubgraphOperations for Subgraph {
    fn node_ids(&self) -> Vec<NodeId> {
        self.nodes.iter().copied().collect()
    }

    fn degree(&self, node_id: NodeId) -> GraphResult<usize> {
        // Wire to existing algorithm implementation
        self.core_degree_algorithm(node_id)
    }

    fn connected_components(&self) -> GraphResult<Vec<Box<dyn SubgraphOperations>>> {
        // Wire to existing algorithm, return trait objects
        let components = self.core_connected_components_algorithm()?;
        Ok(components.into_iter()
            .map(|c| Box::new(c) as Box<dyn SubgraphOperations>)
            .collect())
    }

    // ... implement all 40 trait methods
}

// === 3. WIRE THROUGH FFI ===
// python-groggy/src/ffi/subgraphs/subgraph.rs

#[pymethods]
impl PySubgraph {
    fn degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        // Call trait method
        py.allow_threads(|| {
            match nodes {
                Some(n) if is_single_node => {
                    let node_id = extract_node_id(n)?;
                    self.inner.degree(node_id)  // Trait method
                        .map_err(PyErr::from)
                }
                None => {
                    // Collect degrees for all nodes
                    let degrees: HashMap<_, _> = self.inner.node_ids()
                        .iter()
                        .map(|&id| Ok((id, self.inner.degree(id)?)))  // Trait method
                        .collect::<Result<_, _>>()?;
                    Ok(degrees.into_py(py))
                }
            }
        })
    }

    fn connected_components(&self, py: Python) -> PyResult<PyComponentsArray> {
        py.allow_threads(|| {
            let components = self.inner.connected_components()?;  // Trait method

            // Convert trait objects to concrete PySubgraph
            let py_components = components.into_iter()
                .map(|trait_obj| downcast_trait_object(trait_obj))
                .collect::<Result<Vec<_>, _>>()?;

            Ok(PyComponentsArray::new(py_components))
        })
    }

    // ... wire all 40 trait methods through FFI
}

// === 4. UPDATE GRAPH TO USE TRAITS ===
// python-groggy/src/ffi/api/graph.rs

impl PyGraph {
    fn degree(&self, py: Python, nodes: Option<&PyAny>) -> PyResult<PyObject> {
        // Create subgraph that implements SubgraphOperations
        let subgraph = self.to_full_graph_subgraph(py)?;

        // Call through trait
        subgraph.degree(py, nodes)
    }

    // Still explicit methods, but now backed by traits
}
```

### What Changes

**In Rust Core:**
- ❌ Create `src/core/traits/` module structure
- ❌ Define 4-5 major traits (SubgraphOperations, TableOps, GraphOps, ArrayOps)
- ❌ Implement traits on all core types (Subgraph, Graph, NodesTable, etc.)
- ❌ Handle trait object lifetimes (`Box<dyn SubgraphOperations>`)
- ❌ Wire existing algorithms to trait methods

**In FFI:**
- ❌ Update all FFI wrappers to call trait methods
- ❌ Handle trait object → concrete type conversions
- ❌ Fix lifetime issues with trait objects across FFI boundary
- ❌ Update iterator forwarding to use traits

**In Python:**
- ✅ Re-run stub generator (same as macro approach)
- ✅ IDE autocomplete works (same as macro approach)

### Complexity: **4-5 weeks**

**Week 1-2:** Core trait infrastructure
- Define all traits (~100 trait methods total)
- Implement on core types
- Handle trait object lifetimes
- **Challenge:** Rust trait objects are complex

**Week 2-3:** FFI integration
- Wire traits through PyO3 boundary
- Type conversions for trait objects
- Fix lifetime issues
- **Challenge:** PyO3 + trait objects + lifetimes = hard

**Week 3-4:** Specialized types & conversions
- Implement missing array types
- Complete cross-type conversions
- Testing and validation

---

## Side-by-Side Comparison

| Aspect | Macro Delegation | Full Trait System |
|--------|-----------------|-------------------|
| **Keeps current pattern?** | ✅ Yes - just adds explicit methods | ❌ No - complete rewrite |
| **Changes to Rust core?** | ✅ None | ❌ Extensive (new trait module) |
| **Changes to FFI?** | ✅ Minimal (just add methods) | ❌ Extensive (all wrappers updated) |
| **Refactoring risk?** | ✅ Very low | ❌ High (touching all core types) |
| **IDE autocomplete?** | ✅ Yes | ✅ Yes |
| **Type safety?** | ✅ Compile-time (method exists) | ✅ Compile-time (trait bounds) |
| **Discoverability?** | ✅ Yes | ✅ Yes |
| **Documentation?** | ✅ Auto-generated from Rust | ✅ Auto-generated from traits |
| **Implementation time** | ✅ 2-3 weeks | ❌ 4-5 weeks |
| **Complexity** | ✅ Medium (macros) | ❌ High (traits + lifetimes) |
| **Performance** | ✅ Same as current | ⚠️ Possible regression (trait dispatch) |
| **Extensibility** | ⚠️ Add methods to macro list | ✅ Implement trait on new types |
| **"Proper" Rust?** | ⚠️ Pragmatic | ✅ Idiomatic |

---

## My Recommendation: **Macro Delegation**

### Why?

1. **Solves your problem** - Makes delegation explicit and discoverable
2. **Low risk** - Additive change, doesn't touch working code
3. **Fast** - 2-3 weeks vs 4-5 weeks
4. **Pragmatic** - Keeps pattern that works, just makes it visible
5. **Same result** - IDE autocomplete works either way

### What You Get

**Before (now):**
```python
g = gr.Graph()
g.deg  # ❌ No autocomplete
# User has to read docs to know degree() exists
```

**After (macro delegation):**
```python
g = gr.Graph()
g.deg  # ✅ IDE shows: degree(), degree_centrality()
       # ✅ Tooltips show docs
       # ✅ Type hints work
```

**After (full traits):**
```python
g = gr.Graph()
g.deg  # ✅ IDE shows: degree(), degree_centrality()
       # ✅ Tooltips show docs
       # ✅ Type hints work
# Same result as macro, but took 2 extra weeks
```

### When To Choose Full Traits Instead

Only if you're **already planning a major refactoring** and want to:
- Make codebase more "idiomatic Rust"
- Enable trait-based extensibility for plugins
- Prepare for major architectural changes
- Have 5+ weeks available

Otherwise, macro delegation gives you **90% of the benefits at 50% of the cost**.

---

## Answer to Your Questions

### 1. "Can we use macro delegation on top of our current?"

**YES!** That's exactly what it's designed for.

- ✅ Add explicit methods via macro
- ✅ Keep all existing code unchanged
- ✅ Keep `__getattr__` as fallback
- ✅ Just makes delegation visible

**No refactoring needed.**

### 2. "Otherwise we need full trait system refactoring?"

**NO!** They're **alternatives**, not sequential steps.

```
Current System (working but opaque)
         ├─> Option A: Add macro delegation (2-3 weeks, low risk)
         └─> Option B: Full trait rewrite (4-5 weeks, high risk)
```

You choose **ONE**. Macro delegation doesn't require trait system.

### 3. "What would full trait system involve?"

**Complete architectural rewrite:**
- Create traits in Rust core (`src/core/traits/`)
- Implement traits on all core types (Subgraph, Graph, Tables)
- Wire through FFI with trait object handling
- Update all ~40 methods to use trait dispatch
- Handle lifetime issues across FFI boundary
- Test everything

**It's a major project**, not a minor enhancement.

---

## What I Recommend You Do

1. **Start with macro delegation** (2-3 weeks)
   - Solves your "opaque" problem
   - Low risk, additive change
   - Makes delegation explicit and discoverable

2. **See if you need more**
   - If macro delegation is enough → stop there
   - If you want "proper" traits → refactor later

3. **Don't conflate the two**
   - Macro = pragmatic solution, works with current system
   - Traits = architectural vision, requires rewrite

You can always do traits later if macro delegation proves insufficient. But I suspect macro delegation will be plenty.

Want me to implement the delegation macro as a proof-of-concept?
