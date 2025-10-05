# Type Stubs Status - What Works and What Doesn't

## ✅ What Works NOW (25 methods with proper types)

### Graph Methods
- ✅ `g.view() -> Subgraph` - Main entry for filtering
- ✅ `g.filter_nodes(...) -> Subgraph` - Filter by node query  
- ✅ `g.filter_edges(...) -> Subgraph` - Filter by edge query
- ✅ `g.nodes -> NodesAccessor` - Node operations
- ✅ `g.edges -> EdgesAccessor` - Edge operations
- ✅ `g.node_ids -> NumArray` - Get all node IDs
- ✅ `g.edge_ids -> NumArray` - Get all edge IDs

### Subgraph Methods  
- ✅ `sg.filter_nodes(...) -> Subgraph` - Continue filtering
- ✅ `sg.filter_edges(...) -> Subgraph` - Continue filtering
- ✅ `sg.connected_components() -> ComponentsArray`
- ✅ `sg.nodes -> NodesAccessor`
- ✅ `sg.edges -> EdgesAccessor`
- ✅ `sg.node_ids -> NumArray`
- ✅ `sg.edge_ids -> NumArray`

### Accessor Methods
- ✅ `g.nodes.all() -> Subgraph`
- ✅ `g.nodes.ids() -> NumArray`
- ✅ `g.nodes.array() -> NodesArray`
- ✅ `g.edges.all() -> Subgraph`
- ✅ `g.edges.ids() -> NumArray`
- ✅ `g.edges.array() -> EdgesArray`
- ✅ `g.edges.sources() -> NumArray`
- ✅ `g.edges.targets() -> NumArray`

### Array Methods
- ✅ `NodesArray.filter(...) -> NodesArray`
- ✅ `EdgesArray.filter(...) -> EdgesArray`
- ✅ `NumArray.filter(...) -> NumArray`

## ❌ What Still Returns `Any` (81% of methods)

**The Problem:** 831 methods still return `Any` because we can't safely call them without parameters to infer their return type.

### Examples of Methods Still Typed as `Any`:
```python
# These work at runtime but IDE doesn't know the return type:
g.add_node() -> Any  # Actually returns int
g.add_nodes(5) -> Any  # Actually returns list
g.aggregate(...) -> Any  # Actually returns AggregationResult
g.bfs(...) -> Any  # Depends on parameters
sg.degree(...) -> Any  # Could return NumArray or dict
```

## 🎯 Impact on Method Chaining

### ✅ These Chains Work:
```python
# Full type inference through the chain:
g.view().filter_nodes("attr > 5").filter_edges("weight > 10").nodes.all()
#  └─Subgraph─┘ └───Subgraph───┘ └────Subgraph────┘ └NodesAcc┘└Subgraph┘

g.nodes.all().filter_nodes(...).nodes.ids()
#  └NodesAcc┘└Subgraph┘└─Subgraph──┘└NodesAcc┘└NumArray┘
```

### ❌ These Chains Break Type Inference:
```python
# Type info lost after add_nodes (returns Any):
g.add_nodes(5).???  # Can't chain, returns list but IDE thinks Any

# Type info lost after methods with complex params:
g.aggregate("attr", "sum").???  # Returns AggregationResult but IDE thinks Any
```

## 🔧 Why This Limitation Exists

**The Challenge:** Automatic type inference requires calling methods, but:
1. Most methods need parameters we can't guess
2. Some methods have side effects we shouldn't trigger
3. Some methods fail without proper setup

**Current Approach:**
- ✅ Manual mappings for ~25 most-used methods
- ✅ Runtime testing for parameter-free methods
- ❌ Can't automatically infer methods needing parameters

## 📈 Statistics

| Category | Count | % |
|----------|-------|---|
| Total methods | 1,170 | 100% |
| Return `Any` | 831 | 71% |
| Return primitive (`int`, `str`, `bool`) | 243 | 21% |
| Return specific type (Subgraph, NumArray, etc.) | 24 | 2% |
| Special methods (`__init__`, etc.) | 72 | 6% |

**Graph class specifically:**
- Total methods: 70
- Return `Any`: 57 (81%)  
- Return specific type: 7 (10%)
- Return primitive: 6 (9%)

## 🚀 Solutions to Improve

### Option 1: Manual Mappings (Current)
**Pros:** Works now, covers common cases  
**Cons:** Labor intensive, needs maintenance

**To add more:** Edit `get_known_return_types()` in `scripts/generate_stubs.py`

### Option 2: Parse Rust Doc Comments
**Pros:** Could extract return types from Rust source  
**Cons:** Complex parsing, Rust might not have type info either

### Option 3: PyO3 text_signature Annotations
**Pros:** PyO3 supports `#[pyo3(signature = ...)]` with return types  
**Cons:** Need to add to 800+ methods in Rust code

### Option 4: Gradual Enhancement
**Recommended:** Add manual mappings as you encounter missing types:
1. Notice a method returns `Any` in IDE
2. Test it to find actual return type
3. Add to `get_known_return_types()`
4. Regenerate stubs

## 💡 Workarounds for Users

### If a method returns `Any` but you know the type:
```python
from typing import cast
import groggy as gr

g = gr.Graph()

# Method returns Any in stubs:
result = g.some_method()  # Type: Any

# Tell IDE the actual type:
result = cast(gr.Subgraph, g.some_method())  # Type: Subgraph

# Now chaining works:
result.filter_nodes(...)  # IDE knows methods!
```

### Type Comments (Python 3.5+):
```python
result = g.some_method()  # type: gr.Subgraph
result.filter_nodes(...)  # IDE now knows!
```

## 🎯 Priority Methods to Add Next

Based on common usage, these should be added to manual mappings:

**High Priority:**
- `add_node() -> int` (NodeId)
- `add_nodes(count) -> List[int]`  
- `bfs(...) -> Subgraph` (when returning subgraph)
- `dfs(...) -> Subgraph`
- `shortest_path(...) -> List[int]` (path)

**Medium Priority:**
- `aggregate(...) -> AggregationResult`
- `group_by(...) -> Dict[Any, Any]`  
- `table() -> GraphTable`

**Low Priority:**
- Methods with highly variable return types based on params

## 📝 Summary

**What You Get:**
- ✅ Tab autocomplete works for all 1,170 methods
- ✅ Docstrings appear in IDE (Shift+Tab in Jupyter)
- ✅ 25 key methods have proper types for chaining
- ✅ Core workflow chains work: `g.view().filter_nodes().nodes.all()`

**What You Don't Get:**
- ❌ Full type inference for all 1,170 methods
- ❌ Type hints for methods requiring parameters
- ❌ Automatic type updates when Rust code changes

**Bottom Line:** 
The stubs enable **discovery** (Tab completion) and **documentation** (hovering), plus **proper chaining for the 25 most common methods**. For the remaining 81%, you'll see `Any` but autocomplete still works.

---

**To improve a specific method:**
1. Add it to `get_known_return_types()` in `scripts/generate_stubs.py`
2. Run `python scripts/generate_stubs.py`
3. Restart IDE's language server
