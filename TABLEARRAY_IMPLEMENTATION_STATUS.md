# TableArray/SubgraphArray Implementation Status

## Current Implementation (Already in Rust FFI)

### ✅ BaseTable
- `group_by(columns)` → Returns `PyTableArray`
- `group_by_agg(group_cols, agg_specs)` → Returns aggregated `BaseTable`

### ✅ NodesTable
- `group_by(columns)` → Returns `PyNodesTableArray`

### ✅ EdgesTable
- `group_by(columns)` → Returns `PyEdgesTableArray`

### ✅ TableArray (PyTableArray)
- `__len__()`, `__getitem__()`, `__iter__()`
- `agg(agg_specs: Dict[str, str])` → Returns `BaseTable`
- `filter(query: str)` → Returns `TableArray`
- `to_list()`, `collect()`

### ✅ SubgraphArray (PySubgraphArray)
- `__len__()`, `__getitem__()`, `__iter__()`
- `table()` → Returns `TableArray`
- `sample(k)` → Returns `SubgraphArray`
- `group_by(attr_name, element_type)` → Returns `SubgraphArray`
- `viz()` → Visualization accessor
- `filter(predicate)` → Returns `SubgraphArray`

## Missing Features

### ❌ TableArray Missing

1. **Column Extraction** - Critical for chain
   ```python
   table_array['column_name']  # Should return ArrayArray
   ```
   **Status**: `__getitem__` only supports integer indexing

2. **Map Operation**
   ```python
   table_array.map(lambda t: len(t))  # Returns BaseArray
   ```
   **Status**: Not implemented

### ❌ SubgraphArray Missing

1. **Attribute Extraction** - Critical for chain
   ```python
   components['degree']  # Should return ArrayArray
   ```
   **Status**: `__getitem__` only supports integer indexing

2. **Map Operation**
   ```python
   components.map(lambda sg: len(sg.nodes))
   ```
   **Status**: Not implemented

3. **Agg Operation**
   ```python
   components.agg({'degree': ['mean', 'max']})
   ```
   **Status**: Not implemented

4. **Merge Operation**
   ```python
   components.merge()  # Returns single Graph
   ```
   **Status**: Not implemented

5. **Summary/Describe**
   ```python
   components.summary()  # Returns table with stats
   ```
   **Status**: Not implemented

### ❌ ArrayArray - Completely Missing

This is the intermediate type needed for:
```python
table_array['col'].mean()
components['degree'].mean()
```

**Needs**: Full implementation from scratch

## Implementation Priority

### Phase 1: ArrayArray Foundation (Week 1)
Create `ArrayArray` class with:
- `mean()`, `sum()`, `min()`, `max()`, `std()`, `count()`
- Auto-package results with keys into BaseTable

### Phase 2: Column/Attribute Extraction (Week 1-2)
Update `__getitem__` to handle string keys:
- `PyTableArray.__getitem__(str)` → `ArrayArray`
- `PySubgraphArray.__getitem__(str)` → `ArrayArray`

### Phase 3: Map Operations (Week 2)
Add `map()` methods:
- `PyTableArray.map(func)` → `BaseArray`
- `PySubgraphArray.map(func)` → `BaseArray`

### Phase 4: SubgraphArray Extensions (Week 3)
- `agg()` - aggregate node attributes
- `merge()` - combine into single graph
- `summary()` - stats table
- `nodes_table()` / `edges_table()` - extract tables

## Rust Core Implementation Needed

### In `src/storage/array/`

Create `array_array.rs`:
```rust
pub struct ArrayArray<T> {
    arrays: Vec<BaseArray<T>>,
    keys: Option<Vec<String>>,
}

impl<T: ArrayElement> ArrayArray<T> {
    pub fn mean(&self) -> Vec<T> {
        self.arrays.iter().map(|arr| arr.mean()).collect()
    }

    pub fn to_table_with_keys(&self) -> BaseTable {
        // Package with group keys
    }
}
```

### In `src/storage/table/group_by.rs`

Already exists! Just needs:
- Return type includes group keys
- Support for column extraction

### In `src/subgraphs/`

Add attribute extraction:
```rust
impl Subgraph {
    pub fn extract_node_attribute(&self, attr: &str) -> BaseArray {
        // Get attribute from all nodes
    }
}
```

## Python Layer

Currently NO Python wrappers needed! Everything delegated through FFI.

But we should add type hints in `python-groggy/python/groggy/__init__.pyi`:
```python
class TableArray:
    def __getitem__(self, key: Union[int, str]) -> Union[BaseTable, ArrayArray]: ...
    def agg(self, spec: Dict[str, Union[str, List[str]]]) -> BaseTable: ...
    def map(self, func: Callable[[BaseTable], Any]) -> BaseArray: ...

class ArrayArray:
    def mean(self) -> Union[BaseArray, BaseTable]: ...
    def sum(self) -> Union[BaseArray, BaseTable]: ...
    # ... etc

class SubgraphArray:
    def __getitem__(self, key: Union[int, str]) -> Union[Subgraph, ArrayArray]: ...
    def map(self, func: Callable[[Subgraph], Any]) -> BaseArray: ...
    def agg(self, spec: Dict[str, Union[str, List[str]]]) -> BaseTable: ...
    def summary(self) -> BaseTable: ...
    def merge(self) -> Graph: ...
```

## Testing Plan

### Unit Tests (Rust)
```rust
#[test]
fn test_table_array_column_extraction() {
    let table = create_test_table();
    let groups = table.group_by(&["category"]);
    let col_array = groups.extract_column("value");
    assert_eq!(col_array.len(), groups.len());
}

#[test]
fn test_array_array_aggregation() {
    let arr_arr = create_test_array_array();
    let means = arr_arr.mean();
    assert_eq!(means.len(), arr_arr.len());
}
```

### Integration Tests (Python)
```python
def test_groupby_chain():
    gt = gr.from_csv(nodes_file, edges_file, ...)
    result = gt.edges.group_by('object_name')['success'].mean()

    assert isinstance(result, gr.BaseTable)
    assert 'object_name' in result.column_names()
    assert 'success' in result.column_names()

def test_subgraph_chain():
    g = gr.generators.karate_club()
    result = g.connected_components()['degree'].mean()

    assert isinstance(result, gr.BaseArray)
    assert len(result) == len(g.connected_components())
```

## Summary

**Already implemented (70%)**:
- ✅ `group_by()` on all table types
- ✅ `agg()` on TableArray with dict spec
- ✅ Basic iteration, filtering, sampling
- ✅ SubgraphArray with table(), sample(), viz()

**Missing (30%)**:
- ❌ `ArrayArray` class
- ❌ String indexing on TableArray/SubgraphArray
- ❌ `map()` operations
- ❌ SubgraphArray advanced features (merge, summary, agg)

**Estimated effort**: 2-3 weeks for complete implementation

**Critical path**: ArrayArray → Column extraction → Your use case works!
