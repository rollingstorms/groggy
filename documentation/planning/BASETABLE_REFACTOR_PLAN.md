# GraphTable Paradigm Refactor Plan

## Overview

This document outlines the refactor of the current GraphTable system into a layered hierarchy that aligns with the BaseArray → NodesArray → EdgesArray foundation:
- **BaseTable**: Generic table composed of BaseArrays (like current GraphTable)
- **NodesTable**: Typed table for node data with graph-aware operations
- **EdgesTable**: Typed table for edge data with graph-aware operations  
- **GraphTable**: Composite containing NodesTable + EdgesTable with validation

**Architectural Foundation**: Tables are composed of BaseArrays, creating a unified data hierarchy:
`BaseArray` → `BaseTable` → `NodesTable`/`EdgesTable` → `GraphTable` → `Graph`

## Current Architecture Analysis

The existing `GraphTable` is a generic DataFrame-like structure:
- Collection of `GraphArray` columns with mixed types (→ becomes `BaseArray` columns)
- Supports pandas-like operations (head, tail, sort, group_by, etc.)
- Has graph-aware filtering methods but no semantic guarantees
- Can represent any tabular data, not specifically graph-structured

**Alignment Opportunity**: Current GraphTable maps perfectly to BaseTable, and BaseArrays are the natural column type.

## New Architecture Design

### 0. Table Trait (Foundation)
```rust
/// Core table operations shared by all table types
pub trait Table {
    // Basic table info
    fn nrows(&self) -> usize;
    fn ncols(&self) -> usize;
    fn column_names(&self) -> &[String];
    fn shape(&self) -> (usize, usize) { (self.nrows(), self.ncols()) }
    
    // Column access
    fn column(&self, name: &str) -> Option<&BaseArray>;
    fn column_by_index(&self, index: usize) -> Option<&BaseArray>;
    fn has_column(&self, name: &str) -> bool;
    
    // Data access patterns
    fn head(&self, n: usize) -> Self;
    fn tail(&self, n: usize) -> Self;
    fn slice(&self, start: usize, end: usize) -> Self;
    
    // DataFrame-like operations  
    fn sort_by(&self, column: &str, ascending: bool) -> GraphResult<Self> where Self: Sized;
    fn filter(&self, predicate: &str) -> GraphResult<Self> where Self: Sized;
    fn group_by(&self, columns: &[String]) -> GraphResult<Vec<Self>> where Self: Sized;
    
    // Schema and metadata
    fn schema(&self) -> TableSchema;
    fn dtypes(&self) -> HashMap<String, DataType>;
    fn metadata(&self) -> &TableMetadata;
    
    // Iteration
    fn iter_rows(&self) -> impl Iterator<Item = HashMap<String, AttrValue>>;
    fn iter_column(&self, name: &str) -> Option<impl Iterator<Item = &AttrValue>>;
    
    // Validation and quality
    fn lint(&self) -> Vec<ValidationWarning>;
    fn validate_schema(&self, spec: &SchemaSpec) -> ValidationResult;
    
    // I/O operations
    fn to_parquet(&self, path: &Path) -> GraphResult<()>;
    fn to_csv(&self, path: &Path) -> GraphResult<()>;
    fn load_parquet(path: &Path) -> GraphResult<Self>;
    fn load_csv(path: &Path) -> GraphResult<Self>;
}

/// Optional mutable operations (not all table types need these)
pub trait TableMut: Table {
    fn column_mut(&mut self, name: &str) -> Option<&mut BaseArray>;
    fn add_column(&mut self, name: String, data: BaseArray) -> GraphResult<()>;
    fn drop_column(&mut self, name: &str) -> GraphResult<()>;
    fn rename_column(&mut self, old: &str, new: &str) -> GraphResult<()>;
}
```

### 1. BaseTable (Foundation)

**Key Insight**: BaseTable is fundamentally a **collection of BaseArrays** with coordinated operations. This makes the relationship explicit:

- Each column is a `BaseArray` (columnar storage)
- Table operations coordinate across `BaseArray.iter()` calls  
- Both support `.iter()` chaining with same trait patterns
- Table is the "horizontal" composition of "vertical" arrays

```rust
// Keep most of current GraphTable functionality, but use BaseArrays
pub struct BaseTable {
    columns: Vec<BaseArray>,        // ✨ Composed of BaseArrays!
    column_names: Vec<String>,
    index: Option<BaseArray>,       // ✨ Index is also a BaseArray
    metadata: TableMetadata,
}

impl Table for BaseTable {
    // Implement all trait methods - core table operations
    // All current GraphTable methods: head, tail, sort_by, group_by, etc.
    // Basic pandas-like operations
    // No graph-specific assumptions
}

impl TableMut for BaseTable {
    // Mutable operations for BaseTable (it's the most permissive)
}

impl BaseTable {
    // BaseTable-specific methods beyond the trait
    pub fn new(columns: Vec<BaseArray>, names: Vec<String>) -> Self;
    
    // ✨ Unified chaining - table iteration delegates to array iteration
    pub fn iter(&self) -> TableIterator<Row> {
        // Coordinates BaseArray.iter() calls across columns
    }
    pub fn from_parquet(path: &Path) -> GraphResult<Self>;
    pub fn from_csv(path: &Path) -> GraphResult<Self>;
}
```

### 2. NodesTable (Typed for Nodes)
```rust
pub struct NodesTable {
    base: BaseTable,
    uid_key: String,          // Column containing unique identifiers
    node_id_mapping: Option<HashMap<String, NodeId>>, // uid -> internal node ID
}

impl Table for NodesTable {
    // Delegate all Table trait methods to underlying BaseTable
    fn nrows(&self) -> usize { self.base.nrows() }
    fn ncols(&self) -> usize { self.base.ncols() }
    fn column_names(&self) -> &[String] { self.base.column_names() }
    // ... etc. Most methods can be simple delegation
    
    // Override lint() to include node-specific validations
    fn lint(&self) -> Vec<ValidationWarning> {
        let mut warnings = self.base.lint();
        warnings.extend(self.validate_node_structure());
        warnings
    }
}

impl NodesTable {
    // Convert from BaseTable with validation
    fn from_base_table(table: BaseTable, uid_key: String) -> GraphResult<Self>;
    
    // Node-specific validation methods
    fn validate_uids(&self) -> GraphResult<()>; // Check uniqueness, no nulls
    fn validate_node_structure(&self) -> Vec<ValidationWarning>; // Node-specific lint
    
    // Graph-specific methods
    fn get_by_uid(&self, uid: &str) -> Option<&HashMap<String, AttrValue>>;
    fn iter_with_ids(&self) -> impl Iterator<Item = (NodeId, &HashMap<String, AttrValue>)>;
    
    // Access underlying BaseTable
    fn base(&self) -> &BaseTable;
    fn into_base(self) -> BaseTable;
}

// Conversion methods
impl BaseTable {
    fn to_nodes(self, uid_key: &str) -> GraphResult<NodesTable>;
}
```

### 3. EdgesTable (Typed for Edges)
```rust
pub struct EdgesTable {
    base: BaseTable,
    source_column: String,       // Column containing source node IDs
    target_column: String,       // Column containing destination node IDs
    edge_config: EdgeConfig,  // Policy settings
}

pub struct EdgeConfig {
    drop_self_loops: bool,
    allow_duplicates: bool,
    source_type: String,         // Expected type for src column
    target_type: String,         // Expected type for dst column
}

impl Table for EdgesTable {
    // Delegate most Table trait methods to underlying BaseTable
    fn nrows(&self) -> usize { self.base.nrows() }
    fn ncols(&self) -> usize { self.base.ncols() }
    fn column_names(&self) -> &[String] { self.base.column_names() }
    // ... etc.
    
    // Override lint() to include edge-specific validations
    fn lint(&self) -> Vec<ValidationWarning> {
        let mut warnings = self.base.lint();
        warnings.extend(self.validate_edge_structure());
        warnings
    }
}

impl EdgesTable {
    // Convert from BaseTable with validation
    fn from_base_table(
        table: BaseTable, 
        source: &str, 
        target: &str, 
        config: EdgeConfig
    ) -> GraphResult<Self>;
    
    // Edge-specific validation methods
    fn validate_endpoints(&self) -> GraphResult<()>;
    fn validate_edge_structure(&self) -> Vec<ValidationWarning>; // Edge-specific lint
    
    // Edge-specific methods
    fn iter_edges(&self) -> impl Iterator<Item = (String, String)>; // (src_uid, dst_uid)
    fn to_edge_list(&self) -> Vec<(String, String)>;
    fn filter_by_endpoints(&self, source_uids: &[String], target_uids: &[String]) -> EdgesTable;
    
    // Access underlying BaseTable
    fn base(&self) -> &BaseTable;
    fn into_base(self) -> BaseTable;
}

// Conversion methods
impl BaseTable {
    fn to_edges(self, source: &str, target: &str, config: EdgeConfig) -> GraphResult<EdgesTable>;
    // Convenience method with defaults
    fn to_edges_simple(self, source: &str, target: &str) -> GraphResult<EdgesTable>;
}
```

### 4. GraphTable (Composite)
```rust
pub struct GraphTable {
    nodes: NodesTable,
    edges: EdgesTable,
    validation_policy: ValidationPolicy,
}

pub struct ValidationPolicy {
    on_missing_nodes: MissingNodePolicy, // error, create_nodes, warn
    dedupe_edges: bool,
    validate_on_creation: bool,
}

pub enum MissingNodePolicy {
    Error,        // Strict: fail if edges reference unknown nodes
    CreateNodes,  // Permissive: auto-create missing nodes
    Warn,         // Log warnings but proceed
}

impl Table for GraphTable {
    // For composite GraphTable, Table operations could:
    // 1. Throw error (GraphTable doesn't represent a single table)
    // 2. Default to nodes table operations  
    // 3. Provide summary statistics across both tables
    
    fn nrows(&self) -> usize { 
        self.nodes.nrows() + self.edges.nrows() // Total rows across both tables
    }
    
    fn ncols(&self) -> usize { 
        // Could return combined unique columns, or max of the two
        self.nodes.ncols().max(self.edges.ncols())
    }
    
    fn column_names(&self) -> &[String] {
        // Note: This is tricky for composite - might need to return a computed Vec
        // For now, prioritize nodes table column names
        self.nodes.column_names()
    }
    
    fn lint(&self) -> Vec<ValidationWarning> {
        let mut warnings = self.nodes.lint();
        warnings.extend(self.edges.lint());
        warnings.extend(self.validate_graph_consistency());
        warnings
    }
    
    // Some Table operations might not make sense for GraphTable
    // and could return errors or delegate to nodes table
    fn head(&self, n: usize) -> Self {
        // For demo - could return GraphTable with head of both nodes and edges
        unimplemented!("head() for GraphTable requires design decision")
    }
    
    // ... other trait methods
}

impl GraphTable {
    // Creation methods
    fn new(nodes: NodesTable, edges: EdgesTable, policy: ValidationPolicy) -> GraphResult<Self>;
    fn from_nodes_only(nodes: NodesTable) -> Self; // Empty edges
    fn from_edges_only(edges: EdgesTable, policy: ValidationPolicy) -> GraphResult<Self>; // Infer nodes
    
    // Graph-specific validation methods  
    fn validate(&self) -> GraphResult<ValidationReport>;
    fn validate_graph_consistency(&self) -> Vec<ValidationWarning>; // Cross-table validation
    fn conform(&self, spec: SchemaSpec) -> GraphResult<(Self, ConformReport)>;
    
    // Conversion to final Graph
    fn to_graph(self, report: bool) -> GraphResult<(Graph, Option<BuildReport>)>;
    
    // Access components
    fn nodes(&self) -> &NodesTable;
    fn edges(&self) -> &EdgesTable;
    fn nodes_mut(&mut self) -> &mut NodesTable;
    fn edges_mut(&mut self) -> &mut EdgesTable;
    
    // Round-trip storage
    fn save_bundle(&self, path: &Path) -> GraphResult<()>;
    fn load_bundle(path: &Path) -> GraphResult<Self>;
}
```

## API Design Examples

### Python API Surface
```python
# --- 0) Setup ---------------------------------------------------------------
import gr  # your graph engine's Python API (sketch)

# Pretend inputs (could be CSV/Parquet paths or pandas DataFrames)
NODES_PATH = "nodes.parquet"        # columns like: ENTITY_ID, name, type, ...
EDGES_PATH = "edges.parquet"        # columns like: from, to, relation, weight, ...

# --- 1) Loose ingest → BaseTable -------------------------------------------
raw_nodes = gr.table.from_parquet(NODES_PATH)     # BaseTable (no assumptions)
raw_edges = gr.table.from_parquet(EDGES_PATH)     # BaseTable

# Light, optional cleanup on BaseTable before typing (keeps ingest frictionless)
clean_nodes = (
    raw_nodes
      .rename({"ENTITY_ID": "entity_id"})   # headers are messy? fix here
      .cast({"entity_id": "string"})        # standardize dtypes
      .filter("status == 'active' or status is null")
      .lint()                                # warnings-only; never blocks
)

clean_edges = (
    raw_edges
      .rename({"from": "src", "to": "dst"})  # normalize edge endpoints
      .cast({"src": "string", "dst": "string"})
)

# --- 2) Declare intent (typing) --------------------------------------------
# BaseTable -> NodesTable / EdgesTable
nodes = clean_nodes.to_nodes(uid_key="entity_id")     # or: gr.nodes(clean_nodes, uid_key="entity_id")
edges = clean_edges.to_edges(source="src", target="dst", drop_self_loops=True)

# --- 3) Assemble a GraphTable (strict children) ----------------------------
# Canonical case: both nodes & edges provided
gt = gr.GraphTable(nodes=nodes, edges=edges, on_missing_nodes="error")
gt.preview()  # (nice-to-have) prints small sample, schema, dtypes

# Optional: conform to a stricter schema before graphization
schema = gr.SchemaSpec(
    nodes={"id": "int32", "uid": "string"},
    edges={"src": "int32", "dst": "int32"},
    rules={"dedupe_nodes_on": "uid", "drop_self_loops": True}
)
gt, conform_report = gt.conform(spec=schema)
print(conform_report.summary())

# --- 4) Final gate: GraphTable -> Graph (strictest) ------------------------
g, build_report = gt.to_graph(report=True)
print(build_report.summary())
# Guarantees now hold:
# - contiguous int32 node ids
# - no orphan edges
# - dtype-correct, validated

# --- 5) Round-trip storage + Equivalent Access Patterns -------------------
gt_roundtrip = g.table()           # Graph -> GraphTable storage form
assert gt_roundtrip.nodes.nrows() == gt.nodes.nrows()
assert gt_roundtrip.edges.nrows() == gt.edges.nrows()

# KEY: Equivalent access patterns - these should return the SAME object
nt1 = g.nodes.table()              # NodesTable from graph nodes
nt2 = g.table().nodes              # NodesTable from GraphTable
assert nt1 is nt2                  # Same NodesTable instance

et1 = g.edges.table()              # EdgesTable from graph edges  
et2 = g.table().edges              # EdgesTable from GraphTable
assert et1 is et2                  # Same EdgesTable instance

# --- 6) Iterators & materialization on EdgesTable --------------------------
for (s, d) in et1.iter():          # zero-copy row iterator when possible
    pass

edge_list = et1.to_edge_list()     # List[Tuple[str, str]] when you need materialized edges

# --- 7) Nodes-only pipeline -------------------------------------------------
# Sometimes you only have nodes (e.g., staging attributes before edges exist)
nodes_only = gr.table("users.parquet").to_nodes(uid_key="user_id")
gt_nodes_only = gr.GraphTable(nodes=nodes_only)  # edges := empty EdgesTable
g_nodes_only = gt_nodes_only.to_graph()          # graph with |V|>0, |E|=0

# --- 8) Edges-only pipeline (infer nodes) ----------------------------------
# Sometimes you only have edges; infer the node set from src∪dst
edges_only = gr.table("follows.parquet").rename({"from":"source","to":"target"}).to_edges(source="source", target="target")
gt_edges_only = gr.GraphTable(edges=edges_only)  # infer nodes automatically
g_edges_only, report_edges_only = gt_edges_only.to_graph(report=True)
print("Inferred nodes:", report_edges_only.inferred_nodes_count)

# --- 9) Multi-GraphTable merge (domains / federated data) ------------------
# Say you have separate domain bundles that you want in one graph:
gt_users  = gr.GraphTable(
    nodes=gr.table("users.parquet").to_nodes(uid_key="user_id"),
    edges=gr.table("user_user_edges.parquet").rename({"u":"source","v":"target"}).to_edges(source="source", target="target")
)

gt_orgs   = gr.GraphTable(
    nodes=gr.table("orgs.parquet").to_nodes(uid_key="org_id"),
    edges=gr.table("org_org_edges.parquet").rename({"a":"source","b":"target"}).to_edges(source="source", target="target")
)

gt_events = gr.GraphTable(
    nodes=gr.table("events.parquet").to_nodes(uid_key="event_id"),
    edges=gr.table("user_event_edges.parquet").to_edges(source="user_id", target="event_id")
)

# Unify them into one graph. Two common strategies:
#  A) Shared UID namespace: uid_key="global_id" (all three share a column)
#  B) Namespaced domains: provide domain→uid_key mapping
g_merged, merge_report = gr.Graph([gt_users, gt_orgs, gt_events],
                                  domains={"users":"user_id","orgs":"org_id","events":"event_id"},
                                  dedupe_edges=True).to_graph(report=True)

print(merge_report.node_map_stats())   # shows collisions, domain sizes, etc.

# --- 10) Policy knobs & safety rails ---------------------------------------
# Edges reference unknown nodes?
gt_strict = gr.GraphTable(nodes=nodes, edges=edges, on_missing_nodes="error")
# Want permissive behavior instead (auto-create)?
gt_permissive = gr.GraphTable(nodes=nodes, edges=edges, on_missing_nodes="create_nodes")

# Edge de-duplication (multigraph control)
gt_dedup = gr.GraphTable(nodes=nodes, edges=edges, dedupe_edges=True)

# --- 11) Persistence (bundle) ----------------------------------------------
# A portable, reproducible snapshot of GraphTable storage
gt_roundtrip.save_graphbundle("graphbundle/")
# Contains: metadata.json (schema, uid_key, versions), nodes.parquet, edges.parquet, node_map.parquet, checksums

gt_loaded = gr.load_graphbundle("graphbundle/")
g_loaded  = gt_loaded.to_graph()

# --- 12) Minimal integrity checks you'll probably assert in tests ----------
assert g.num_nodes() == gt_roundtrip.nodes.nrows()
assert g.num_edges() == gt_roundtrip.edges.nrows()
assert gr.validate_ids(g)  # contiguous int32, no gaps
```

## Implementation Strategy

### Phase 1: BaseTable Foundation
1. **Rename Current GraphTable**: `GraphTable` → `BaseTable`
2. **Preserve All Functionality**: Keep all pandas-like operations
3. **Update Imports**: Throughout both Rust and Python codebases
4. **Test Coverage**: Ensure all existing functionality works

### Phase 2: NodesTable Implementation
1. **Create NodesTable Struct**: Wraps BaseTable with node-specific functionality
2. **Add Validation Methods**: UID uniqueness, type checking, lint warnings
3. **Implement Conversions**: `BaseTable::to_nodes()` method
4. **Python Bindings**: Expose NodesTable in Python FFI
5. **Node-Specific Methods**: `get_by_uid()`, `iter_with_ids()`, etc.

### Phase 3: EdgesTable Implementation  
1. **Create EdgesTable Struct**: Wraps BaseTable with edge-specific functionality
2. **Add EdgeConfig**: Policy settings for validation behavior
3. **Implement Conversions**: `BaseTable::to_edges()` method
4. **Edge Iteration**: Zero-copy iterators for edge traversal
5. **Python Bindings**: Expose EdgesTable in Python FFI

### Phase 4: New GraphTable (Composite)
1. **Create Composite GraphTable**: Contains NodesTable + EdgesTable
2. **Validation Policies**: Implement different strictness levels
3. **Schema Conformance**: `conform()` method with reporting
4. **Graph Conversion**: Enhanced `to_graph()` with validation
5. **Bundle Storage**: Save/load functionality

### Phase 5: Graph Integration & Equivalence
1. **Graph Methods**: Implement `g.table()`, `g.nodes.table()`, `g.edges.table()`
2. **Ensure Equivalence**: `g.nodes.table() === g.table().nodes`
3. **Caching Strategy**: Share instances between access paths
4. **Round-trip Testing**: Verify Graph ↔ GraphTable consistency

### Phase 6: Multi-GraphTable Support
1. **Merge Functionality**: Combine multiple GraphTables
2. **Domain Mapping**: Handle namespace conflicts  
3. **Federated Data**: Support for separate domain bundles
4. **Conflict Resolution**: Handle UID collisions

## File Structure Changes

### Core Library (src/)
```
src/
├── storage/                    # Unified data layer
│   ├── array/                 # ✨ Foundation layer (from BaseArray plan)
│   │   ├── base_array.rs      # BaseArray implementation  
│   │   ├── array_iterator.rs  # ArrayIterator<T> for chaining
│   │   ├── nodes_array.rs     # Typed array for nodes
│   │   ├── edges_array.rs     # Typed array for edges
│   │   └── chainable.rs       # Shared chaining traits
│   ├── table/                 # ✨ Composition layer (built on BaseArrays)
│   │   ├── table_trait.rs     # Table trait definition
│   │   ├── base_table.rs      # Renamed from GraphTable, uses BaseArrays
│   │   ├── table_iterator.rs  # TableIterator<Row> for table chaining
│   │   ├── nodes_table.rs     # NodesTable implementation
│   │   ├── edges_table.rs     # EdgesTable implementation
│   │   └── graph_table.rs     # Composite GraphTable
│   ├── validation.rs          # NEW: Validation policies and reports
│   └── schema.rs              # NEW: Schema specifications
```

### Python FFI Library (python-groggy/src/)  
```
python-groggy/src/ffi/storage/
├── array/                     # ✨ Array FFI bindings
│   ├── base_array.rs         # PyBaseArray bindings
│   ├── array_iterator.rs     # PyArrayIterator bindings  
│   ├── nodes_array.rs        # PyNodesArray bindings
│   └── edges_array.rs        # PyEdgesArray bindings
├── table/                     # ✨ Table FFI bindings (built on array bindings)
│   ├── base_table.rs         # PyBaseTable bindings
│   ├── table_iterator.rs     # PyTableIterator bindings
│   ├── nodes_table.rs        # PyNodesTable bindings
│   ├── edges_table.rs        # PyEdgesTable bindings
│   └── graph_table.rs        # PyGraphTable bindings
└── validation.rs              # Validation report bindings
```

## Migration Strategy

### Backward Compatibility
- Keep `GraphTable` as alias to `BaseTable` initially
- Keep `GraphArray` as alias to `BaseArray` initially  
- Gradual deprecation with clear migration path
- Maintain all existing method signatures

### Coordinated Array + Table Migration
**Key Insight**: These systems must be migrated together since BaseTable depends on BaseArray:

1. **Phase 0**: Rename GraphArray → BaseArray, GraphTable → BaseTable (aliases maintained)
2. **Phase 1**: Implement BaseArray chaining system 
3. **Phase 2**: Update BaseTable to use BaseArray columns explicitly
4. **Phase 3**: Add typed arrays (NodesArray, EdgesArray) and typed tables
5. **Phase 4**: Remove aliases, complete unified system

### Testing Strategy
1. **Phase-by-phase testing**: Each phase independently tested
2. **Round-trip validation**: Graph ↔ GraphTable ↔ Graph consistency
3. **Performance benchmarks**: Ensure no regression in table operations
4. **Integration tests**: Real-world data ingestion pipelines

### Documentation Updates
1. **API Documentation**: Updated with new hierarchy
2. **Migration Guide**: Step-by-step upgrade instructions  
3. **Usage Examples**: Common patterns with new API
4. **Performance Guide**: Best practices for large datasets

## Benefits of New Architecture

### 1. **Unified Data Architecture** 
- `ArrayOps<T>` trait provides consistent API across all array types
- `Table` trait provides consistent API across all table types
- **Composability**: Tables are built from Arrays, enabling deep operation reuse
- Functions can accept any data type generically: `fn analyze<T: Table>(table: &T)` or `fn process<A: ArrayOps<T>>(array: &A)`
- **Unified Chaining**: Both `.array.iter()` and `table.iter()` use same patterns
- Reduces code duplication - shared implementation across data layers

### 2. **Type Safety**
- NodesTable guarantees valid node structure
- EdgesTable ensures proper edge format
- Compile-time checks prevent misuse

### 2. **Flexible Validation**
- Configurable strictness levels
- Lint warnings for data quality
- Clear error reporting

### 3. **Performance**
- Zero-copy operations where possible
- Lazy evaluation for large datasets
- Efficient iterator patterns

### 4. **Usability**
- Clean API with clear intent
- Multiple access patterns (`g.nodes.table()` === `g.table().nodes`)
- Pandas-like familiarity

### 5. **Extensibility**
- Easy to add new table types
- Pluggable validation policies
- Schema evolution support

## Risk Mitigation

### 1. **Breaking Changes**
- **Risk**: Existing code breaks with new API
- **Mitigation**: Comprehensive backward compatibility layer

### 2. **Performance Regression**
- **Risk**: New abstractions add overhead
- **Mitigation**: Benchmarking at each phase, zero-cost abstractions

### 3. **Complexity**
- **Risk**: Four table types confuse users
- **Mitigation**: Clear documentation, guided examples, type inference

### 4. **Data Integrity**
- **Risk**: Validation gaps between table types
- **Mitigation**: Comprehensive test suite, validation at boundaries

## Implementation Checklist

### Phase 1: BaseTable Foundation & Table Trait
- [ ] Define Table trait and TableMut trait in src/storage/table_trait.rs
- [ ] Rename GraphTable to BaseTable in src/storage/
- [ ] Implement Table trait for BaseTable (and TableMut)
- [ ] Update all internal references and imports
- [ ] Create GraphTable alias for backward compatibility
- [ ] Update Python bindings
- [ ] Run full test suite to ensure no regressions

### Phase 2: NodesTable
- [ ] Create NodesTable struct with BaseTable composition
- [ ] Implement Table trait for NodesTable (delegate to BaseTable)
- [ ] Implement validation methods (validate_uids, lint)
- [ ] Add conversion method BaseTable::to_nodes()
- [ ] Create Python bindings for NodesTable
- [ ] Add comprehensive tests

### Phase 3: EdgesTable  
- [ ] Create EdgesTable struct and EdgeConfig
- [ ] Implement Table trait for EdgesTable (delegate to BaseTable)
- [ ] Implement validation and iteration methods
- [ ] Add conversion method BaseTable::to_edges()
- [ ] Create Python bindings for EdgesTable
- [ ] Add comprehensive tests

### Phase 4: Composite GraphTable
- [ ] Create new GraphTable with NodesTable + EdgesTable
- [ ] Implement Table trait for GraphTable (composite semantics)
- [ ] Implement ValidationPolicy and conformance
- [ ] Add enhanced to_graph() method
- [ ] Create bundle storage functionality
- [ ] Update Python bindings

### Phase 5: Graph Integration
- [ ] Add g.table(), g.nodes.table(), g.edges.table() methods
- [ ] Implement equivalence: g.nodes.table() === g.table().nodes
- [ ] Add caching for shared instances
- [ ] Comprehensive round-trip testing

### Phase 6: Advanced Features
- [ ] Multi-GraphTable merge functionality
- [ ] Domain mapping and conflict resolution
- [ ] Schema specification system
- [ ] Performance optimizations

### Documentation & Migration
- [ ] Update all API documentation
- [ ] Create migration guide from old to new API
- [ ] Add usage examples and best practices
- [ ] Performance benchmarking and optimization guide

---

## Architectural Alignment Summary

This refactor transforms the current generic table system into a robust, type-safe graph data management layer that maintains backward compatibility while adding powerful new capabilities for graph-specific operations.

### Unified Foundation: Arrays → Tables
```
BaseArray (columnar storage, .iter() chaining)
    ↓ composed into
BaseTable (multiple BaseArray columns, coordinated operations)
    ↓ typed as  
NodesTable / EdgesTable (semantic validation)
    ↓ combined into
GraphTable (cross-table validation + graph conversion)
    ↓ converted to
Graph (optimized graph operations)
```

### Shared Design Patterns
- **Trait-Based Operations**: Both arrays and tables use trait-based method injection
- **Progressive Type Safety**: Base → Typed → Validated → Optimized
- **Unified Chaining**: `.iter()` works consistently across all data types
- **Composable Architecture**: Higher-level constructs build on lower-level primitives
- **Policy-Driven Validation**: Configurable strictness at all levels

### Integration Benefits
- **Code Reuse**: Table operations can delegate to Array operations
- **Performance**: Zero-copy operations where possible through shared iterators
- **Maintainability**: Single trait system covers all data operations
- **Extensibility**: Easy to add new array types → new table types automatically
- **Testing**: Hierarchical testing matches hierarchical architecture

This creates a **coherent data architecture** where the array foundation directly enables table capabilities, making the relationship explicit in both code and usage patterns.