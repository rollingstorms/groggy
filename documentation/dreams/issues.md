# Issues & Future Improvements

## Bulk API Design Mismatch

**Issue**: The bulk attribute API has a design that doesn't match common conversion use cases.

**Current API Signature**:
```rust
// Expects: HashMap<AttrName, Vec<(NodeId, AttrValue)>>
graph.set_node_attrs(attrs_values: HashMap<AttrName, Vec<(NodeId, AttrValue)>>)

// Expects: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>  
graph.set_edge_attrs(attrs_values: HashMap<AttrName, Vec<(EdgeId, AttrValue)>>)
```

**Problem**: This API is designed for setting the same attribute across multiple entities, not multiple attributes on a single entity. Common conversion patterns need to set multiple different attributes on one node/edge at a time.

**Common Use Case**:
```rust
// Converting from external formats (NetworkX, GraphTable, etc.)
// Need to set multiple attributes on single entities:
for node in nodes {
    node_attrs = {
        "name": "Alice", 
        "age": 28,
        "department": "Engineering"
    }
    // Current: 3 individual calls to set_node_attr()
    // Wanted: 1 bulk call with multiple attrs for this node
}
```

**Current Workaround**: Individual `set_node_attr()` calls work fine but miss bulk optimization opportunities.

**Possible Solutions**:
1. Add complementary API: `set_node_multi_attrs(NodeId, HashMap<AttrName, AttrValue>)`
2. Restructure conversion logic to group by attribute names instead of entities
3. Accept current approach as the individual calls are fast enough

**Files Affected**:
- `src/storage/table/graph_table.rs:363` - GraphTable to_graph conversion
- `src/utils/convert.rs:256` - NetworkX conversion
- Any future conversion utilities

**Priority**: Low - Current individual calls work correctly, just not maximally optimized.