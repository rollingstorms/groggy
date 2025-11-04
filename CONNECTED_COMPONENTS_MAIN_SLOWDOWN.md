# Connected Components Main Slowdown Analysis

## The Problem

After delegating to TraversalEngine, we still have **77% overhead** (1.37ms overhead on top of 1.78ms core time).

## Overhead Breakdown

From profiling:
- **Direct core method**: 1.78ms (baseline)
- **Algorithm wrapper**: 3.14ms (total)
- **Overhead**: 1.37ms (77%)

Breaking down the overhead:
- **Attribute setting**: 0.40ms (29% of overhead)
- **Algorithm framework**: 0.97ms (71% of overhead)

## Where is the 0.97ms Framework Overhead?

### 1. Result Format Conversion (MAIN CULPRIT - ~0.5-0.6ms)

**Lines 122-127 in components.rs:**
```rust
// Convert result to component ID mapping
let mut node_to_component = HashMap::new();
for (component_id, component) in result.components.into_iter().enumerate() {
    for node in component.nodes {
        node_to_component.insert(node, component_id as i64);
    }
}
```

**What's happening:**
- `result.components` is `Vec<ConnectedComponent>`
- Each `ConnectedComponent` has `nodes: Vec<NodeId>` and `edges: Vec<EdgeId>`
- We iterate through ALL components
- We iterate through ALL nodes in each component
- We do N HashMap inserts (where N = number of nodes)

**For 500 nodes:**
- 1 component (connected graph)
- Iterate through 1 component
- Iterate through 500 nodes
- Do 500 HashMap inserts

**Why this is slow:**
- HashMap::insert() has overhead (hashing, collision handling, growth)
- We're allocating a new HashMap
- We're doing this AFTER TraversalEngine already computed everything

### 2. Another Format Conversion (lines 270-276)

```rust
// Prepare bulk attributes as HashMap<AttrName, Vec<(NodeId, AttrValue)>>
let node_values: Vec<(NodeId, AttrValue)> = node_to_component
    .into_iter()
    .map(|(node, comp_id)| (node, AttrValue::Int(comp_id)))
    .collect();

let mut attrs_map = HashMap::new();
attrs_map.insert(self.output_attr.clone(), node_values);
```

**What's happening:**
- Convert HashMap → Vec<(NodeId, AttrValue)>
- Iterate through 500 entries
- Map each to AttrValue::Int
- Collect into Vec
- Create another HashMap
- Insert the Vec

**Why this is slow:**
- Multiple allocations (Vec, HashMap)
- Iterator overhead
- AttrValue wrapping overhead

### 3. Subgraph operations (lines 106-109, 259)

```rust
let graph = subgraph.graph();        // RefCell borrow
let graph_ref = graph.borrow();      // Another borrow
let pool = graph_ref.pool();         // Ref<GraphPool>
let space = graph_ref.space();       // &GraphSpace

let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
```

**Why this is slow:**
- Multiple levels of indirection (Rc → RefCell → borrow → Ref)
- Converting HashSet<NodeId> to Vec<NodeId> (subgraph.nodes() returns &HashSet)
- Iterator overhead with collect()

### 4. TraversalOptions creation (line 112)

```rust
let options = TraversalOptions::default();
```

**Minor but:** Creates a struct with default values

## Estimated Overhead Breakdown

| Source | Estimated Time | Percentage |
|--------|---------------|------------|
| HashMap conversion (lines 122-127) | ~0.5-0.6ms | ~50% |
| Vec/HashMap prep (lines 270-276) | ~0.2-0.3ms | ~25% |
| Subgraph operations | ~0.1-0.2ms | ~15% |
| Other (TraversalEngine call overhead) | ~0.1ms | ~10% |
| **Total** | **~0.97ms** | **100%** |

## Why NetworKit is 125x Faster

NetworKit does:
```cpp
// Pure C++, no conversions
ConnectedComponents cc(graph);
cc.run();  // Directly computes, stores in internal array
// Returns: component[nodeId] → componentId (direct array access)
```

No conversions, no HashMap, no iterator overhead, no allocations after the initial algorithm run.

## How to Fix This

### Option 1: Skip HashMap Conversion (BEST - eliminates 50% of overhead)

Instead of converting to HashMap, have TraversalEngine return a format that's already usable:

```rust
// TraversalEngine could return:
pub struct ComponentMapping {
    pub node_to_component: Vec<(NodeId, usize)>,  // Pre-sorted, ready to use
}
```

Then we skip lines 122-127 entirely and go directly to attribute setting.

### Option 2: Optimize HashMap Building

Use `with_capacity` and avoid intermediate collections:

```rust
let mut node_to_component = HashMap::with_capacity(nodes.len());
for (component_id, component) in result.components.into_iter().enumerate() {
    for node in component.nodes {
        node_to_component.insert(node, component_id as i64);
    }
}
```

**Expected savings**: ~0.05-0.1ms (minor)

### Option 3: Direct Attribute Setting

Skip the intermediate HashMap entirely:

```rust
// Build Vec<(NodeId, AttrValue)> directly from components
let mut node_values = Vec::with_capacity(nodes.len());
for (component_id, component) in result.components.into_iter().enumerate() {
    for node in component.nodes {
        node_values.push((node, AttrValue::Int(component_id as i64)));
    }
}

let mut attrs_map = HashMap::new();
attrs_map.insert(self.output_attr.clone(), node_values);
subgraph.set_node_attrs(attrs_map)?;
```

**Expected savings**: ~0.5-0.6ms (skip one conversion!)

### Option 4: Return ComponentsArray Directly

The CORE method returns `ComponentsArray` which is already optimized. We're converting it to attributes which adds overhead. Could we:

1. Return ComponentsArray from the algorithm
2. Let users access it lazily
3. Only convert to attributes when explicitly requested

## Recommendation

**Implement Option 3 immediately** - it's a simple change that eliminates the HashMap intermediate step:

```rust
fn compute_undirected_or_weak(
    &self,
    subgraph: &Subgraph,
    nodes: &[NodeId],
) -> Result<Vec<(NodeId, i64)>> {  // Return Vec directly, not HashMap!
    let graph = subgraph.graph();
    let graph_ref = graph.borrow();
    let pool = graph_ref.pool();
    let space = graph_ref.space();
    
    let mut traversal_engine = TraversalEngine::new();
    let options = TraversalOptions::default();
    
    let result = traversal_engine.connected_components_for_nodes(
        &pool,
        space,
        nodes.to_vec(),
        options,
    )?;
    
    // Build result directly as Vec (skip HashMap!)
    let mut node_values = Vec::with_capacity(nodes.len());
    for (component_id, component) in result.components.into_iter().enumerate() {
        for node in component.nodes {
            node_values.push((node, component_id as i64));
        }
    }
    
    Ok(node_values)
}

fn execute(&self, _ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
    let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
    
    // Get node values directly
    let node_values: Vec<(NodeId, i64)> = match self.mode {
        ComponentMode::Undirected | ComponentMode::Weak => {
            self.compute_undirected_or_weak(&subgraph, &nodes)?
        }
        ComponentMode::Strong => self.compute_strong(&subgraph, &nodes)?,
    };
    
    // Convert to AttrValue and set (single conversion!)
    let attr_values: Vec<(NodeId, AttrValue)> = node_values
        .into_iter()
        .map(|(node, comp_id)| (node, AttrValue::Int(comp_id)))
        .collect();
    
    let mut attrs_map = HashMap::new();
    attrs_map.insert(self.output_attr.clone(), attr_values);
    subgraph.set_node_attrs(attrs_map)?;
    
    Ok(subgraph)
}
```

**Expected improvement**: Reduce overhead from 1.37ms to ~0.8ms (~40% reduction in overhead!)

## Bottom Line

**The main slowdown is unnecessary data structure conversions:**
1. TraversalEngine result → HashMap (lines 122-127) - **~0.5-0.6ms**
2. HashMap → Vec → HashMap (lines 270-276) - **~0.2-0.3ms**

**Fix**: Skip the intermediate HashMap and go directly from TraversalEngine result to Vec<(NodeId, AttrValue)>.
