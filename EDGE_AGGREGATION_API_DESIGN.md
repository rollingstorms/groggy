# Edge Aggregation Control API Design 🔗⚡

## Overview

This document details the proposed API design for controlling edge aggregation during meta-node creation in hierarchical subgraphs. The current implementation hardcodes edge aggregation strategies, but users need fine-grained control over how edges are handled when collapsing subgraphs into meta-nodes.

## Current Implementation Analysis

### Existing `create_meta_edges` Behavior
- **Location**: `src/traits/subgraph_operations.rs:545-636`
- **Current Logic**: 
  - Automatically creates meta-edges from meta-node to external nodes
  - Counts parallel edges (`edge_count` attribute)
  - Hardcoded aggregation: `sum` for numbers, `concat` for strings
  - Always marks as `entity_type="meta"`

### Current Limitations
1. **No user control** over edge aggregation strategies
2. **No choice** in edge creation behavior (always creates to external)
3. **No control** over meta-to-meta edge handling
4. **Hardcoded attribute aggregation** (sum/concat only)

## Proposed API Design

### Enhanced `add_to_graph` Method Signature

```python
def add_to_graph(
    self, 
    agg_functions: Optional[Dict[str, Union[str, Tuple, Dict]]] = None,
    edge_config: Optional[Dict[str, Any]] = None
) -> MetaNode
```

### Edge Configuration Parameters

```python
edge_config = {
    # === EXTERNAL EDGE HANDLING ===
    "edge_to_external": "aggregate",  # "copy" | "aggregate" | "count" | "none"
    
    # === META-TO-META EDGE HANDLING ===  
    "edge_to_meta": "auto",  # "auto" | "explicit" | "none"
    
    # === EDGE ATTRIBUTE AGGREGATION ===
    "edge_aggregation": {
        "weight": "sum",           # Specific attribute strategies
        "priority": "max", 
        "relationship": "concat",
        "_default": "sum"          # Default for unlisted attributes
    },
    
    # === EDGE CREATION FILTERS ===
    "edge_filters": {
        "min_edge_count": 1,       # Only create meta-edges with >= N parallel edges
        "attribute_filters": {     # Only aggregate edges matching criteria
            "relationship": ["friend", "colleague"],
            "weight": {"min": 0.5}
        }
    },
    
    # === META-EDGE PROPERTIES ===
    "meta_edge_attributes": {
        "source_count": True,      # Add source node count attribute
        "edge_count": True,        # Add parallel edge count (default: True)
        "entity_type": True        # Mark as meta (default: True)
    }
}
```

## Detailed Parameter Specifications

### 1. External Edge Handling (`edge_to_external`)

Controls how edges from subgraph nodes to external nodes are handled:

```python
"edge_to_external": {
    "copy": {
        # Creates separate meta-edges for each original edge
        # Preserves all original edge attributes
        # Results in multiple meta-edges to same target
        "example": "meta_node --weight:1.0--> external, meta_node --weight:2.0--> external"
    },
    
    "aggregate": {  # DEFAULT
        # Creates single meta-edge with aggregated attributes
        # Uses edge_aggregation rules for attribute combination
        "example": "meta_node --weight:3.0,edge_count:2--> external"
    },
    
    "count": {
        # Creates single meta-edge with only count information
        # Loses original edge attributes, keeps only edge_count
        "example": "meta_node --edge_count:2--> external"
    },
    
    "none": {
        # No meta-edges to external nodes created
        # Complete isolation of meta-node from external graph
        "example": "meta_node (no external connections)"
    }
}
```

### 2. Meta-to-Meta Edge Handling (`edge_to_meta`)

Controls how edges between meta-nodes are handled:

```python
"edge_to_meta": {
    "auto": {  # DEFAULT
        # Automatically creates meta-to-meta edges based on subgraph connections
        # Uses same aggregation rules as external edges
        "example": "Subgraphs A and B connected → meta_A --aggregated--> meta_B"
    },
    
    "explicit": {
        # Only creates meta-to-meta edges when explicitly requested
        # Requires additional API calls or configuration
        "example": "Only user-specified meta-to-meta connections"
    },
    
    "none": {
        # No meta-to-meta edges created automatically
        # Meta-nodes remain isolated from each other
        "example": "meta_A, meta_B (no connections between them)"
    }
}
```

### 3. Edge Attribute Aggregation (`edge_aggregation`)

Granular control over how edge attributes are combined:

```python
"edge_aggregation": {
    # === NUMERICAL AGGREGATION ===
    "sum": "Add all values together",
    "mean": "Average of all values", 
    "max": "Maximum value",
    "min": "Minimum value",
    "count": "Count of non-null values",
    
    # === STRING AGGREGATION ===
    "concat": "Join with comma separator",
    "concat_unique": "Join unique values only",
    "first": "Take first non-null value",
    "last": "Take last non-null value",
    
    # === LIST AGGREGATION ===
    "flatten": "Combine all list elements",
    "union": "Set union of all elements",
    
    # === ADVANCED ===
    "custom": {
        "function": lambda values: custom_agg_logic(values),
        "default_value": None
    }
}
```

## Implementation Strategy

### 1. Python API Extension

```python
# Current Usage (unchanged)
meta_node = subgraph.add_to_graph({"salary": "sum", "age": "mean"})

# New Enhanced Usage
meta_node = subgraph.add_to_graph(
    agg_functions={"salary": "sum", "age": "mean"},
    edge_config={
        "edge_to_external": "aggregate",
        "edge_aggregation": {
            "weight": "sum",
            "relationship": "concat_unique",
            "_default": "mean"
        },
        "edge_filters": {
            "min_edge_count": 2,
            "attribute_filters": {"weight": {"min": 0.1}}
        }
    }
)
```

### 2. Rust Implementation Structure

```rust
// New struct for edge configuration
#[derive(Debug, Clone)]
pub struct EdgeAggregationConfig {
    pub edge_to_external: ExternalEdgeStrategy,
    pub edge_to_meta: MetaEdgeStrategy,
    pub edge_aggregation: HashMap<AttrName, EdgeAggregationFunction>,
    pub edge_filters: EdgeFilterConfig,
    pub meta_edge_attributes: MetaEdgeAttributeConfig,
}

// Enhanced create_meta_edges signature
fn create_meta_edges(
    &self, 
    meta_node_id: NodeId,
    config: &EdgeAggregationConfig
) -> GraphResult<()>
```

### 3. Backward Compatibility

```python
# Old API continues to work (uses default edge_config)
meta_node = subgraph.add_to_graph({"salary": "sum"})

# Equivalent to:
meta_node = subgraph.add_to_graph(
    agg_functions={"salary": "sum"},
    edge_config=DEFAULT_EDGE_CONFIG
)
```

## Example Use Cases

### 1. Social Network Analysis
```python
# Community detection with relationship aggregation
community_meta = community_subgraph.add_to_graph(
    agg_functions={"influence": "sum", "member_count": "count"},
    edge_config={
        "edge_to_external": "aggregate",
        "edge_aggregation": {
            "friendship_strength": "mean",
            "interaction_count": "sum",
            "relationship_types": "concat_unique"
        },
        "edge_filters": {
            "min_edge_count": 3,  # Only strong community connections
            "attribute_filters": {
                "friendship_strength": {"min": 0.5}
            }
        }
    }
)
```

### 2. Transportation Networks
```python
# Route optimization with capacity aggregation  
route_meta = route_subgraph.add_to_graph(
    agg_functions={"total_capacity": "sum", "stop_count": "count"},
    edge_config={
        "edge_to_external": "copy",  # Preserve individual route options
        "edge_aggregation": {
            "travel_time": "min",     # Best case travel time
            "capacity": "sum",        # Total capacity
            "cost": "mean"            # Average cost
        }
    }
)
```

### 3. Hierarchical Systems
```python
# Department/organization hierarchies
dept_meta = dept_subgraph.add_to_graph(
    agg_functions={"budget": "sum", "employee_count": "count"},
    edge_config={
        "edge_to_external": "none",   # Isolate departments initially
        "edge_to_meta": "explicit",   # Only specified inter-dept connections
        "meta_edge_attributes": {
            "source_count": True,
            "reporting_chain": True
        }
    }
)
```

## Implementation Phases

### Phase 1: Core Infrastructure
1. **EdgeAggregationConfig struct** - Rust configuration types
2. **Enhanced create_meta_edges** - Support basic strategies
3. **Python FFI extension** - `edge_config` parameter parsing

### Phase 2: Advanced Aggregation
1. **Custom aggregation functions** - User-defined edge attribute aggregation
2. **Edge filtering** - Conditional meta-edge creation
3. **Meta-to-meta handling** - Inter-meta-node edge strategies

### Phase 3: Performance & Polish
1. **Performance optimization** - Efficient aggregation algorithms
2. **Validation & error handling** - Robust configuration validation
3. **Documentation & examples** - Complete API documentation

## API Compatibility Matrix

| Feature | Current API | Phase 1 | Phase 2 | Phase 3 |
|---------|-------------|---------|---------|---------|
| Basic aggregation | ✅ | ✅ | ✅ | ✅ |
| Edge to external control | ❌ | ✅ | ✅ | ✅ |
| Custom edge aggregation | ❌ | ✅ | ✅ | ✅ |
| Edge filtering | ❌ | ❌ | ✅ | ✅ |
| Meta-to-meta control | ❌ | ❌ | ✅ | ✅ |
| Custom aggregation functions | ❌ | ❌ | ✅ | ✅ |
| Performance optimization | ❌ | ❌ | ❌ | ✅ |

## Default Configuration

```python
DEFAULT_EDGE_CONFIG = {
    "edge_to_external": "aggregate",
    "edge_to_meta": "auto", 
    "edge_aggregation": {
        "_default": "sum"  # Numbers: sum, Strings: concat
    },
    "edge_filters": {
        "min_edge_count": 1
    },
    "meta_edge_attributes": {
        "edge_count": True,
        "entity_type": True,
        "source_count": False
    }
}
```

This design provides:
- **Full backward compatibility** with existing API
- **Progressive enhancement** through optional parameters  
- **Fine-grained control** over edge aggregation strategies
- **Performance considerations** with filtering and optimization options
- **Extensible architecture** for future enhancements

The API follows the established pattern of the current `add_to_graph` method while providing comprehensive control over the previously hardcoded edge aggregation behavior.