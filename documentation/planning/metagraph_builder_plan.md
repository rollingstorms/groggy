# MetaGraph Composer Plan

## Overview

The current edge configuration system is powerful but clunky. We need a cleaner, more intuitive API for composing meta-graphs through subgraph collapse operations.

## Current Problems

1. **Complex Configuration**: `EdgeAggregationConfig` with multiple enums and HashMap parameters
2. **Scattered API**: Edge configuration separated from node aggregation
3. **Poor Discoverability**: Users need to understand complex internals
4. **Inconsistent Patterns**: Different methods for different collapse types

## Proposed Clean API

### Core Collapse Method

```python
# Main collapse method with clean, discoverable parameters
plan = subgraph.collapse(
    # Node aggregation - flexible input formats
    node_aggs={                    # Dict format
        "total_income": "sum",     # Simple: target = function on same-named source
        "avg_age": ("mean", "age"), # Tuple: target = (function, source)
        "dept_size": "count",      # Count doesn't need source
    },
    # OR list-of-tuples format
    # node_aggs=[
    #     ("total_income", "sum", "income"),  # (target, function, source)
    #     ("avg_age", "mean", "age"),
    #     ("dept_size", "count"),             # source optional for count
    # ],
    
    # Edge aggregation - simple and intuitive
    edge_aggs={                    # Optional, defaults to reasonable behavior
        "weight": "mean",          # Aggregate numeric weights
        "project": "concat",       # Concatenate text values
        # "edge_count": True,     # Auto-added if include_edge_count=True
    },
    
    # Edge strategy - small, clear enum
    edge_strategy="aggregate",     # aggregate | keep_external | drop_all | contract_all
    
    # Presets for common patterns
    preset=None,                   # "social_network" | "org_hierarchy" | "flow_network"
    
    # Control flags
    include_edge_count=True,       # Add edge_count attribute to meta-edges
    mark_entity_type=True,         # Mark meta-nodes/edges with entity_type="meta"
    entity_type="meta",            # Default entity type, overrideable
    
    # Future: Node/edge cleanup options
    # drop_original_nodes=False,   # Remove original nodes after collapse
    # drop_original_edges=False,   # Remove original edges after collapse
)

# Returns a MetaNodePlan object - not yet added to graph
meta_node = plan.add_to_graph()    # Execute the plan, return MetaNode
```

## Edge Strategy Breakdown

### 1. `edge_strategy="aggregate"` (Default)
- Combine parallel edges between subgraph and external nodes
- Apply `edge_aggs` functions to merge attributes
- Create single meta-edge per external target
- Most common use case

### 2. `edge_strategy="keep_external"`
- Copy all external edges as-is to meta-node
- Preserve original edge attributes
- Multiple edges possible to same target
- Good for preserving edge identity

### 3. `edge_strategy="drop_all"`
- Don't create any meta-edges
- Isolate the meta-node completely
- Useful for pure hierarchical grouping

### 4. `edge_strategy="contract_all"`
- Advanced: Contract edges through the subgraph
- External A -> subgraph -> External B becomes A -> B
- Subgraph becomes pure aggregation node
- Complex but powerful for flow analysis

## Return Value: MetaNodePlan

```python
class MetaNodePlan:
    """A plan for creating a meta-node (not yet executed)"""
    
    def add_to_graph(self) -> MetaNode:
        """Execute the plan, add meta-node to graph"""
        pass
    
    def preview(self) -> dict:
        """Show what the plan will create without executing"""
        return {
            "meta_node_attributes": {...},
            "meta_edges_count": 5,
            "external_targets": [node_ids...],
            "aggregated_attributes": {...}
        }
    
    # Future: Additional operations before execution
    def with_additional_agg(self, attr, func) -> 'MetaNodePlan':
        """Add more aggregations to the plan"""
        pass
    
    def with_preset(self, preset_name) -> 'MetaNodePlan':
        """Apply a preset configuration"""
        pass
```

## Preset Configurations

```python
PRESETS = {
    "social_network": {
        "edge_strategy": "aggregate",
        "edge_aggs": {"weight": "mean", "type": "concat"},
        "include_edge_count": True,
        "entity_type": "community"
    },
    
    "org_hierarchy": {
        "edge_strategy": "aggregate", 
        "edge_aggs": {"reports_to": "first", "weight": "sum"},
        "include_edge_count": False,
        "entity_type": "department"
    },
    
    "flow_network": {
        "edge_strategy": "contract_all",
        "edge_aggs": {"capacity": "min", "flow": "sum"},
        "include_edge_count": False,
        "entity_type": "junction"
    }
}
```

## Implementation Strategy

### Phase 1: Core API Design
1. Create `MetaNodePlan` class in Rust
2. Implement `collapse()` method with basic parameters
3. Support dict/tuple formats for `node_aggs`
4. Implement 4 edge strategies

### Phase 2: Python Integration
1. Add PyO3 bindings for `MetaNodePlan`
2. Create clean Python API with docstrings
3. Add parameter validation and helpful error messages
4. Test with existing examples

### Phase 3: Advanced Features
1. Implement preset system
2. Add `.preview()` method
3. Add builder pattern methods (`.with_*`)
4. Add node/edge cleanup options

### Phase 4: Documentation & Examples
1. Update all documentation to use new API
2. Create comprehensive examples for each strategy
3. Migration guide from old API
4. Performance benchmarking

## Migration Strategy

### Backward Compatibility
- Keep existing `add_to_graph()` method working
- Add deprecation warnings
- Provide automatic migration tool

### New API Adoption
```python
# Old way (still works but deprecated)
meta_node = subgraph.add_to_graph(
    {"salary": "mean"}, 
    {"edge_to_external": "aggregate"}
)

# New way (recommended)
plan = subgraph.collapse(
    node_aggs={"avg_salary": ("mean", "salary")},
    edge_strategy="aggregate"
)
meta_node = plan.add_to_graph()
```

## Advanced Use Cases

### Chaining Operations
```python
# Build complex meta-graphs step by step
dept_plan = company_subgraph.collapse(
    node_aggs={"avg_salary": ("mean", "salary")},
    edge_strategy="aggregate",
    entity_type="department"
)

division_plan = dept_plan.add_to_graph().as_subgraph().collapse(
    node_aggs={"total_budget": ("sum", "avg_salary")}, 
    edge_strategy="contract_all",
    entity_type="division"
)

division_meta = division_plan.add_to_graph()
```

### Custom Edge Processing
```python
# For complex edge transformations
plan = subgraph.collapse(
    node_aggs={"size": "count"},
    edge_strategy="aggregate",
    edge_aggs={
        "bandwidth": "sum",
        "latency": "mean", 
        "protocols": lambda values: list(set(values))  # Custom function
    }
)
```

## Benefits

1. **Intuitive**: Parameters match user mental models
2. **Flexible**: Supports simple and complex use cases
3. **Discoverable**: Clear parameter names and presets
4. **Extensible**: Easy to add new strategies and features
5. **Testable**: Plan object enables testing without side effects
6. **Performant**: Single operation instead of multiple method calls

## Questions to Resolve

1. Should `collapse()` return `MetaNodePlan` or execute immediately?
2. Best format for `node_aggs` - dict vs list vs both?
3. How to handle edge direction in `contract_all` strategy?
4. Should we support lambda functions in aggregation?
5. Node/edge cleanup - separate method or parameter?

## Next Steps

1. ✅ Design clean API (this document)
2. Create `MetaNodePlan` Rust implementation
3. Implement basic edge strategies
4. Add Python bindings
5. Test with existing hierarchical examples
6. Migrate documentation and examples