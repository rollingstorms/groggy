# Meta API Discovery and Testing System

## Overview

Create a comprehensive system that discovers every method in the Groggy API, builds a graph representation of the API structure, generates dynamic tests, and provides this meta-graph as a canonical example for the library itself.

## Project Goals

1. **Complete API Discovery**: Identify every method available on all main objects
2. **Graph-Based API Representation**: Model the API as a graph with objects as nodes and methods as edges
3. **Dynamic Test Generation**: Auto-generate comprehensive tests using the API graph
4. **Meta-Example**: Use the API graph as a primary example demonstrating Groggy's capabilities
5. **Self-Documenting System**: Create a system that keeps itself up-to-date as the API evolves

## Phase 1: Complete Method Discovery

### Target Objects for Discovery
- **Core Objects**: `Graph`, `Subgraph`
- **Accessor Objects**: `NodesAccessor`, `EdgesAccessor`  
- **Storage Objects**: `Table` (BaseTable, NodesTable, EdgesTable), `Matrix`
- **Array Objects**: `BaseArray`, `NodesArray`, `EdgesArray`, `SubgraphArray`, `TableArray`, `MatrixArray`
- **Meta Objects**: `MetaNode`, `MetaEdge`
- **Component Objects**: `ComponentsArray`, `NeighborhoodResult`

### Discovery Challenges
- **Shared Traits**: Many objects implement common traits (ArrayOps, TableOps, etc.)
- **Dynamic Delegation**: Some objects use `__getattr__` to delegate methods
- **Runtime Properties**: Properties vs methods distinction
- **Generic Methods**: Template methods that work across types
- **Inheritance Chains**: Methods from parent classes/traits

### Discovery Strategy
```python
def discover_all_methods():
    """
    Comprehensive method discovery using multiple approaches:
    1. dir() introspection
    2. __dict__ examination  
    3. Method resolution order (MRO) traversal
    4. Dynamic attribute testing
    5. Trait implementation detection
    """
    discovered_methods = {}
    
    for obj_type in TARGET_OBJECTS:
        methods = []
        
        # Strategy 1: Standard introspection
        methods.extend(introspect_standard_methods(obj_type))
        
        # Strategy 2: Dynamic delegation testing
        methods.extend(test_delegation_methods(obj_type))
        
        # Strategy 3: Trait method inference
        methods.extend(infer_trait_methods(obj_type))
        
        # Strategy 4: Documentation parsing
        methods.extend(parse_documentation_methods(obj_type))
        
        discovered_methods[obj_type] = methods
    
    return discovered_methods
```

### Method Metadata Collection
For each discovered method, collect:
- **Signature**: Parameters, defaults, return type hints
- **Source**: Direct method, delegated, trait implementation
- **Return Type**: Actual return type through dynamic testing
- **Dependencies**: Required object state, parameters
- **Side Effects**: Mutating vs non-mutating
- **Documentation**: Docstring, examples

## Phase 2: API Graph Construction

### Graph Schema
```python
# Node Types
NODE_TYPES = {
    'object_type': ['Graph', 'Subgraph', 'NodesAccessor', ...],
    'method': ['add_node', 'bfs', 'table', 'head', ...]
}

# Edge Types  
EDGE_TYPES = {
    'has_method': 'object_type -> method',
    'returns': 'method -> object_type', 
    'delegates_to': 'method -> method',
    'implements_trait': 'object_type -> trait'
}
```

### Graph Construction Process
```python
def build_api_graph():
    """
    Create Groggy graph representing the API structure
    """
    api_graph = gr.Graph()
    
    # Add object type nodes
    for obj_type in discovered_methods.keys():
        api_graph.add_node(obj_type, type='object_type', category='core_api')
    
    # Add method nodes
    for obj_type, methods in discovered_methods.items():
        for method in methods:
            method_id = f"{obj_type}.{method.name}"
            api_graph.add_node(method_id, 
                             type='method',
                             name=method.name,
                             signature=method.signature,
                             return_type=method.return_type,
                             source_type=method.source)
            
            # Add has_method edge
            api_graph.add_edge(obj_type, method_id, 
                             relationship='has_method')
            
            # Add returns edge if return type is known
            if method.return_type in TARGET_OBJECTS:
                api_graph.add_edge(method_id, method.return_type,
                                 relationship='returns')
    
    return api_graph
```

### Graph Analysis Capabilities
- **Method Chains**: Find all possible delegation paths
- **Return Type Mapping**: What methods return what objects
- **Circular Dependencies**: Objects that can create each other
- **Coverage Analysis**: Which combinations are possible
- **Bottleneck Identification**: Critical connector methods

## Phase 3: Dynamic Test Generation

### Test Generation Strategy
```python
def generate_comprehensive_tests(api_graph):
    """
    Generate tests for every method using the API graph itself as test data
    """
    test_suite = TestSuite()
    
    # Strategy 1: Individual method tests
    for obj_type in api_graph.nodes.filter(type='object_type'):
        for method in api_graph.neighbors(obj_type, filter='has_method'):
            test_suite.add(generate_method_test(obj_type, method))
    
    # Strategy 2: Chain tests  
    for chain in api_graph.find_chains(max_length=5):
        test_suite.add(generate_chain_test(chain))
    
    # Strategy 3: Round-trip tests
    for cycle in api_graph.find_cycles():
        test_suite.add(generate_cycle_test(cycle))
    
    return test_suite

def generate_method_test(obj_type, method):
    """
    Generate a test for a single method using the API graph as test data
    """
    return f"""
def test_{obj_type}_{method.name}():
    # Use the API graph itself as test data
    api_graph = generate_api_graph()  
    
    # Create instance of {obj_type}
    obj = create_test_{obj_type}(api_graph)
    
    # Test method with API graph data
    result = obj.{method.name}({generate_test_params(method, api_graph)})
    
    # Validate result type and properties
    assert isinstance(result, {method.return_type})
    validate_{method.name}_result(result, api_graph)
"""
```

### Test Data Strategy
Use the API graph itself as test data:
- **Nodes**: Represent objects, methods, concepts
- **Edges**: Represent relationships, method calls, data flow
- **Attributes**: Method signatures, return types, documentation
- **Structure**: Inheritance, delegation, composition patterns

### Test Categories
1. **Smoke Tests**: Every method can be called without crashing
2. **Type Tests**: Return types match expectations  
3. **Chain Tests**: Method combinations work as expected
4. **Performance Tests**: Methods complete within reasonable time
5. **State Tests**: Methods maintain object consistency
6. **Documentation Tests**: Examples in docstrings actually work

## Phase 4: Meta-Example and Generator Integration

### API Graph as Canonical Example
```python
def api_discovery_graph():
    """
    Generator function that creates the API discovery graph
    This becomes THE canonical example of Groggy's capabilities
    """
    # Discovery process
    methods = discover_all_methods()
    
    # Graph construction  
    api_graph = build_api_graph(methods)
    
    # Enrichment
    api_graph = enrich_with_metadata(api_graph)
    api_graph = add_examples(api_graph)
    api_graph = compute_metrics(api_graph)
    
    return api_graph
```

### Integration with Generators Module
```python
# In python-groggy/python/groggy/generators.py
def groggy_graph():
    """The Groggy API represented as a Groggy graph"""
    return _generate_api_discovery_graph()

def karate_club():
    """Classic karate club graph"""
    return _generate_karate_club()

def social_network():
    """Example social network"""
    return _generate_social_network()

# New meta-examples
def delegation_patterns():
    """Graph showing all possible delegation patterns"""
    return _generate_delegation_patterns()

def method_coverage():
    """Graph showing test coverage of all methods"""  
    return _generate_method_coverage()
```

## Phase 5: Self-Updating System

### Continuous Discovery
```python
def update_api_discovery():
    """
    Re-run discovery and update all generated artifacts
    Should be run whenever the API changes
    """
    # Re-discover methods
    new_methods = discover_all_methods()
    
    # Compare with previous discovery
    changes = compare_api_versions(current_methods, new_methods)
    
    # Update graph
    api_graph = update_api_graph(changes)
    
    # Regenerate tests
    test_suite = regenerate_tests(api_graph)
    
    # Update documentation
    update_api_documentation(api_graph, changes)
    
    return changes
```

### Integration Points
- **CI/CD**: Run discovery on each commit
- **Documentation**: Auto-update API docs
- **Testing**: Auto-update test suite
- **Examples**: Keep examples current

## Implementation Plan

### Week 1: Core Discovery Engine
- [ ] Implement basic method discovery for Graph object
- [ ] Handle standard methods, properties, delegated methods
- [ ] Create metadata collection system
- [ ] Test with Graph object methods

### Week 2: Complete Object Coverage  
- [ ] Extend discovery to all target objects
- [ ] Implement trait method inference
- [ ] Handle delegation patterns
- [ ] Create method signature analysis

### Week 3: Graph Construction
- [ ] Build API graph from discovered methods
- [ ] Implement return type detection
- [ ] Add relationship modeling
- [ ] Create graph analysis tools

### Week 4: Test Generation
- [ ] Implement basic test generation
- [ ] Use API graph as test data
- [ ] Generate chain tests
- [ ] Create validation system

### Week 5: Meta-Example Integration
- [ ] Integrate with generators module
- [ ] Create compelling examples
- [ ] Add documentation
- [ ] Polish user experience

### Week 6: Self-Updating System
- [ ] Implement change detection
- [ ] Create update workflows
- [ ] Integration with CI/CD
- [ ] Final testing and refinement

## Success Metrics

1. **Coverage**: Discovery finds 100% of available methods
2. **Accuracy**: Generated tests have >95% success rate
3. **Completeness**: API graph captures all relationships
4. **Usability**: Meta-example demonstrates clear value
5. **Maintainability**: System stays current automatically

## File Structure
```
documentation/planning/
├── META_API_DISCOVERY_AND_TESTING.md        # This file
└── api_discovery_implementation_notes.md    # Implementation details

python-groggy/python/groggy/
├── generators.py                             # Add api_discovery_graph()
├── meta/
│   ├── __init__.py
│   ├── discovery.py                          # Core discovery engine
│   ├── graph_builder.py                     # API graph construction
│   ├── test_generator.py                    # Dynamic test generation
│   └── analyzer.py                          # Graph analysis tools

tests/
├── test_meta_discovery.py                   # Test the discovery system
├── generated/                               # Auto-generated tests
│   ├── test_graph_methods.py
│   ├── test_subgraph_methods.py
│   └── ...
└── meta_examples/
    ├── api_discovery_examples.py           # Examples using API graph
    └── delegation_pattern_examples.py      # Delegation examples
```

## Next Steps

1. **Start with Core Discovery**: Implement method discovery for Graph object
2. **Validate Approach**: Ensure we can find all methods reliably
3. **Build Graph**: Create first version of API graph
4. **Generate Tests**: Create basic test generation
5. **Iterate**: Refine based on results

This meta-approach will ensure we have complete API coverage, comprehensive testing, and a self-documenting system that showcases Groggy's capabilities while ensuring quality.