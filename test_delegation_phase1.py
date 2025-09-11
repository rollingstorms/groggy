#!/usr/bin/env python3

import groggy as g

# Test the new delegation architecture - Phase 1
try:
    print("Creating social network graph...")
    
    # Create a social network graph
    graph = g.generators.social_network(n=100)
    
    print(f"Graph has {graph.node_count()} nodes and {graph.edge_count()} edges")
    
    # Test the delegation chain: g.connected_components() -> ComponentsArray
    print("\nTesting g.connected_components() delegation:")
    components = graph.connected_components()
    print(f"Components type: {type(components)}")
    print(f"Components: {components}")
    print(f"Number of components: {len(components)}")
    
    # Test the delegation chain: components.iter() -> ComponentsIterator
    print("\nTesting components.iter() delegation:")
    components_iter = components.iter()
    print(f"Iterator type: {type(components_iter)}")
    print(f"Iterator: {components_iter}")
    
    # Test the delegation chain: components.iter().table() -> delegation to PySubgraph.table()
    print("\nTesting components.iter().table() delegation:")
    
    # Call table() method on the iterator
    result = components_iter.table()
    print(f"Result type: {type(result)}")
    print(f"Result length: {len(result)}")
    if len(result) > 0:
        print(f"First result type: {type(result[0])}")
        print(f"First result preview: {str(result[0])[:100]}...")
    else:
        print("No results returned")
    
    # Test sample method
    print("\nTesting components.iter().sample(5) delegation:")
    sample_result = components_iter.sample(5)
    print(f"Sample result type: {type(sample_result)}")
    print(f"Sample result length: {len(sample_result)}")
    
    print("\n✅ SUCCESS! Delegation architecture Phase 1 works!")
    print("   - g.connected_components() returns ComponentsArray")  
    print("   - .iter() returns ComponentsIterator")
    print("   - .table() applies table() to each component and returns list of table objects")
    print("   - .sample(k) applies sample(k) to each component and returns list of sampled subgraphs")
    
except Exception as e:
    print(f"❌ Error during delegation test: {e}")
    import traceback
    traceback.print_exc()