#!/usr/bin/env python3

"""
Test PySubgraph API Consistency - verify node_ids/edge_ids properties work
"""

import sys
sys.path.insert(0, 'python-groggy/python')

import groggy

def test_subgraph_node_ids_edge_ids():
    """Test that PySubgraph has node_ids and edge_ids properties like PyGraph"""
    print("🧪 Testing PySubgraph API Consistency - node_ids/edge_ids properties...")
    
    # Create a graph with nodes and edges
    g = groggy.Graph()
    nodes = [g.add_node() for _ in range(6)]
    
    # Add some attributes
    for i, node in enumerate(nodes):
        g.set_node_attribute(node, 'value', groggy.AttrValue(i * 10))
    
    # Create some edges
    edges = []
    for i in range(len(nodes) - 1):
        edge = g.add_edge(nodes[i], nodes[i + 1])
        edges.append(edge)
        g.set_edge_attribute(edge, 'weight', groggy.AttrValue(1.0 + i * 0.5))
    
    print(f"✅ Created graph with {len(nodes)} nodes and {len(edges)} edges")
    
    # Test PyGraph node_ids and edge_ids properties (baseline)
    try:
        graph_node_ids = g.node_ids
        graph_edge_ids = g.edge_ids
        print(f"✅ PyGraph properties work: {len(graph_node_ids)} node_ids, {len(graph_edge_ids)} edge_ids")
    except Exception as e:
        print(f"❌ PyGraph properties failed: {e}")
        return False
    
    # Create a subgraph by filtering
    try:
        filtered_nodes = g.filter_nodes("value > 20")  # Should get nodes with value 30, 40, 50
        print(f"✅ Created filtered subgraph with {len(filtered_nodes)} nodes")
    except Exception as e:
        print(f"❌ Subgraph creation failed: {e}")
        return False
    
    # Test PySubgraph node_ids property
    try:
        subgraph_node_ids = filtered_nodes.node_ids
        print(f"✅ PySubgraph.node_ids works: {subgraph_node_ids}")
        
        # Verify it's a list of node IDs
        if isinstance(subgraph_node_ids, list) and len(subgraph_node_ids) > 0:
            print(f"   Contains {len(subgraph_node_ids)} node IDs")
            print(f"   First node ID: {subgraph_node_ids[0]} (type: {type(subgraph_node_ids[0])})")
        else:
            print(f"❌ node_ids returned unexpected data: {subgraph_node_ids}")
            return False
            
    except Exception as e:
        print(f"❌ PySubgraph.node_ids failed: {e}")
        return False
    
    # Test PySubgraph edge_ids property  
    try:
        subgraph_edge_ids = filtered_nodes.edge_ids
        print(f"✅ PySubgraph.edge_ids works: {subgraph_edge_ids}")
        
        # Verify it's a list of edge IDs
        if isinstance(subgraph_edge_ids, list):
            print(f"   Contains {len(subgraph_edge_ids)} edge IDs")
            if len(subgraph_edge_ids) > 0:
                print(f"   First edge ID: {subgraph_edge_ids[0]} (type: {type(subgraph_edge_ids[0])})")
        else:
            print(f"❌ edge_ids returned unexpected data: {subgraph_edge_ids}")
            return False
            
    except Exception as e:
        print(f"❌ PySubgraph.edge_ids failed: {e}")
        return False
    
    # Test API consistency: same interface as PyGraph
    try:
        # Test that both have the same property names
        graph_properties = ['node_ids', 'edge_ids', 'nodes', 'edges']
        
        for prop in graph_properties:
            if hasattr(g, prop) and hasattr(filtered_nodes, prop):
                print(f"✅ Both PyGraph and PySubgraph have '{prop}' property")
            else:
                print(f"❌ API inconsistency: missing '{prop}' property")
                return False
        
    except Exception as e:
        print(f"❌ API consistency check failed: {e}")
        return False
    
    # Test that node_ids matches what we expect from the filter
    try:
        # Get node values to verify filtering worked correctly
        expected_values = []
        for node_id in subgraph_node_ids:
            value = g.get_node_attribute(node_id, 'value')
            expected_values.append(value.value)
        
        print(f"✅ Filtered node values: {expected_values}")
        
        # All values should be > 20
        if all(v > 20 for v in expected_values):
            print(f"✅ Filter condition satisfied: all values > 20")
        else:
            print(f"❌ Filter condition not satisfied: {expected_values}")
            return False
            
    except Exception as e:
        print(f"❌ Value verification failed: {e}")
        return False
    
    return True

def test_edge_subgraph_properties():
    """Test edge_ids property with edge-focused subgraphs"""
    print("\n🧪 Testing edge subgraph properties...")
    
    # Create graph with edges
    g = groggy.Graph()
    nodes = [g.add_node() for _ in range(4)]
    
    edges = []
    weights = [1.0, 2.5, 0.5, 3.0]
    for i, weight in enumerate(weights):
        if i < len(nodes) - 1:
            edge = g.add_edge(nodes[i], nodes[i + 1])
            g.set_edge_attribute(edge, 'weight', groggy.AttrValue(weight))
            edges.append(edge)
    
    print(f"✅ Created graph with {len(edges)} edges with different weights")
    
    # Filter edges by weight
    try:
        # Filter for edges with weight > 1.5
        filtered_edges = g.filter_edges("weight > 1.5")  # Should get edges with weight 2.5, 3.0
        print(f"✅ Filtered edges: {len(filtered_edges)} edges with weight > 1.5")
        
        # Test edge_ids property on edge subgraph
        edge_ids_list = filtered_edges.edge_ids
        print(f"✅ Edge subgraph edge_ids: {edge_ids_list}")
        
        # Verify the values
        for edge_id in edge_ids_list:
            weight = g.get_edge_attribute(edge_id, 'weight')
            print(f"   Edge {edge_id}: weight = {weight.value}")
            if weight.value <= 1.5:
                print(f"❌ Edge filter failed: weight {weight.value} should be > 1.5")
                return False
        
        print(f"✅ All edge weights correctly > 1.5")
        return True
        
    except Exception as e:
        print(f"❌ Edge subgraph test failed: {e}")
        return False

if __name__ == "__main__":
    try:
        success1 = test_subgraph_node_ids_edge_ids()
        success2 = test_edge_subgraph_properties()
        
        if success1 and success2:
            print("\n🎉 PySubgraph API Consistency tests passed!")
            print("✅ node_ids and edge_ids properties work correctly")
            print("✅ API is now consistent between PyGraph and PySubgraph")
        else:
            print("\n❌ Some PySubgraph API Consistency tests failed!")
            
    except Exception as e:
        print(f"\n❌ PySubgraph API Consistency test crashed: {e}")
        import traceback
        traceback.print_exc()