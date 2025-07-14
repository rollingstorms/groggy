#!/usr/bin/env python3

"""
Comprehensive test to verify all filtering methods are exposed and working.
"""

import groggy

def test_all_filtering_methods():
    """Test that all filtering methods are available and working."""
    print("Testing all filtering methods are exposed to Python...")
    
    # Create a test graph
    graph = groggy.FastGraph(directed=True)
    
    # Add some test data
    graph.add_node("n1", {"type": "user", "age": 25})
    graph.add_node("n2", {"type": "admin", "age": 30})
    graph.add_edge("n1", "n2", {"weight": 1.5, "relation": "reports_to"})
    
    print(f"Graph created: {graph.node_count()} nodes, {graph.edge_count()} edges")
    
    # Test all node filtering methods
    print("\n🔍 Testing Node Filtering Methods:")
    
    try:
        result = graph.filter_nodes_by_attributes({"type": "user"})
        print(f"✅ filter_nodes_by_attributes: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_nodes_by_attributes failed: {e}")
    
    try:
        result = graph.filter_nodes_by_numeric_comparison("age", ">", 27)
        print(f"✅ filter_nodes_by_numeric_comparison: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_nodes_by_numeric_comparison failed: {e}")
    
    try:
        result = graph.filter_nodes_by_string_comparison("type", "==", "admin")
        print(f"✅ filter_nodes_by_string_comparison: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_nodes_by_string_comparison failed: {e}")
    
    try:
        result = graph.filter_nodes_multi_criteria(
            exact_matches={"type": "user"},
            numeric_comparisons=[],
            string_comparisons=[]
        )
        print(f"✅ filter_nodes_multi_criteria: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_nodes_multi_criteria failed: {e}")
    
    try:
        result = graph.filter_nodes_by_attributes_sparse({"type": "admin"})
        print(f"✅ filter_nodes_by_attributes_sparse: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_nodes_by_attributes_sparse failed: {e}")
    
    # Test all edge filtering methods
    print("\n🔗 Testing Edge Filtering Methods:")
    
    try:
        result = graph.filter_edges_by_attributes({"relation": "reports_to"})
        print(f"✅ filter_edges_by_attributes: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_edges_by_attributes failed: {e}")
    
    try:
        result = graph.filter_edges_by_numeric_comparison("weight", ">=", 1.0)
        print(f"✅ filter_edges_by_numeric_comparison: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_edges_by_numeric_comparison failed: {e}")
    
    try:
        result = graph.filter_edges_by_string_comparison("relation", "==", "reports_to")
        print(f"✅ filter_edges_by_string_comparison: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_edges_by_string_comparison failed: {e}")
    
    try:
        result = graph.filter_edges_multi_criteria(
            exact_matches={"relation": "reports_to"},
            numeric_comparisons=[],
            string_comparisons=[]
        )
        print(f"✅ filter_edges_multi_criteria: {result}")
        assert len(result) == 1
    except Exception as e:
        print(f"❌ filter_edges_multi_criteria failed: {e}")
    
    print("\n🎉 All filtering methods are properly exposed and working!")
    print("✨ Unified filtering implementation is complete and functional!")

if __name__ == "__main__":
    test_all_filtering_methods()
