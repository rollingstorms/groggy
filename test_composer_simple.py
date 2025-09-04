#!/usr/bin/env python3
"""Simple test script for MetaGraph Composer functionality"""

import sys
sys.path.append('.')

try:
    import groggy as gr
    print("✓ Groggy imported successfully")
except ImportError as e:
    print(f"✗ Failed to import groggy: {e}")
    sys.exit(1)

def test_basic_graph_operations():
    """Test basic graph operations to ensure API works"""
    print("\n=== Testing Basic Graph Operations ===")
    
    # Create a simple graph
    g = gr.Graph(directed=False)
    print(f"Created graph: {g}")
    
    # Add nodes with proper API
    g.add_node(name="Alice", age=25, income=50000, department="engineering")
    g.add_node(name="Bob", age=30, income=60000, department="engineering")
    g.add_node(name="Charlie", age=28, income=55000, department="engineering")
    
    print(f"Added 3 nodes, graph now has {g.node_count()} nodes")
    
    # Add edges
    g.add_edge(0, 1, weight=0.8, edge_type="collaboration")
    g.add_edge(1, 2, weight=0.9, edge_type="collaboration")
    g.add_edge(0, 2, weight=0.7, edge_type="collaboration")
    
    print(f"Added 3 edges, graph now has {g.edge_count()} edges")
    
    # Create subgraph
    try:
        subgraph = g.nodes[[0, 1, 2]]
        print(f"✓ Created subgraph with {subgraph.node_count()} nodes")
        return subgraph
    except Exception as e:
        print(f"✗ Failed to create subgraph: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_collapse_method():
    """Test the collapse method on subgraph"""
    print("\n=== Testing Collapse Method ===")
    
    subgraph = test_basic_graph_operations()
    if subgraph is None:
        return False
    
    try:
        # Check if collapse method exists
        if hasattr(subgraph, 'collapse'):
            print("✓ Collapse method found on subgraph")
            
            # Try calling it with simple parameters
            result = subgraph.collapse()
            print(f"✓ Collapse method executed successfully: {result}")
            print(f"  Result type: {type(result)}")
            
            # Try with parameters
            try:
                result2 = subgraph.collapse(
                    node_aggs={"avg_age": ("mean", "age")},
                    edge_strategy="aggregate"
                )
                print(f"✓ Collapse with parameters: {result2}")
            except Exception as e:
                print(f"⚠️ Collapse with parameters failed: {e}")
                import traceback
                traceback.print_exc()
                
            return True
        else:
            print("✗ Collapse method not found on subgraph")
            print(f"Available methods: {[m for m in dir(subgraph) if not m.startswith('_')]}")
            return False
            
    except Exception as e:
        print(f"✗ Collapse method test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run tests"""
    print("Testing MetaGraph Composer Implementation")
    print("=" * 50)
    
    result = test_collapse_method()
    
    print(f"\n{'='*50}")
    if result:
        print("🎉 Basic test passed!")
        return 0
    else:
        print("❌ Basic test failed")
        return 1

if __name__ == "__main__":
    sys.exit(main())