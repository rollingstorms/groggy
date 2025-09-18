#!/usr/bin/env python3
"""
Test script for complete subgraph visualization functionality.
Tests both interactive() and static_viz() methods.
"""

def test_subgraph_viz_complete():
    """Test complete subgraph visualization with both interactive and static export."""
    try:
        import groggy
        print("✅ Groggy import successful")
        
        # Create a test graph
        graph = groggy.Graph()
        
        # Add nodes and edges
        for i in range(10):
            graph.add_node(node_id=i, **{"label": f"Node {i}", "value": i * 10})
            
        # Add edges to create some structure
        edges = [(0, 1), (1, 2), (2, 3), (0, 4), (4, 5), (5, 6), (6, 7), (7, 8), (8, 9), (9, 0)]
        for src, dst in edges:
            graph.add_edge(source=src, target=dst, **{"weight": abs(src - dst)})
            
        print(f"✅ Created graph with nodes and edges")
        
        # First test the main graph's viz functionality
        try:
            graph_viz = graph.viz()
            print("✅ graph.viz() method works")
            
            # Test graph's interactive visualization 
            try:
                interactive_result = graph_viz.interactive()
                print(f"✅ graph.viz().interactive() works - type: {type(interactive_result)}")
            except Exception as e:
                print(f"❌ graph.viz().interactive() failed: {e}")
                
        # Test graph's static visualization
            try:
                print("Available viz methods:", [method for method in dir(graph_viz) if not method.startswith('_')])
                # Try different possible method names
                for method_name in ['static_viz', 'export', 'save', 'export_static', 'to_svg', 'static']:
                    if hasattr(graph_viz, method_name):
                        print(f"✅ Found method: {method_name}")
                        try:
                            result = getattr(graph_viz, method_name)("test_graph.svg")
                            print(f"✅ {method_name}() works - result: {result}")
                            break
                        except Exception as e:
                            print(f"❌ {method_name}() failed: {e}")
                    else:
                        print(f"⚠️  Method {method_name} not found")
            except Exception as e:
                print(f"❌ graph.viz() static methods failed: {e}")
                
        except Exception as e:
            print(f"❌ graph.viz() failed: {e}")
        
        # Now try subgraph approach using filter_nodes
        try:
            # Filter to create a subgraph-like view
            filtered_graph = graph.filter_nodes("value < 50")  # Filter nodes with value < 50
            print(f"✅ Created filtered graph")
            
            # Test filtered graph viz
            filtered_viz = filtered_graph.viz()
            print("✅ filtered_graph.viz() method works")
            
            # Test filtered interactive visualization 
            try:
                filtered_interactive = filtered_viz.interactive()
                print(f"✅ filtered.viz().interactive() works - type: {type(filtered_interactive)}")
            except Exception as e:
                print(f"❌ filtered.viz().interactive() failed: {e}")
                
            # Test filtered static visualization
            try:
                filtered_static = filtered_viz.static("test_filtered.svg")
                print(f"✅ filtered.viz().static() works - result: {filtered_static}")
                
                # Check if file was created
                import os
                if os.path.exists("test_filtered.svg"):
                    file_size = os.path.getsize("test_filtered.svg")
                    print(f"📄 SVG file created: test_filtered.svg ({file_size} bytes)")
                else:
                    print("⚠️  SVG file not found at expected location")
                    
            except Exception as e:
                print(f"❌ filtered.viz().static() failed: {e}")
                
        except Exception as e:
            print(f"❌ filter_nodes approach failed: {e}")
            
        # Try using neighborhood to create subgraph-like structures
        try:
            neighborhood_result = graph.neighborhood(0, radius=2)
            print(f"✅ Got neighborhood: {type(neighborhood_result)}")
            
            # If neighborhood returns something with viz method
            if hasattr(neighborhood_result, 'viz'):
                neighborhood_viz = neighborhood_result.viz()
                print("✅ neighborhood.viz() method works")
            else:
                print("⚠️  neighborhood doesn't have viz method")
                
        except Exception as e:
            print(f"❌ neighborhood approach failed: {e}")
            
        print("\n🎉 Graph visualization test completed!")
        
    except ImportError as e:
        print(f"❌ Import failed: {e}")
        print("💡 Make sure to run 'maturin develop' first")
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    test_subgraph_viz_complete()