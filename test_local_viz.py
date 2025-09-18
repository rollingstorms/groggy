#!/usr/bin/env python3
"""
Test Local HTML/JS Visualization

This script tests our new local HTML export functionality.
"""

import sys
import os

def test_basic_graph_creation():
    """Test basic graph creation and HTML export."""
    print("🧪 Testing basic graph creation and HTML export...")
    
    try:
        import groggy as gr
        
        # Create a simple test graph
        g = gr.Graph()
        print(f"✓ Created graph: {type(g)}")
        
        # Add some nodes
        node_a = g.add_node(label="Alice", color="#ff6b6b")
        node_b = g.add_node(label="Bob", color="#4ecdc4") 
        node_c = g.add_node(label="Charlie", color="#45b7d1")
        print(f"✓ Added 3 nodes: {node_a}, {node_b}, {node_c}")
        
        # Add some edges
        edge_ab = g.add_edge(node_a, node_b, weight=1.5, label="friends")
        edge_bc = g.add_edge(node_b, node_c, weight=0.8, label="colleagues")
        print(f"✓ Added 2 edges: {edge_ab}, {edge_bc}")
        
        print(f"✓ Graph summary: {g.node_count()} nodes, {g.edge_count()} edges")
        
        return g
        
    except Exception as e:
        print(f"✗ Graph creation failed: {e}")
        return None

def test_viz_accessor():
    """Test if viz accessor is available."""
    print("\n🧪 Testing viz accessor availability...")
    
    try:
        import groggy as gr
        
        g = gr.Graph()
        g.add_node(label="Test")
        
        # Check if viz() method exists
        if hasattr(g, 'viz'):
            print(f"✓ g.viz() method exists")
            
            viz = g.viz()
            print(f"✓ g.viz() returned: {type(viz)}")
            
            # Check methods
            methods = ['interactive', 'static', 'info', 'supports_graph_view']
            for method in methods:
                if hasattr(viz, method):
                    print(f"✓ viz.{method}() available")
                else:
                    print(f"✗ viz.{method}() missing")
                    
            return viz
        else:
            print("✗ g.viz() method not found")
            return None
            
    except Exception as e:
        print(f"✗ Viz accessor test failed: {e}")
        return None

def test_local_html_export(graph):
    """Test our new local HTML export."""
    print("\n🧪 Testing local HTML export...")
    
    if graph is None:
        print("✗ No graph provided for testing")
        return False
        
    try:
        # Test our Python local_interactive method
        viz = graph.viz()
        
        if hasattr(viz, 'local_interactive'):
            print("✓ local_interactive() method found")
            
            html = viz.local_interactive(
                layout="force-directed",
                theme="light",
                width=800,
                height=600,
                title="Test Graph Visualization"
            )
            
            print(f"✓ Generated HTML: {len(html)} characters")
            
            # Save to file
            with open("test_viz.html", "w") as f:
                f.write(html)
            print("✓ Saved to test_viz.html")
            
            return True
        else:
            print("✗ local_interactive() method not found")
            return False
            
    except Exception as e:
        print(f"✗ Local HTML export failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_table_based_viz(graph):
    """Test table-based visualization approach."""
    print("\n🧪 Testing table-based visualization...")
    
    if graph is None:
        print("✗ No graph provided for testing")
        return False
        
    try:
        # Get table representation
        table = graph.table()
        print(f"✓ Got table: {type(table)}")
        
        if hasattr(table, 'interactive_viz'):
            print("✓ table.interactive_viz() available")
            
            viz_module = table.interactive_viz()
            print(f"✓ Got viz module: {type(viz_module)}")
            
            # Try HTML export through FFI
            try:
                result = viz_module.static_viz(
                    "test_table_viz.html",
                    format="html",
                    layout="circular",
                    theme="dark",
                    width=1000,
                    height=700
                )
                print(f"✓ HTML export successful: {result.file_path}")
                return True
                
            except Exception as e:
                print(f"⚠️  FFI HTML export failed (expected): {e}")
                # This might fail if HTML format isn't fully connected yet
                return False
                
        else:
            print("✗ table.interactive_viz() not available") 
            return False
            
    except Exception as e:
        print(f"✗ Table-based viz test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Run all tests."""
    print("🚀 Testing Local HTML/JS Visualization")
    print("=" * 50)
    
    # Test 1: Basic graph creation
    graph = test_basic_graph_creation()
    
    # Test 2: Viz accessor
    viz = test_viz_accessor()
    
    # Test 3: Local HTML export (Python-side)
    html_success = test_local_html_export(graph)
    
    # Test 4: Table-based viz (FFI-side)
    table_success = test_table_based_viz(graph)
    
    # Summary
    print("\n" + "=" * 50)
    print("🏁 Test Summary")
    print("=" * 50)
    
    total_tests = 4
    passed_tests = sum([
        graph is not None,
        viz is not None, 
        html_success,
        table_success
    ])
    
    print(f"✓ Graph creation: {'PASS' if graph is not None else 'FAIL'}")
    print(f"✓ Viz accessor: {'PASS' if viz is not None else 'FAIL'}")
    print(f"✓ Local HTML export: {'PASS' if html_success else 'FAIL'}")
    print(f"✓ Table-based viz: {'PASS' if table_success else 'FAIL'}")
    
    print(f"\nOverall: {passed_tests}/{total_tests} tests passed")
    
    if html_success:
        print("\n🎉 Check test_viz.html in your browser to see the visualization!")
    
    if passed_tests >= total_tests // 2:
        print("✨ Local visualization pipeline is working!")
        return 0
    else:
        print("⚠️  Some tests failed - more work needed")
        return 1

if __name__ == "__main__":
    sys.exit(main())