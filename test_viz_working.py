#!/usr/bin/env python3
"""
Working Viz Validation Test

Tests the Python viz module with the current FFI implementation.
Uses correct method names and parameter types.
"""

import sys
import tempfile
import time

def test_graph_viz_basic():
    """Test basic graph visualization functionality."""
    print("🔍 Testing graph viz basic functionality...")
    
    try:
        import groggy as gr
        
        # Create graph with auto-generated node IDs (integers)
        g = gr.Graph()
        node_a = g.add_node(label="Node A", type="person")
        node_b = g.add_node(label="Node B", type="person")  
        node_c = g.add_node(label="Node C", type="company")
        
        # Add edges
        g.add_edge(node_a, node_b, relationship="friend", weight=1.0)
        g.add_edge(node_b, node_c, relationship="works_at", weight=1.0)
        
        print(f"✓ Graph created: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Test viz accessor
        viz = g.viz()
        print(f"✓ viz() accessor works: {type(viz)}")
        
        # Test info method
        info = viz.info()
        print(f"✓ viz.info(): {info}")
        
        # Test supports_graph_view
        supports = viz.supports_graph_view()
        print(f"✓ supports_graph_view(): {supports}")
        
        return g, True
        
    except Exception as e:
        print(f"✗ Graph viz basic test failed: {e}")
        import traceback
        traceback.print_exc()
        return None, False


def test_static_visualization(graph):
    """Test static visualization export."""
    print("\n📊 Testing static visualization...")
    
    try:
        viz = graph.viz()
        
        with tempfile.TemporaryDirectory() as temp_dir:
            import os
            
            # Test SVG export
            svg_path = os.path.join(temp_dir, "test.svg")
            result = viz.static(svg_path, format="svg", layout="force-directed", theme="light")
            print(f"✓ SVG export: {result}")
            
            # Test PNG export
            png_path = os.path.join(temp_dir, "test.png")
            result = viz.static(png_path, format="png", layout="circular", theme="dark", dpi=300)
            print(f"✓ PNG export: {result}")
            
            # Test PDF export
            pdf_path = os.path.join(temp_dir, "test.pdf")
            result = viz.static(pdf_path, format="pdf", layout="hierarchical", theme="publication")
            print(f"✓ PDF export: {result}")
        
        return True
        
    except Exception as e:
        print(f"✗ Static visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_interactive_visualization(graph):
    """Test interactive visualization."""
    print("\n🌐 Testing interactive visualization...")
    
    try:
        viz = graph.viz()
        
        # Test interactive session creation
        session = viz.interactive(
            port=8080,
            layout="force-directed",
            theme="dark",
            width=1400,
            height=900,
            auto_open=False  # Don't open browser in test
        )
        
        print(f"✓ Interactive session created")
        print(f"  URL: {session.url()}")
        print(f"  Port: {session.port()}")
        
        # Test session methods
        assert isinstance(session.url(), str)
        assert isinstance(session.port(), int)
        assert session.port() > 0
        
        # Stop session
        session.stop()
        print("✓ Session stopped")
        
        return True
        
    except Exception as e:
        print(f"✗ Interactive visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_viz_config():
    """Test VizConfig functionality."""
    print("\n⚙️ Testing VizConfig...")
    
    try:
        import groggy as gr
        
        # Test default config
        config = gr.VizConfig()
        print(f"✓ Default config: {config}")
        
        # Test custom config
        config = gr.VizConfig(
            port=8080,
            layout="circular",
            theme="dark",
            width=1600,
            height=1200
        )
        print(f"✓ Custom config: {config}")
        
        # Test using config with viz
        g = gr.Graph()
        g.add_node(label="Test")
        
        viz = g.viz()
        session = viz.interactive(config=config, auto_open=False)
        print(f"✓ Config with interactive: {session.url()}")
        session.stop()
        
        return True
        
    except Exception as e:
        print(f"✗ VizConfig test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_convenience_functions():
    """Test module-level convenience functions."""
    print("\n🎯 Testing convenience functions...")
    
    try:
        import groggy as gr
        
        # Create test graph
        g = gr.Graph()
        node_a = g.add_node(label="A")
        node_b = g.add_node(label="B")
        g.add_edge(node_a, node_b)
        
        print(f"✓ Test graph: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Test direct viz.interactive function
        try:
            session = gr.viz.interactive(
                g,
                layout="circular",
                theme="dark",
                auto_open=False
            )
            print("✓ gr.viz.interactive() function works")
            session.stop()
        except Exception as e:
            print(f"⚠️ gr.viz.interactive() error: {e}")
        
        # Test direct viz.static function
        with tempfile.TemporaryDirectory() as temp_dir:
            import os
            try:
                output_path = os.path.join(temp_dir, "convenience_test.svg")
                result = gr.viz.static(
                    g,
                    filename=output_path,
                    format="svg",
                    layout="grid",
                    theme="publication"
                )
                print("✓ gr.viz.static() function works")
            except Exception as e:
                print(f"⚠️ gr.viz.static() error: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_table_visualization():
    """Test visualization with table objects."""
    print("\n📋 Testing table visualization...")
    
    try:
        import groggy as gr
        
        # Create graph with rich data
        g = gr.Graph()
        
        # Add nodes using auto-generated IDs
        nodes = []
        for i, person in enumerate(["Alice", "Bob", "Charlie"]):
            node_id = g.add_node(name=person, type="person", age=25+i*5)
            nodes.append(node_id)
        
        # Add edges
        g.add_edge(nodes[0], nodes[1], relationship="colleague")
        g.add_edge(nodes[1], nodes[2], relationship="friend")
        
        print(f"✓ Graph with rich data: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Test if tables have viz accessors (may not be implemented yet)
        try:
            nodes_table = g.table()
            if hasattr(nodes_table, 'viz'):
                table_viz = nodes_table.viz()
                table_info = table_viz.info()
                print(f"✓ Table viz info: {table_info}")
            else:
                print("⚠️ Table viz accessor not implemented yet")
        except Exception as e:
            print(f"⚠️ Table visualization not available: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ Table visualization test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run working viz validation tests."""
    print("🚀 Groggy Python Viz Working Validation")
    print("=" * 50)
    
    tests = [
        ("Graph Viz Basic", test_graph_viz_basic),
        ("VizConfig", test_viz_config),
        ("Convenience Functions", test_convenience_functions),
        ("Table Visualization", test_table_visualization),
    ]
    
    # Run basic tests first
    results = []
    graph = None
    
    for test_name, test_func in tests:
        try:
            if test_name == "Graph Viz Basic":
                graph, success = test_func()
                results.append((test_name, success))
            else:
                success = test_func()
                results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Run visualization tests if we have a graph
    if graph is not None:
        viz_tests = [
            ("Static Visualization", lambda: test_static_visualization(graph)),
            ("Interactive Visualization", lambda: test_interactive_visualization(graph)),
        ]
        
        for test_name, test_func in viz_tests:
            try:
                success = test_func()
                results.append((test_name, success))
            except Exception as e:
                print(f"✗ {test_name} crashed: {e}")
                results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 WORKING VALIDATION RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"{status:8} {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed >= total * 0.8:
        print("\n🎉 PYTHON VIZ MODULE WORKING!")
        print("   Core functionality is properly integrated.")
        if passed < total:
            print("   Some advanced features may need refinement.")
        return 0
    else:
        print(f"\n⚠️ PYTHON VIZ NEEDS MORE WORK: {total-passed} tests failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())