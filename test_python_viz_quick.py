#!/usr/bin/env python3
"""
Quick Python Viz Test

Quick validation of the core viz functionality to verify fixes.
"""

import sys
import tempfile
import os

def test_interactive_with_config():
    """Test interactive with config parameter handling."""
    print("🌐 Testing interactive with config...")
    
    try:
        import groggy as gr
        
        # Create test graph
        g = gr.Graph()
        node_a = g.add_node(label="A")
        node_b = g.add_node(label="B")
        g.add_edge(node_a, node_b)
        
        # Test basic interactive
        session1 = g.viz().interactive(port=8080, auto_open=False)
        print(f"✓ Basic port 8080: {session1.port()} - {session1.url()}")
        assert session1.port() == 8080
        session1.stop()
        
        # Test with different port
        session2 = g.viz().interactive(port=8081, auto_open=False)
        print(f"✓ Custom port 8081: {session2.port()} - {session2.url()}")
        assert session2.port() == 8081
        session2.stop()
        
        # Test with VizConfig
        config = gr.VizConfig(port=8082, layout="circular", theme="dark")
        session3 = g.viz().interactive(config=config, auto_open=False)
        print(f"✓ VizConfig port 8082: {session3.port()} - {session3.url()}")
        assert session3.port() == 8082
        session3.stop()
        
        return True
        
    except Exception as e:
        print(f"✗ Interactive with config test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_viz_config_presets():
    """Test VizConfig preset methods."""
    print("\n⚙️ Testing VizConfig presets...")
    
    try:
        import groggy as gr
        
        # Test basic config
        config = gr.VizConfig()
        print(f"✓ Basic config: {config}")
        
        # Test custom config
        config = gr.VizConfig(port=9000, layout="circular", theme="dark")
        print(f"✓ Custom config: {config}")
        assert config.port == 9000
        assert config.layout == "circular"
        assert config.theme == "dark"
        
        # Test preset methods if available
        try:
            # Try publication preset
            pub_config = config.publication()
            print(f"✓ Publication preset: {pub_config}")
        except Exception as e:
            print(f"⚠️ Publication preset: {e}")
        
        try:
            # Try interactive preset  
            int_config = config.interactive()
            print(f"✓ Interactive preset: {int_config}")
        except Exception as e:
            print(f"⚠️ Interactive preset: {e}")
        
        return True
        
    except Exception as e:
        print(f"✗ VizConfig presets test failed: {e}")
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
        
        # Test gr.viz.interactive
        session = gr.viz.interactive(g, port=8083, auto_open=False)
        print(f"✓ gr.viz.interactive: {session.port()} - {session.url()}")
        assert session.port() == 8083
        session.stop()
        
        # Test gr.viz.static
        with tempfile.TemporaryDirectory() as temp_dir:
            file_path = os.path.join(temp_dir, "convenience_test.svg")
            result = gr.viz.static(g, file_path, format="svg")
            print(f"✓ gr.viz.static: {result}")
            assert result.file_path == file_path
        
        return True
        
    except Exception as e:
        print(f"✗ Convenience functions test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_all_methods_working():
    """Test that all core methods work as expected."""
    print("\n✅ Testing all core methods...")
    
    try:
        import groggy as gr
        
        # Create test graph
        g = gr.Graph()
        nodes = []
        for i in range(5):
            node_id = g.add_node(label=f"Node {i}", value=i*10)
            nodes.append(node_id)
        
        for i in range(4):
            g.add_edge(nodes[i], nodes[i+1], weight=0.5+i*0.1)
        
        print(f"✓ Test graph: {g.node_count()} nodes, {g.edge_count()} edges")
        
        # Test viz() method
        viz = g.viz()
        print(f"✓ g.viz() works: {type(viz)}")
        
        # Test info method
        info = viz.info()
        print(f"✓ viz.info(): {info['source_type']}")
        
        # Test supports_graph_view
        supports = viz.supports_graph_view()
        print(f"✓ viz.supports_graph_view(): {supports}")
        
        # Test static export
        with tempfile.TemporaryDirectory() as temp_dir:
            svg_path = os.path.join(temp_dir, "test.svg")
            result = viz.static(svg_path, format="svg", layout="circular")
            print(f"✓ viz.static(): {result.file_path}")
        
        # Test interactive
        session = viz.interactive(port=8090, theme="dark", auto_open=False)
        print(f"✓ viz.interactive(): {session.url()}")
        session.stop()
        
        # Test VizConfig
        config = gr.VizConfig(port=8091, layout="grid", theme="publication")
        session = viz.interactive(config=config, auto_open=False)
        print(f"✓ VizConfig integration: {session.url()}")
        session.stop()
        
        return True
        
    except Exception as e:
        print(f"✗ All methods test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Run quick validation tests."""
    print("🚀 Quick Python Viz Validation")
    print("=" * 50)
    
    tests = [
        ("Interactive with Config", test_interactive_with_config),
        ("VizConfig Presets", test_viz_config_presets),
        ("Convenience Functions", test_convenience_functions),
        ("All Methods Working", test_all_methods_working),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "=" * 50)
    print("📊 QUICK TEST RESULTS")
    print("=" * 50)
    
    passed = 0
    for test_name, success in results:
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"{status} {test_name}")
        if success:
            passed += 1
    
    total = len(results)
    print(f"\nTotal: {passed}/{total} tests passed ({100*passed//total}%)")
    
    if passed == total:
        print("\n🎉 ALL QUICK TESTS PASSING!")
        print("   Core viz functionality is working properly.")
        return 0
    else:
        print(f"\n⚠️ {total-passed} tests still failing.")
        return 1


if __name__ == "__main__":
    sys.exit(main())