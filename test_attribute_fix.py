#!/usr/bin/env python3
"""
Test script to verify that complex AttrValue types are properly sanitized
for realtime visualization and don't show up as [object Object].
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr
import numpy as np
import json

def test_complex_attributes():
    """Test various complex attribute types that might cause [object Object] display."""

    print("🧪 Testing complex attribute sanitization fix...")

    # Create a graph with various complex attribute types
    g = gr.Graph()

    # Add nodes with different attribute types
    node1 = g.add_node(
        label="Node with simple attrs",
        age=25,
        score=99.5,
        active=True,
        name="Alice"
    )

    node2 = g.add_node(
        label="Node with vector attrs",
        # These should be converted to readable strings
        coords=[1.2, 3.4, 5.6],  # FloatVec
        tags=["important", "user", "verified"],  # TextVec
        flags=[True, False, True, False],  # BoolVec
        counts=[10, 20, 30, 40, 50]  # IntVec
    )

    node3 = g.add_node(
        label="Node with large vectors",
        # Large vectors should show summary
        embedding=list(np.random.random(128)),  # Large FloatVec
        large_tags=[f"tag_{i}" for i in range(50)],  # Large TextVec
        big_counts=list(range(100))  # Large IntVec
    )

    # Add edges with attributes
    edge1 = g.add_edge(node1, node2, weight=0.8, type="friendship")
    edge2 = g.add_edge(node2, node3,
        weight=0.6,
        source="ml_model",  # Use simple string instead of dict
        confidence=0.95
    )

    print(f"✅ Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Test the graph visualization to see if attributes display correctly
    print("\n📊 Testing realtime visualization...")

    try:
        # Start realtime viz - this should trigger our attribute sanitization
        viz = g.graph_viz()
        viz.show(port=8080, open_browser=False)

        print("✅ Realtime viz started on port 8080")
        print("🔍 Check http://localhost:8080 and click on nodes to inspect attributes")
        print("   - Node 1: Should show simple attributes (age, score, active, name)")
        print("   - Node 2: Should show readable vector summaries (not [object Object])")
        print("   - Node 3: Should show large vector summaries with counts")
        print("   - Edges: Should show metadata as readable JSON")

        # Let's also check what the data source returns
        print("\n🔬 Inspecting data source output...")

        # Get the data source to see what's being sent
        data_source = viz.data_source
        nodes = data_source.get_graph_nodes()

        for i, node in enumerate(nodes[:3]):  # Check first 3 nodes
            print(f"\nNode {i+1} (ID: {node.id}):")
            print(f"  Label: {node.label}")
            print(f"  Attributes ({len(node.attributes)} total):")
            for key, value in node.attributes.items():
                print(f"    {key}: {value} (type: {type(value).__name__})")

        edges = data_source.get_graph_edges()
        for i, edge in enumerate(edges[:2]):  # Check first 2 edges
            print(f"\nEdge {i+1} (ID: {edge.id}):")
            print(f"  Source: {edge.source} -> Target: {edge.target}")
            print(f"  Attributes ({len(edge.attributes)} total):")
            for key, value in edge.attributes.items():
                print(f"    {key}: {value} (type: {type(value).__name__})")

        return viz

    except Exception as e:
        print(f"❌ Error starting realtime viz: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_attribute_serialization():
    """Test that attributes serialize to JSON without [object Object]."""

    print("\n🔬 Testing JSON serialization of attributes...")

    g = gr.Graph()
    node = g.add_node(
        simple_text="hello",
        simple_int=42,
        simple_float=3.14,
        simple_bool=True,
        vector_attr=[1.0, 2.0, 3.0],  # This might cause issues
        large_vector=list(range(100))  # This definitely might cause issues
    )

    # Try to create a data source and get nodes
    try:
        # Create the viz data source which should sanitize attributes
        viz = g.graph_viz()
        data_source = viz.data_source
        nodes = data_source.get_graph_nodes()

        if nodes:
            node_data = nodes[0]
            print(f"Node attributes after sanitization:")

            # Try to serialize to JSON to check for [object Object] issues
            try:
                import json
                # Convert AttrValue objects to JSON-serializable format
                serializable_attrs = {}
                for key, attr_value in node_data.attributes.items():
                    # AttrValue should now be simple types after sanitization
                    if hasattr(attr_value, '__dict__'):
                        serializable_attrs[key] = str(attr_value)
                    else:
                        serializable_attrs[key] = attr_value

                json_str = json.dumps(serializable_attrs, indent=2)
                print("✅ JSON serialization successful:")
                print(json_str)

                # Check for [object Object] in the JSON
                if "[object Object]" in json_str:
                    print("❌ Found [object Object] in JSON!")
                else:
                    print("✅ No [object Object] found in JSON")

            except Exception as e:
                print(f"❌ JSON serialization failed: {e}")

    except Exception as e:
        print(f"❌ Error testing serialization: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Testing AttrValue sanitization fix for realtime visualization")
    print("=" * 60)

    # Test 1: Complex attributes
    viz = test_complex_attributes()

    # Test 2: JSON serialization
    test_attribute_serialization()

    print("\n" + "=" * 60)
    print("🎯 Summary:")
    print("1. Check the console output above for any [object Object] or serialization errors")
    print("2. Visit http://localhost:8080 and click on nodes to inspect attributes")
    print("3. If you still see [object Object], the issue might be in the JavaScript side")
    print("4. Press Ctrl+C to stop the server when done testing")

    if viz:
        try:
            print("\n⏳ Server running... Press Ctrl+C to stop")
            input("Press Enter to stop the server...")
        except KeyboardInterrupt:
            pass
        finally:
            print("🛑 Stopping server...")