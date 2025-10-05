#!/usr/bin/env python3
"""
Debug script to check what's happening with attributes.
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr

def debug_attributes():
    """Debug what's happening with attributes."""
    print("🔍 Debugging attribute handling...")

    g = gr.Graph()

    # Add a simple node with attributes
    node1 = g.add_node(
        label="Test Node",
        age=25,
        score=99.5,
        active=True,
        department="Engineering"
    )

    node2 = g.add_node(
        label="Node with vectors",
        tags=["tag1", "tag2"],
        coords=[1.0, 2.0, 3.0]
    )

    edge1 = g.add_edge(node1, node2, weight=0.8, relationship="test")

    print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Check what methods are available
    print("\n📋 Available graph methods:")
    methods = [method for method in dir(g) if not method.startswith('_')]
    attr_methods = [m for m in methods if 'attr' in m.lower()]
    print(f"Attribute-related methods: {attr_methods}")

    # Try to check individual node/edge attributes
    print(f"\n🔍 Checking node attributes using available methods:")

    # Check if there are attribute access methods
    try:
        # Let's see what the graph object tells us
        print(f"Graph object type: {type(g)}")
        print(f"Node 1 ID: {node1}")
        print(f"Node 2 ID: {node2}")
        print(f"Edge 1 ID: {edge1}")

        # Try getting nodes collection
        if hasattr(g, 'nodes'):
            print(f"Graph has 'nodes' attribute: {type(g.nodes)}")
            nodes_methods = [method for method in dir(g.nodes) if not method.startswith('_')]
            print(f"Nodes methods: {nodes_methods[:10]}...")  # Show first 10

    except Exception as e:
        print(f"Error checking graph structure: {e}")

    # Test the data source directly
    print("\n📊 Testing data source...")

    try:
        viz = g.graph_viz()
        data_source = viz.data_source

        print(f"Data source type: {type(data_source)}")
        print(f"Data source supports graph view: {data_source.supports_graph_view()}")

        # Get nodes from data source
        nodes = data_source.get_graph_nodes()
        print(f"\nData source returned {len(nodes)} nodes:")

        for i, node in enumerate(nodes):
            print(f"\nNode {i+1}:")
            print(f"  ID: {node.id}")
            print(f"  Label: {node.label}")
            print(f"  Attributes count: {len(node.attributes)}")
            print(f"  Attributes: {dict(node.attributes)}")

            # Check the type of each attribute value
            print(f"  Attribute types:")
            for key, value in node.attributes.items():
                print(f"    {key}: {value} (type: {type(value).__name__})")

        # Get edges from data source
        edges = data_source.get_graph_edges()
        print(f"\nData source returned {len(edges)} edges:")

        for i, edge in enumerate(edges):
            print(f"\nEdge {i+1}:")
            print(f"  ID: {edge.id}")
            print(f"  Source: {edge.source} -> Target: {edge.target}")
            print(f"  Attributes count: {len(edge.attributes)}")
            print(f"  Attributes: {dict(edge.attributes)}")

        return viz

    except Exception as e:
        print(f"❌ Error with data source: {e}")
        import traceback
        traceback.print_exc()
        return None

def check_sanitization_working():
    """Check if the sanitization code is actually being called."""
    print("\n🧪 Testing if sanitization is working...")

    g = gr.Graph()

    # Add node with vector attributes (these should trigger sanitization)
    node_id = g.add_node(
        label="Vector Test",
        simple_text="hello",
        vector_floats=[1.0, 2.0, 3.0, 4.0, 5.0],  # FloatVec
        vector_strings=["a", "b", "c"],  # TextVec
        large_vector=list(range(50))  # Large IntVec - should be summarized
    )

    try:
        viz = g.graph_viz()
        data_source = viz.data_source
        nodes = data_source.get_graph_nodes()

        if nodes:
            node = nodes[0]
            print(f"Node attributes after sanitization:")
            for key, value in node.attributes.items():
                print(f"  {key}: {repr(value)} (type: {type(value).__name__})")

                # Check if large vectors are summarized
                if key == "large_vector" and isinstance(value, str):
                    if "values:" in str(value):
                        print(f"  ✅ Large vector was summarized correctly")
                    else:
                        print(f"  ❓ Large vector format: {value}")
                elif key.startswith("vector_") and isinstance(value, str):
                    print(f"  ✅ Vector attribute converted to string: {value}")
                elif key.startswith("vector_") and not isinstance(value, str):
                    print(f"  ❌ Vector attribute NOT converted: {type(value)}")

        return viz

    except Exception as e:
        print(f"❌ Error testing sanitization: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔧 Debugging attribute handling in Groggy realtime viz")
    print("=" * 70)

    viz1 = debug_attributes()
    viz2 = check_sanitization_working()

    print("\n" + "=" * 70)
    print("🎯 Key things to check:")
    print("1. Are all nodes showing their attributes?")
    print("2. Are vector attributes converted to readable strings?")
    print("3. Are there any [object Object] displays?")

    if viz1 or viz2:
        try:
            print("\n🌐 Server should be running at http://localhost:8080")
            print("   Click on nodes to see their attributes")
            print("   Press Enter to continue...")
            input()
        except KeyboardInterrupt:
            pass