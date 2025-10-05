#!/usr/bin/env python3
"""
Simple test to check if attributes are being preserved and displayed correctly.
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr

def test_basic_attributes():
    """Test basic attributes to see what's happening."""
    print("🧪 Testing basic attribute preservation...")

    g = gr.Graph()

    # Add nodes with simple attributes
    node1 = g.add_node(
        label="Simple Node 1",
        age=25,
        score=99.5,
        active=True,
        department="Engineering"
    )

    node2 = g.add_node(
        label="Simple Node 2",
        age=30,
        score=87.2,
        active=False,
        department="Marketing"
    )

    node3 = g.add_node(
        label="Node with Lists",
        tags=["tag1", "tag2", "tag3"],  # TextVec
        coords=[1.0, 2.0, 3.0],  # FloatVec
        counts=[10, 20, 30]  # IntVec
    )

    # Add edges
    edge1 = g.add_edge(node1, node2, weight=0.8, relationship="colleague")
    edge2 = g.add_edge(node2, node3, weight=0.6, relationship="friend")

    print(f"✅ Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Check what attributes look like at the graph level
    print("\n🔍 Checking node attributes directly from graph:")
    for i, node_id in enumerate([node1, node2, node3]):
        node_attrs = g.node_attributes(node_id)
        print(f"Node {i+1} (ID {node_id}) attributes: {node_attrs}")

    # Check edge attributes
    print("\n🔍 Checking edge attributes directly from graph:")
    for i, edge_id in enumerate([edge1, edge2]):
        edge_attrs = g.edge_attributes(edge_id)
        print(f"Edge {i+1} (ID {edge_id}) attributes: {edge_attrs}")

    # Now check what the data source sees
    print("\n📊 Testing data source (what realtime viz sees)...")

    try:
        viz = g.graph_viz()
        data_source = viz.data_source

        print("🔬 Data source nodes:")
        nodes = data_source.get_graph_nodes()
        for i, node in enumerate(nodes):
            print(f"  Node {i+1}: ID={node.id}, Label='{node.label}'")
            print(f"    Attributes ({len(node.attributes)}): {dict(node.attributes)}")

            # Check if any attributes are complex types
            for key, value in node.attributes.items():
                print(f"      {key}: {value} (Python type: {type(value)}, AttrValue: {type(value).__name__})")

        print("\n🔬 Data source edges:")
        edges = data_source.get_graph_edges()
        for i, edge in enumerate(edges):
            print(f"  Edge {i+1}: ID={edge.id}, {edge.source} -> {edge.target}")
            print(f"    Attributes ({len(edge.attributes)}): {dict(edge.attributes)}")

        # Check if the issue is in the sanitization
        print("\n🧪 Testing if sanitization is being called...")
        # The sanitization should happen in DataSourceRealtimeAccessor.convert_nodes()

        return viz

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_missing_attributes_issue():
    """Specifically test the issue where some nodes have no attributes."""
    print("\n🔎 Investigating missing attributes issue...")

    g = gr.Graph()

    # Create several nodes with attributes
    nodes = []
    for i in range(5):
        node_id = g.add_node(
            label=f"Test Node {i}",
            index=i,
            value=i * 10.5,
            flag=(i % 2 == 0)
        )
        nodes.append(node_id)

        # Verify attributes were added
        attrs = g.node_attributes(node_id)
        print(f"Node {i} (ID {node_id}) created with attrs: {attrs}")

    # Add edges
    for i in range(len(nodes) - 1):
        edge_id = g.add_edge(nodes[i], nodes[i+1], weight=i * 0.1)
        attrs = g.edge_attributes(edge_id)
        print(f"Edge {i} (ID {edge_id}) created with attrs: {attrs}")

    # Now check the data source
    try:
        viz = g.graph_viz()
        data_source = viz.data_source

        ds_nodes = data_source.get_graph_nodes()
        ds_edges = data_source.get_graph_edges()

        print(f"\n📊 Data source has {len(ds_nodes)} nodes, {len(ds_edges)} edges")

        # Check which nodes are missing attributes
        missing_attrs_count = 0
        for i, node in enumerate(ds_nodes):
            if len(node.attributes) == 0:
                missing_attrs_count += 1
                print(f"❌ Node {i} (ID {node.id}) has NO attributes!")
            else:
                print(f"✅ Node {i} (ID {node.id}) has {len(node.attributes)} attributes")

        if missing_attrs_count > 0:
            print(f"\n❌ PROBLEM FOUND: {missing_attrs_count}/{len(ds_nodes)} nodes missing attributes!")
        else:
            print(f"\n✅ All nodes have attributes")

        return viz

    except Exception as e:
        print(f"❌ Error: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔧 Testing attribute preservation and sanitization")
    print("=" * 60)

    # Test 1: Basic attributes
    viz1 = test_basic_attributes()

    # Test 2: Missing attributes issue
    viz2 = test_missing_attributes_issue()

    print("\n" + "=" * 60)
    if viz1 or viz2:
        print("✅ Tests completed. Check the output above for issues.")
        print("🌐 You can also test the web interface at http://localhost:8080")
    else:
        print("❌ Tests failed to create visualization")