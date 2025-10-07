#!/usr/bin/env python3
"""
Test the VizAccessor to see what methods it has and test the attribute sanitization.
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr

def test_viz_accessor():
    """Test the VizAccessor methods."""
    print("🔍 Testing VizAccessor...")

    g = gr.Graph()

    # Create nodes with various attribute types
    node1 = g.add_node(
        label="Simple Node",
        age=25,
        score=99.5,
        active=True,
        department="Engineering"
    )

    node2 = g.add_node(
        label="Vector Node",
        tags=["important", "user", "verified"],  # TextVec
        coords=[1.0, 2.0, 3.0],  # FloatVec
        counts=[10, 20, 30, 40, 50],  # IntVec
        large_list=list(range(100))  # Large IntVec - should be summarized
    )

    node3 = g.add_node(
        label="Empty Node"  # Node with minimal attributes
    )

    # Add edges
    g.add_edge(node1, node2, weight=0.8, relationship="colleague")
    g.add_edge(node2, node3, weight=0.3, relationship="friend")

    print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Get the viz accessor
    viz = g.viz
    print(f"VizAccessor type: {type(viz)}")

    # Check what methods it has
    viz_methods = [method for method in dir(viz) if not method.startswith('_')]
    print(f"VizAccessor methods: {viz_methods}")

    # Look for realtime/show methods
    realtime_methods = [m for m in viz_methods if any(keyword in m.lower() for keyword in ['realtime', 'show', 'serve', 'start'])]
    print(f"Realtime methods: {realtime_methods}")

    # Try each realtime method
    for method_name in realtime_methods:
        try:
            method = getattr(viz, method_name)
            print(f"\n🧪 Testing viz.{method_name}()...")
            print(f"Method: {method}")

            if hasattr(method, '__call__'):
                # Try calling with minimal arguments
                if method_name == 'show':
                    result = method(port=8080, open_browser=False)
                    print(f"show() result: {result}")
                    return result
                elif 'realtime' in method_name:
                    result = method()
                    print(f"{method_name}() result: {result}")
                    return result
            else:
                print(f"Not callable: {method}")

        except Exception as e:
            print(f"❌ Error with {method_name}: {e}")
            import traceback
            traceback.print_exc()

    return None

def test_node_attributes():
    """Test getting node attributes correctly."""
    print("\n🔍 Testing node attribute access...")

    g = gr.Graph()
    node_id = g.add_node(
        label="Test Node",
        age=25,
        score=99.5,
        active=True,
        tags=["tag1", "tag2"],
        coords=[1.0, 2.0, 3.0]
    )

    print(f"Created node {node_id}")

    # Try different ways to get attributes
    try:
        # Method 1: Try getting all attributes
        all_attrs = g.nodes.attribute_names()
        print(f"All node attribute names: {all_attrs}")

        # Method 2: Try getting individual attributes
        for attr_name in all_attrs:
            try:
                value = g.get_node_attr(node_id, attr_name)
                print(f"  {attr_name}: {value} (type: {type(value).__name__})")
            except Exception as e:
                print(f"  {attr_name}: Error - {e}")

    except Exception as e:
        print(f"❌ Error getting attributes: {e}")

def test_directly_with_accessor():
    """Test if we can access the data source through another path."""
    print("\n🔍 Testing direct data source access...")

    g = gr.Graph()
    node1 = g.add_node(label="Test 1", value=42)
    node2 = g.add_node(label="Test 2", value=99, tags=["a", "b"])
    g.add_edge(node1, node2, weight=0.5)

    # Check if we can access the table/data source methods
    try:
        # Maybe through the nodes accessor?
        nodes_accessor = g.nodes
        print(f"Nodes accessor type: {type(nodes_accessor)}")
        nodes_methods = [m for m in dir(nodes_accessor) if not m.startswith('_')]
        print(f"Nodes accessor methods: {nodes_methods[:10]}...")

        # Try table() method if it exists
        if hasattr(nodes_accessor, 'table'):
            table = nodes_accessor.table()
            print(f"Nodes table type: {type(table)}")
            print(f"Table shape: {table.shape if hasattr(table, 'shape') else 'No shape'}")

            # Check if this table can be used as a data source
            if hasattr(table, 'get_graph_nodes'):
                nodes = table.get_graph_nodes()
                print(f"Found {len(nodes)} nodes from table:")
                for i, node in enumerate(nodes):
                    print(f"  Node {i}: {node.id}, {node.label}, {len(node.attributes)} attrs")
                    for key, value in node.attributes.items():
                        print(f"    {key}: {value} (type: {type(value).__name__})")

    except Exception as e:
        print(f"❌ Error with direct access: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    print("🔧 Testing VizAccessor and attribute handling")
    print("=" * 60)

    # Test 1: VizAccessor methods
    viz_result = test_viz_accessor()

    # Test 2: Node attributes
    test_node_attributes()

    # Test 3: Direct data source access
    test_directly_with_accessor()

    print("\n" + "=" * 60)
    if viz_result:
        print("✅ Visualization started - check http://localhost:8080")
        print("🎯 Click on nodes to see if attributes display correctly")
        print("   Look for: [object Object] vs readable attribute values")

        try:
            input("Press Enter when done testing...")
        except KeyboardInterrupt:
            pass
    else:
        print("❌ Could not start visualization to test attributes")