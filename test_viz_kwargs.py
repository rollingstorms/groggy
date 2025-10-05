#!/usr/bin/env python3
"""
Test the new VizConfig kwargs system for g.viz.show()
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr

def test_basic_viz_kwargs():
    """Test basic VizConfig kwargs parsing and application."""
    print("🧪 Testing VizConfig kwargs system...")

    # Create a simple test graph
    g = gr.Graph()

    # Add nodes with various attributes
    node1 = g.add_node(
        label="Node A",
        age=25,
        score=85.5,
        category="important"
    )

    node2 = g.add_node(
        label="Node B",
        age=30,
        score=92.3,
        category="normal"
    )

    node3 = g.add_node(
        label="Node C",
        age=35,
        score=78.9,
        category="important"
    )

    # Add edges
    edge1 = g.add_edge(node1, node2, weight=0.8, relationship="collaborates")
    edge2 = g.add_edge(node2, node3, weight=0.6, relationship="supervises")
    edge3 = g.add_edge(node1, node3, weight=0.4, relationship="mentors")

    print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

    # Test 1: Single value parameters
    print("\n🎯 Test 1: Single value parameters")
    try:
        result = g.viz.show(
            layout="honeycomb",
            node_color="red",
            node_size=10.0,
            edge_color="blue",
            edge_width=2.0,
            verbose=3
        )
        print("✅ Single value parameters parsed successfully")
        return result
    except Exception as e:
        print(f"❌ Error with single values: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_column_parameters():
    """Test column-based parameters."""
    print("\n🎯 Test 2: Column-based parameters")

    g = gr.Graph()

    # Add nodes with attributes for column mapping
    node1 = g.add_node(label="A", size_attr=5.0, color_attr="red")
    node2 = g.add_node(label="B", size_attr=8.0, color_attr="green")
    node3 = g.add_node(label="C", size_attr=12.0, color_attr="blue")

    g.add_edge(node1, node2, width_attr=1.0)
    g.add_edge(node2, node3, width_attr=3.0)

    try:
        result = g.viz.show(
            layout="grid",
            node_size="size_attr",          # Column reference
            node_color="color_attr",        # Column reference
            edge_width="width_attr",        # Column reference
            label="label",                  # Column reference
            verbose=3
        )
        print("✅ Column parameters parsed successfully")
        return result
    except Exception as e:
        print(f"❌ Error with column parameters: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_array_parameters():
    """Test array-based parameters."""
    print("\n🎯 Test 3: Array-based parameters")

    g = gr.Graph()

    # Add 3 nodes
    node1 = g.add_node(label="Node 1")
    node2 = g.add_node(label="Node 2")
    node3 = g.add_node(label="Node 3")

    # Add 2 edges
    edge1 = g.add_edge(node1, node2)
    edge2 = g.add_edge(node2, node3)

    try:
        result = g.viz.show(
            layout="circular",
            node_color=["red", "green", "blue"],        # Array for 3 nodes
            node_size=[5.0, 10.0, 15.0],               # Array for 3 nodes
            edge_width=[2.0, 4.0],                     # Array for 2 edges
            edge_color=["purple", "orange"],           # Array for 2 edges
            verbose=3
        )
        print("✅ Array parameters parsed successfully")
        return result
    except Exception as e:
        print(f"❌ Error with array parameters: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_mixed_parameters():
    """Test mixed parameter types and advanced features."""
    print("\n🎯 Test 4: Mixed and advanced parameters")

    g = gr.Graph()

    # Add nodes with mixed attributes
    node1 = g.add_node(label="Alpha", priority=1, active=True)
    node2 = g.add_node(label="Beta", priority=5, active=False)
    node3 = g.add_node(label="Gamma", priority=3, active=True)

    g.add_edge(node1, node2, strength=0.9)
    g.add_edge(node2, node3, strength=0.5)

    try:
        result = g.viz.show(
            layout="force_directed",
            # Node styling - mixed types
            node_color="priority",                     # Column mapping
            node_size=[8.0, 12.0, 10.0],             # Direct array
            node_opacity=0.8,                        # Single value

            # Edge styling
            edge_width="strength",                    # Column mapping
            edge_color="gray",                       # Single value

            # Labels
            label="label",                           # Column mapping
            label_size=12.0,                        # Single value

            # Scaling ranges
            node_size_range=(5.0, 15.0),           # Min/max scaling
            edge_width_range=(1.0, 5.0),           # Min/max scaling

            # Color settings
            color_palette=["lightblue", "orange", "lightgreen"],
            color_scale_type="categorical",

            # Interaction
            tooltip_columns=["label", "priority", "active"],
            click_behavior="select",
            hover_behavior="highlight",

            # Layout params
            charge=-200.0,
            distance=100.0,
            iterations=50,

            verbose=3
        )
        print("✅ Mixed parameters parsed successfully")
        return result
    except Exception as e:
        print(f"❌ Error with mixed parameters: {e}")
        import traceback
        traceback.print_exc()
        return None

if __name__ == "__main__":
    print("🔧 Testing VizConfig kwargs system")
    print("=" * 60)

    try:
        # Test 1: Basic single values
        result1 = test_basic_viz_kwargs()

        # Test 2: Column parameters
        result2 = test_column_parameters()

        # Test 3: Array parameters
        result3 = test_array_parameters()

        # Test 4: Mixed parameters
        result4 = test_mixed_parameters()

        print(f"\n📊 Test Results:")
        print(f"  Test 1 (Single values): {'✅ PASS' if result1 else '❌ FAIL'}")
        print(f"  Test 2 (Column params): {'✅ PASS' if result2 else '❌ FAIL'}")
        print(f"  Test 3 (Array params): {'✅ PASS' if result3 else '❌ FAIL'}")
        print(f"  Test 4 (Mixed params): {'✅ PASS' if result4 else '❌ FAIL'}")

        success_count = sum(1 for result in [result1, result2, result3, result4] if result is not None)
        print(f"\n🎯 Overall: {success_count}/4 tests passed")

        if result4:  # If the most complex test passed, show the server
            print(f"\n🌐 Server running at: {result4}")
            print("You can view the visualization with mixed parameters!")
            input("Press Enter to stop...")

    except Exception as e:
        print(f"❌ Test framework error: {e}")
        import traceback
        traceback.print_exc()