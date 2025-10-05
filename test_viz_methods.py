#!/usr/bin/env python3
"""
Check what visualization methods are available.
"""

import sys
import os
sys.path.insert(0, '/Users/michaelroth/Documents/Code/groggy/python-groggy/python')

import groggy as gr

def check_viz_methods():
    """Check what visualization methods are available."""
    print("🔍 Checking available visualization methods...")

    g = gr.Graph()
    node1 = g.add_node(label="Test", age=25)
    node2 = g.add_node(label="Test2", age=30)
    g.add_edge(node1, node2, weight=0.5)

    # Check all methods
    all_methods = [method for method in dir(g) if not method.startswith('_')]
    print(f"Total methods: {len(all_methods)}")

    # Look for viz/visualization methods
    viz_methods = [m for m in all_methods if any(keyword in m.lower() for keyword in ['viz', 'visual', 'show', 'render', 'plot', 'display'])]
    print(f"Visualization-related methods: {viz_methods}")

    # Check specific methods that might be available
    potential_methods = ['show', 'viz', 'visualize', 'render', 'plot', 'realtime', 'streaming']
    available_methods = []
    for method in potential_methods:
        if hasattr(g, method):
            available_methods.append(method)
            print(f"✅ Found method: {method}")

    print(f"\nAvailable viz methods: {available_methods}")

    # Try the methods
    for method_name in available_methods:
        try:
            method = getattr(g, method_name)
            print(f"\n🧪 Testing {method_name}()...")
            print(f"Method type: {type(method)}")
            print(f"Method signature: {method.__doc__ if hasattr(method, '__doc__') else 'No docs'}")

            # Try calling it with minimal args
            if method_name == 'show':
                result = method()
                print(f"show() result type: {type(result)}")
                return result

        except Exception as e:
            print(f"❌ Error testing {method_name}: {e}")

    return None

def check_node_attrs_directly():
    """Check node attributes using the available methods."""
    print("\n🔍 Checking node attributes directly...")

    g = gr.Graph()
    node1 = g.add_node(
        label="Test Node",
        age=25,
        score=99.5,
        active=True,
        tags=["tag1", "tag2", "tag3"],  # This should be a vector
        coords=[1.0, 2.0, 3.0]  # This should be a vector
    )

    print(f"Created node with ID: {node1}")

    # Try to get attributes
    try:
        attrs = g.get_node_attrs(node1)
        print(f"Node attributes: {attrs}")
        print(f"Attributes type: {type(attrs)}")

        for key, value in attrs.items():
            print(f"  {key}: {value} (type: {type(value).__name__})")

    except Exception as e:
        print(f"❌ Error getting node attrs: {e}")

if __name__ == "__main__":
    print("🔧 Checking what visualization methods are available")
    print("=" * 60)

    # Check methods
    viz_result = check_viz_methods()

    # Check attributes directly
    check_node_attrs_directly()

    print("\n" + "=" * 60)
    print("Summary:")
    if viz_result:
        print("✅ Found visualization method")
        print("Check http://localhost:8080 and click nodes to see attributes")
    else:
        print("❌ No working visualization method found")
        print("The realtime viz might not be available in this version")