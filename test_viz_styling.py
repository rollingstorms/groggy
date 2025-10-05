#!/usr/bin/env python3
"""
Test VizConfig styling implementation
Tests all three parameter types: array, column, single value
"""

import groggy as gr
import time

print("=" * 80)
print("Testing VizConfig Styling Implementation")
print("=" * 80)

# Create a small test graph with attributes
g = gr.Graph()

# Add nodes with attributes for column-based styling
node_ids = []
for i in range(5):
    nid = g.add_node(
        label=f"Node {i}",
        priority=i,
        category=["A", "B", "A", "C", "B"][i],
        size_val=float(5 + i * 3)
    )
    node_ids.append(nid)

# Add edges with attributes
g.add_edge(node_ids[0], node_ids[1], weight=1.0, type="strong")
g.add_edge(node_ids[1], node_ids[2], weight=2.0, type="weak")
g.add_edge(node_ids[2], node_ids[3], weight=1.5, type="strong")
g.add_edge(node_ids[3], node_ids[4], weight=0.5, type="weak")
g.add_edge(node_ids[4], node_ids[0], weight=1.0, type="medium")

print("\n✅ Created test graph:")
print(f"   Nodes: {len(node_ids)}")
print(f"   Edges: {g.num_edges()}")
print(f"\nNode attributes:")
for i, nid in enumerate(node_ids):
    attrs = g.nodes.table().to_pandas().iloc[i]
    print(f"   Node {i}: priority={attrs['priority']}, category={attrs['category']}, size_val={attrs['size_val']}")

# Test 1: Single values (all nodes/edges same style)
print("\n" + "=" * 80)
print("TEST 1: Single Value Parameters")
print("=" * 80)
print("Setting:")
print("  - All nodes RED with size 10")
print("  - All edges BLUE with width 3")
print("\nStarting server...")

try:
    result = g.viz.show(
        layout="circular",
        node_color="red",
        node_size=10.0,
        edge_color="blue",
        edge_width=3.0
    )
    print("✅ Server started successfully!")
    print("\n📊 Check the visualization at http://127.0.0.1:8080/")
    print("   All nodes should be RED with size 10")
    print("   All edges should be BLUE with width 3")

    time.sleep(3)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Stop the server
try:
    g.viz.stop()
    print("\n✅ Server stopped")
except:
    pass

time.sleep(1)

# Test 2: Column references (styling from attributes)
print("\n" + "=" * 80)
print("TEST 2: Column Reference Parameters")
print("=" * 80)
print("Setting:")
print("  - Node colors from 'category' column (A/B/C)")
print("  - Node sizes from 'size_val' column (5, 8, 11, 14, 17)")
print("  - Node labels from 'label' column")
print("\nStarting server...")

try:
    result = g.viz.show(
        layout="circular",
        node_color="category",  # Column reference - will show raw values
        node_size="size_val",   # Column reference - numeric values
        label="label",          # Column reference - text values
        label_size=14.0
    )
    print("✅ Server started successfully!")
    print("\n📊 Check the visualization at http://127.0.0.1:8080/")
    print("   Node colors should vary by category")
    print("   Node sizes should vary (5, 8, 11, 14, 17)")
    print("   Labels should show 'Node 0', 'Node 1', etc.")

    time.sleep(3)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Stop the server
try:
    g.viz.stop()
    print("\n✅ Server stopped")
except:
    pass

time.sleep(1)

# Test 3: Array parameters (direct styling per node/edge)
print("\n" + "=" * 80)
print("TEST 3: Array Parameters")
print("=" * 80)
print("Setting:")
print("  - Node colors as array: [red, green, blue, orange, purple]")
print("  - Node shapes as array: [circle, square, triangle, diamond, circle]")
print("  - Edge styles as array: [solid, dashed, dotted, dashed, solid]")
print("\nStarting server...")

try:
    result = g.viz.show(
        layout="circular",
        node_color=["red", "green", "blue", "orange", "purple"],
        node_size=[8.0, 12.0, 10.0, 15.0, 9.0],
        node_shape=["circle", "square", "triangle", "diamond", "circle"],
        edge_color=["red", "green", "blue", "orange", "purple"],
        edge_width=[1.0, 2.0, 3.0, 2.0, 1.5],
        edge_style=["solid", "dashed", "dotted", "dashed", "solid"],
        label="label"
    )
    print("✅ Server started successfully!")
    print("\n📊 Check the visualization at http://127.0.0.1:8080/")
    print("   Node 0: red circle")
    print("   Node 1: green square")
    print("   Node 2: blue triangle")
    print("   Node 3: orange diamond")
    print("   Node 4: purple circle")
    print("   Edges should have varying colors, widths, and styles")

    time.sleep(3)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Stop the server
try:
    g.viz.stop()
    print("\n✅ Server stopped")
except:
    pass

time.sleep(1)

# Test 4: Mixed parameters with advanced styling
print("\n" + "=" * 80)
print("TEST 4: Mixed Parameters + Advanced Styling")
print("=" * 80)
print("Setting:")
print("  - Node colors from array")
print("  - Node sizes from column 'size_val'")
print("  - Node borders (width 2, black color)")
print("  - Node opacity 0.8")
print("  - Edge opacity 0.6")
print("\nStarting server...")

try:
    result = g.viz.show(
        layout="force_directed",
        node_color=["#ff6b6b", "#4ecdc4", "#45b7d1", "#f9ca24", "#6c5ce7"],
        node_size="size_val",
        node_opacity=0.8,
        node_border_width=2.0,
        node_border_color="black",
        edge_color="#95a5a6",
        edge_opacity=0.6,
        edge_width=2.0,
        label="label",
        label_color="white",
        label_size=14.0
    )
    print("✅ Server started successfully!")
    print("\n📊 Check the visualization at http://127.0.0.1:8080/")
    print("   Nodes should have varying colors (array)")
    print("   Nodes should have varying sizes (from size_val column)")
    print("   Nodes should have black borders (width 2)")
    print("   Nodes should be semi-transparent (opacity 0.8)")
    print("   Edges should be gray with opacity 0.6")
    print("   Labels should be white, size 14")

    print("\n⏳ Server running for 10 seconds...")
    time.sleep(10)

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback
    traceback.print_exc()

# Stop the server
try:
    g.viz.stop()
    print("\n✅ Server stopped")
except:
    pass

print("\n" + "=" * 80)
print("✅ All tests completed!")
print("=" * 80)
