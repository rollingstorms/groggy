#!/usr/bin/env python3
"""
Test VizConfig styling with debug API inspection
Uses the debug endpoints to verify JSON structure
"""

import groggy as gr
import requests
import json
import time

print("=" * 80)
print("VizConfig Styling - Debug API Inspection")
print("=" * 80)

# Create test graph
g = gr.Graph()

node_ids = []
for i in range(3):
    nid = g.add_node(
        label=f"Node {i}",
        score=float(i * 10),
        category=["A", "B", "C"][i]
    )
    node_ids.append(nid)

g.add_edge(node_ids[0], node_ids[1], weight=1.0)
g.add_edge(node_ids[1], node_ids[2], weight=2.0)

print("\n✅ Created test graph: 3 nodes, 2 edges")

# Start server with styling
print("\n🚀 Starting visualization server with styling...")
try:
    g.viz.show(
        layout="circular",
        node_color=["red", "green", "blue"],
        node_size=[8.0, 12.0, 10.0],
        node_shape=["circle", "square", "triangle"],
        node_opacity=0.9,
        node_border_width=2.0,
        node_border_color="black",
        label="label",
        label_size=14.0,
        label_color="#333",
        edge_color=["#ff0000", "#00ff00"],
        edge_width=[2.0, 3.0],
        edge_opacity=0.7,
        edge_style=["solid", "dashed"]
    )
    print("✅ Server started")
except Exception as e:
    print(f"❌ Error starting server: {e}")
    import traceback
    traceback.print_exc()
    exit(1)

# Give server time to initialize
time.sleep(2)

# Query debug API
print("\n" + "=" * 80)
print("Querying Debug API at http://127.0.0.1:8080/api/debug")
print("=" * 80)

try:
    response = requests.get("http://127.0.0.1:8080/api/debug", timeout=5)
    response.raise_for_status()

    data = response.json()

    print("\n✅ Received debug data")
    print(f"   Nodes: {len(data.get('nodes', []))}")
    print(f"   Edges: {len(data.get('edges', []))}")

    # Inspect node styling fields
    print("\n" + "=" * 80)
    print("NODE STYLING FIELDS")
    print("=" * 80)

    nodes = data.get('nodes', [])
    for i, node in enumerate(nodes):
        print(f"\nNode {i} (id={node.get('id')}):")
        print(f"  ✓ color: {node.get('color', 'NOT SET')}")
        print(f"  ✓ size: {node.get('size', 'NOT SET')}")
        print(f"  ✓ shape: {node.get('shape', 'NOT SET')}")
        print(f"  ✓ opacity: {node.get('opacity', 'NOT SET')}")
        print(f"  ✓ border_color: {node.get('border_color', 'NOT SET')}")
        print(f"  ✓ border_width: {node.get('border_width', 'NOT SET')}")
        print(f"  ✓ label: {node.get('label', 'NOT SET')}")
        print(f"  ✓ label_size: {node.get('label_size', 'NOT SET')}")
        print(f"  ✓ label_color: {node.get('label_color', 'NOT SET')}")

        # Also show attributes
        attrs = node.get('attributes', {})
        if attrs:
            print(f"  Attributes: {list(attrs.keys())}")

    # Inspect edge styling fields
    print("\n" + "=" * 80)
    print("EDGE STYLING FIELDS")
    print("=" * 80)

    edges = data.get('edges', [])
    for i, edge in enumerate(edges):
        print(f"\nEdge {i} (id={edge.get('id')}, {edge.get('source')} → {edge.get('target')}):")
        print(f"  ✓ color: {edge.get('color', 'NOT SET')}")
        print(f"  ✓ width: {edge.get('width', 'NOT SET')}")
        print(f"  ✓ opacity: {edge.get('opacity', 'NOT SET')}")
        print(f"  ✓ style: {edge.get('style', 'NOT SET')}")

        # Also show attributes
        attrs = edge.get('attributes', {})
        if attrs:
            print(f"  Attributes: {list(attrs.keys())}")

    # Print full JSON for verification
    print("\n" + "=" * 80)
    print("FULL JSON STRUCTURE (nodes only)")
    print("=" * 80)
    print(json.dumps(nodes, indent=2))

    # Verification checks
    print("\n" + "=" * 80)
    print("VERIFICATION CHECKS")
    print("=" * 80)

    checks = []

    # Check node styling
    if len(nodes) >= 3:
        checks.append(("Node 0 color is 'red'", nodes[0].get('color') == 'red'))
        checks.append(("Node 1 color is 'green'", nodes[1].get('color') == 'green'))
        checks.append(("Node 2 color is 'blue'", nodes[2].get('color') == 'blue'))
        checks.append(("Node 0 size is 8.0", nodes[0].get('size') == 8.0))
        checks.append(("Node 1 size is 12.0", nodes[1].get('size') == 12.0))
        checks.append(("Node 0 shape is 'circle'", nodes[0].get('shape') == 'circle'))
        checks.append(("Node 1 shape is 'square'", nodes[1].get('shape') == 'square'))
        checks.append(("Node 2 shape is 'triangle'", nodes[2].get('shape') == 'triangle'))
        checks.append(("Node 0 opacity is 0.9", nodes[0].get('opacity') == 0.9))
        checks.append(("Node 0 border_width is 2.0", nodes[0].get('border_width') == 2.0))
        checks.append(("Node 0 border_color is 'black'", nodes[0].get('border_color') == 'black'))
        checks.append(("Node 0 label is 'Node 0'", nodes[0].get('label') == 'Node 0'))
        checks.append(("Node 0 label_size is 14.0", nodes[0].get('label_size') == 14.0))
        checks.append(("Node 0 label_color is '#333'", nodes[0].get('label_color') == '#333'))

    # Check edge styling
    if len(edges) >= 2:
        checks.append(("Edge 0 color is '#ff0000'", edges[0].get('color') == '#ff0000'))
        checks.append(("Edge 1 color is '#00ff00'", edges[1].get('color') == '#00ff00'))
        checks.append(("Edge 0 width is 2.0", edges[0].get('width') == 2.0))
        checks.append(("Edge 1 width is 3.0", edges[1].get('width') == 3.0))
        checks.append(("Edge 0 opacity is 0.7", edges[0].get('opacity') == 0.7))
        checks.append(("Edge 0 style is 'solid'", edges[0].get('style') == 'solid'))
        checks.append(("Edge 1 style is 'dashed'", edges[1].get('style') == 'dashed'))

    passed = 0
    failed = 0

    for check_name, result in checks:
        if result:
            print(f"  ✅ {check_name}")
            passed += 1
        else:
            print(f"  ❌ {check_name}")
            failed += 1

    print(f"\n📊 Results: {passed} passed, {failed} failed out of {len(checks)} checks")

    if failed == 0:
        print("\n🎉 All checks passed! VizConfig styling is working correctly!")
    else:
        print(f"\n⚠️  {failed} checks failed. Review the output above.")

except requests.exceptions.RequestException as e:
    print(f"❌ Error querying debug API: {e}")
except Exception as e:
    print(f"❌ Unexpected error: {e}")
    import traceback
    traceback.print_exc()

# Keep server running for manual inspection
print("\n" + "=" * 80)
print("🌐 Visualization available at: http://127.0.0.1:8080/")
print("📊 Debug API at: http://127.0.0.1:8080/api/debug")
print("\nPress Ctrl+C to stop the server...")
print("=" * 80)

try:
    while True:
        time.sleep(1)
except KeyboardInterrupt:
    print("\n\n🛑 Stopping server...")
    try:
        g.viz.stop()
        print("✅ Server stopped")
    except:
        pass
