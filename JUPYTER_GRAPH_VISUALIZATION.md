# 📱 Jupyter Graph Visualization Guide

## 🎯 Quick Start for Jupyter Notebooks

### Step 1: Setup in Jupyter Cell

```python
# Add groggy to your path (adjust as needed)
import sys
import os
sys.path.insert(0, 'path/to/groggy/python-groggy/python')

# Load Jupyter visualization functions
exec(open('path/to/jupyter_graph_viz.py').read())

from IPython.display import display, HTML
```

### Step 2: Create and Display Graph

```python
# Method 1: Quick test graph
g, iframe = quick_test_graph()
display(HTML(iframe))
```

```python
# Method 2: Custom graph data
nodes = [
    {'id': 'Alice', 'name': 'Alice Smith', 'role': 'manager'},
    {'id': 'Bob', 'name': 'Bob Jones', 'role': 'developer'},
    {'id': 'Carol', 'name': 'Carol Brown', 'role': 'designer'}
]
edges = [
    {'source': 'Alice', 'target': 'Bob'},
    {'source': 'Alice', 'target': 'Carol'}
]

g, iframe = jupyter_graph_viz_with_data(nodes, edges)
display(HTML(iframe))
```

```python
# Method 3: Existing groggy graph
import groggy as gr

g = gr.Graph()
node_center = g.add_node()
g.set_node_attr(node_center, "name", "Hub")

# Add satellite nodes
for i in range(5):
    node = g.add_node()
    g.set_node_attr(node, "name", f"Node_{i}")
    g.add_edge(node_center, node)

iframe = jupyter_graph_viz(g)
display(HTML(iframe))
```

## 🔧 Server Management

```python
# Check active servers
show_server_status()

# Clean up old servers (keep only the most recent)
cleanup_servers(keep_last=1)

# Clean up all servers
cleanup_all_servers()
```

## 🎨 What You'll See

The visualization will show:
- **📊 Table Tab**: Node data with attributes (name, role, etc.)
- **🕸️ Graph Tab**: Interactive network with nodes and edges
- **🎮 Controls**: Pan, zoom, drag, reset view, change layout
- **📱 Responsive**: Works in Jupyter notebook iframe

## 🔍 Key Features

### ✅ **Working Canvas System**
- HTML5 Canvas rendering with real-time WebSocket updates
- Interactive controls (pan, zoom, drag nodes)
- Multiple layout algorithms (force-directed, circular, etc.)

### ✅ **Jupyter-Safe**
- Prevents server garbage collection during cell execution
- Server reference management with cleanup functions
- Multiple visualizations can run simultaneously

### ✅ **Real Graph Data**
- Shows actual nodes and edges (not empty canvas)
- Node attributes displayed in table
- Edge connections properly rendered

## 🧪 Complete Example Notebook

```python
# === CELL 1: Setup ===
import sys
import os

# Adjust this path to your groggy installation
sys.path.insert(0, '/path/to/groggy/python-groggy/python')

# Load the Jupyter visualization functions
exec(open('/path/to/jupyter_graph_viz.py').read())

from IPython.display import display, HTML
import groggy as gr

print("🚀 Groggy Graph Visualization Ready!")
```

```python
# === CELL 2: Social Network Example ===
# Create a social network graph
nodes = [
    {'id': 'alice', 'name': 'Alice', 'department': 'Engineering', 'level': 'Senior'},
    {'id': 'bob', 'name': 'Bob', 'department': 'Design', 'level': 'Mid'},
    {'id': 'carol', 'name': 'Carol', 'department': 'Product', 'level': 'Junior'},
    {'id': 'dave', 'name': 'Dave', 'department': 'Engineering', 'level': 'Senior'},
    {'id': 'eve', 'name': 'Eve', 'department': 'Design', 'level': 'Senior'}
]

edges = [
    {'source': 'alice', 'target': 'bob'},
    {'source': 'alice', 'target': 'dave'},
    {'source': 'bob', 'target': 'carol'},
    {'source': 'carol', 'target': 'eve'},
    {'source': 'dave', 'target': 'eve'}
]

g, iframe = jupyter_graph_viz_with_data(nodes, edges)
print(f"📊 Created social network: {g.node_count()} people, {g.edge_count()} connections")
display(HTML(iframe))
```

```python
# === CELL 3: Check What You See ===
print("🔍 What you should see:")
print("📋 Table Tab: 5 rows with name, department, level columns")
print("🕸️ Graph Tab: 5 nodes connected in a network")
print("🎮 Interactive: Click nodes, drag them around, zoom in/out")
print("🔄 Layout: Try the 'Change Layout' button for different arrangements")

show_server_status()
```

```python
# === CELL 4: Cleanup When Done ===
# Keep only the most recent visualization
cleanup_servers(keep_last=1)

# Or clean up everything
# cleanup_all_servers()
```

## 🎉 Expected Output

When you run the cells above, you should see:

1. **✅ Server startup message**: "Graph visualization server started on port XXXX"
2. **✅ Interactive iframe**: Embedded visualization with table and graph tabs
3. **✅ Table data**: Actual node attributes displayed properly
4. **✅ Graph rendering**: Nodes and edges visible and interactive
5. **✅ Working controls**: Pan, zoom, drag, layout buttons functional

## 🔧 Troubleshooting

### Problem: Empty graph tab
**Solution**: Make sure you're using `jupyter_graph_viz()` function, not the old table approach

### Problem: Server not starting
**Solution**: Check that the path to `python-groggy/python` is correct

### Problem: "[object Object]" in table cells
**Solution**: This was fixed in our cleanup - you should see actual values now

### Problem: Multiple servers running
**Solution**: Use `cleanup_servers(keep_last=1)` to clean up old servers

## 🌟 Pro Tips

1. **Use custom data** for meaningful visualizations
2. **Set node attributes** - they show up in the table tab
3. **Try different layouts** using the Change Layout button
4. **Clean up servers** when done to free memory
5. **Check server status** to see what's running

The unified canvas-drawn Rust engine provides **real graph visualization** that actually works in Jupyter! 🎯