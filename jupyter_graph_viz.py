"""
Jupyter-safe graph visualization for Groggy graphs
Copy this to your Jupyter notebook for working graph visualization
"""

import groggy
from IPython.display import display, HTML
import weakref
import atexit

# Global registry to prevent server garbage collection
_groggy_servers = []
_server_cleanup_registered = False

def _register_cleanup():
    """Register cleanup function to run when Python exits"""
    global _server_cleanup_registered
    if not _server_cleanup_registered:
        atexit.register(cleanup_all_servers)
        _server_cleanup_registered = True

def jupyter_graph_viz(graph):
    """
    Jupyter-safe version of graph_viz() that prevents server garbage collection

    Args:
        graph: Groggy Graph object

    Returns:
        str: HTML iframe that can be displayed in Jupyter

    Example:
        import groggy as gr
        g = gr.Graph()
        node1 = g.add_node()
        node2 = g.add_node()
        g.add_edge(node1, node2)

        iframe_html = jupyter_graph_viz(g)
        display(HTML(iframe_html))
    """
    global _groggy_servers

    # Register cleanup on first use
    _register_cleanup()

    try:
        # Generate the iframe HTML using our working graph_viz() method
        iframe_html = graph.graph_viz()

        # Extract port from iframe to identify the server
        import re
        port_match = re.search(r'127\.0\.0\.1:(\d+)', iframe_html)
        if port_match:
            port = int(port_match.group(1))

            # Store server reference to prevent garbage collection
            # Note: We can't directly access the server handle from graph_viz(),
            # but we can store the graph reference which keeps the server alive
            server_info = {
                'graph_ref': graph,
                'port': port,
                'iframe': iframe_html
            }
            _groggy_servers.append(server_info)

            print(f"📌 Graph visualization server started on port {port}")
            print(f"🔧 Keeping {len(_groggy_servers)} servers alive in Jupyter")

        return iframe_html

    except Exception as e:
        print(f"❌ Failed to create graph visualization: {e}")
        return f"<div>Error: Could not create graph visualization - {e}</div>"

def jupyter_graph_viz_with_data(nodes_data=None, edges_data=None):
    """
    Create a graph with sample data and display it in Jupyter

    Args:
        nodes_data: List of node dictionaries with 'id' and optional attributes
        edges_data: List of edge dictionaries with 'source', 'target'

    Returns:
        tuple: (graph, iframe_html)

    Example:
        nodes = [{'id': 'A', 'name': 'Node A'}, {'id': 'B', 'name': 'Node B'}]
        edges = [{'source': 'A', 'target': 'B'}]
        g, iframe = jupyter_graph_viz_with_data(nodes, edges)
        display(HTML(iframe))
    """
    import groggy as gr

    # Create graph
    g = gr.Graph()

    # Add sample data if none provided
    if nodes_data is None:
        nodes_data = [
            {'id': 'Node_0', 'name': 'Start', 'value': 10},
            {'id': 'Node_1', 'name': 'Middle', 'value': 20},
            {'id': 'Node_2', 'name': 'End', 'value': 30}
        ]

    if edges_data is None:
        edges_data = [
            {'source': 'Node_0', 'target': 'Node_1'},
            {'source': 'Node_1', 'target': 'Node_2'}
        ]

    # Add nodes
    node_id_map = {}
    for node_data in nodes_data:
        node_id = g.add_node()
        node_id_map[node_data['id']] = node_id

        # Add attributes
        for key, value in node_data.items():
            if key != 'id':
                g.set_node_attr(node_id, key, value)

    # Add edges
    for edge_data in edges_data:
        source_id = node_id_map.get(edge_data['source'])
        target_id = node_id_map.get(edge_data['target'])

        if source_id is not None and target_id is not None:
            g.add_edge(source_id, target_id)

    print(f"✅ Created graph: {g.node_count()} nodes, {g.edge_count()} edges")

    # Generate visualization
    iframe_html = jupyter_graph_viz(g)

    return g, iframe_html

def show_server_status():
    """Display information about active graph visualization servers"""
    global _groggy_servers

    print(f"📊 Graph Visualization Server Status")
    print(f"=" * 40)
    print(f"Active servers: {len(_groggy_servers)}")

    for i, server_info in enumerate(_groggy_servers):
        port = server_info.get('port', 'unknown')
        graph = server_info.get('graph_ref')
        nodes = graph.node_count() if graph else 0
        edges = graph.edge_count() if graph else 0
        print(f"  Server {i+1}: Port {port} - {nodes} nodes, {edges} edges")

def cleanup_servers(keep_last=1):
    """
    Clean up old graph visualization servers

    Args:
        keep_last: Number of most recent servers to keep alive (default: 1)
    """
    global _groggy_servers

    if len(_groggy_servers) <= keep_last:
        print(f"📌 Keeping all {len(_groggy_servers)} servers (≤ {keep_last})")
        return

    # Keep only the most recent servers
    servers_to_remove = len(_groggy_servers) - keep_last
    removed_servers = _groggy_servers[:servers_to_remove]
    _groggy_servers = _groggy_servers[servers_to_remove:]

    print(f"🧹 Cleaned up {len(removed_servers)} servers, keeping {len(_groggy_servers)}")

def cleanup_all_servers():
    """Clean up all graph visualization servers"""
    global _groggy_servers
    count = len(_groggy_servers)
    _groggy_servers.clear()
    if count > 0:
        print(f"🧹 Cleaned up all {count} graph visualization servers")

# Convenience function for quick testing
def quick_test_graph():
    """Create and display a quick test graph visualization"""
    print("🧪 Creating quick test graph...")

    g, iframe = jupyter_graph_viz_with_data()

    print("\n📱 Copy this to display in Jupyter:")
    print("display(HTML(iframe))")
    print(f"\n🌐 Or visit directly: http://127.0.0.1:{_groggy_servers[-1]['port']}")

    return g, iframe

# Example usage for Jupyter notebooks
JUPYTER_EXAMPLE = '''
# === JUPYTER NOTEBOOK EXAMPLE ===
# Copy and paste this into a Jupyter cell:

import sys
import os

# Add groggy to path (adjust path as needed)
sys.path.insert(0, 'path/to/groggy/python-groggy/python')

# Import the jupyter graph viz functions
exec(open('jupyter_graph_viz.py').read())

# Method 1: Quick test
g, iframe = quick_test_graph()
display(HTML(iframe))

# Method 2: With your own data
nodes = [
    {'id': 'A', 'name': 'Alice', 'role': 'admin'},
    {'id': 'B', 'name': 'Bob', 'role': 'user'},
    {'id': 'C', 'name': 'Carol', 'role': 'user'}
]
edges = [
    {'source': 'A', 'target': 'B'},
    {'source': 'A', 'target': 'C'}
]
g, iframe = jupyter_graph_viz_with_data(nodes, edges)
display(HTML(iframe))

# Method 3: Existing graph
import groggy as gr
g = gr.Graph()
# ... add your nodes and edges ...
iframe = jupyter_graph_viz(g)
display(HTML(iframe))

# Clean up when done
cleanup_servers(keep_last=1)  # Keep only the most recent
'''

if __name__ == "__main__":
    print("🚀 Jupyter Graph Visualization Module Loaded")
    print("=" * 50)
    print("Available functions:")
    print("  - jupyter_graph_viz(graph)")
    print("  - jupyter_graph_viz_with_data(nodes, edges)")
    print("  - quick_test_graph()")
    print("  - show_server_status()")
    print("  - cleanup_servers(keep_last=1)")
    print("  - cleanup_all_servers()")
    print("\nFor Jupyter notebook example:")
    print("print(JUPYTER_EXAMPLE)")