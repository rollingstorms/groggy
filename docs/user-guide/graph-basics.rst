Graph Basics
============

This guide covers the fundamental concepts and operations for working with graphs in Groggy.

Creating Your First Graph
--------------------------

.. code-block:: python

   import groggy as gr

   # Create a new directed graph
   g = gr.Graph(directed=True)

   # Create an undirected graph
   g_undirected = gr.Graph(directed=False)

Adding Nodes
------------

Individual Nodes
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add nodes with attributes (returns numeric IDs)
   alice = g.add_node(name="Alice", age=30, role="engineer", active=True)
   bob = g.add_node(name="Bob", age=25, role="designer", active=False)

   # Store the returned IDs for later use
   node1 = g.add_node(name="Node 1", value=100)
   user_node = g.add_node(user_id="user_123", email="user@example.com")

Batch Node Addition
~~~~~~~~~~~~~~~~~~~

For better performance with large datasets:

.. code-block:: python

   # Prepare node data (no 'id' field needed)
   nodes = [
       {'name': 'Alice', 'age': 30, 'department': 'engineering'},
       {'name': 'Bob', 'age': 25, 'department': 'design'},
       {'name': 'Charlie', 'age': 35, 'department': 'management'}
   ]

   # Add all nodes at once (returns list of node IDs)
   node_ids = g.add_nodes(nodes)

Adding Edges
------------

Individual Edges
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add edges with attributes (using node IDs)
   g.add_edge(alice, bob, relationship="collaborates", weight=0.8)
   g.add_edge(bob, node_ids[2], relationship="reports_to", weight=1.0)  # charlie

Batch Edge Addition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use actual node IDs (assuming alice, bob, charlie from above)
   edges = [
       (alice, bob, {'weight': 0.8, 'type': 'collaboration'}),
       (bob, node_ids[2], {'weight': 1.0, 'type': 'reporting'}),
       (alice, node_ids[2], {'weight': 0.6, 'type': 'communication'})
   ]

   g.add_edges(edges)

Graph Properties
----------------

Basic Information
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Graph size
   print(f"Nodes: {g.node_count()}")
   print(f"Edges: {g.edge_count()}")

   # Graph type  
   print(f"Directed: {g.is_directed}")

   # Check if graph is connected
   print(f"Connected: {g.is_connected()}")

Node and Edge Queries
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check if entities exist
   print(g.has_node(alice))         # True
   print(g.has_edge(alice, bob))    # True

   # Get node attributes using node views
   alice_data = g.nodes[alice]
   print(alice_data)  # Node attribute access

   # Get edge attributes using edge views  
   edges_table = g.edges.table()
   # Find edge between alice and bob
   alice_bob_edges = edges_table[(edges_table['source'] == alice) & (edges_table['target'] == bob)]

Degree Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get degree for specific node
   alice_degree = g.degree(alice)
   print(f"Alice's degree: {alice_degree}")

   # Get degrees for all nodes
   all_degrees = g.degree()
   print(all_degrees)  # List or array of degrees

   # For directed graphs
   if g.is_directed:
       in_degree = g.in_degree(bob)
       out_degree = g.out_degree(bob)
       print(f"Bob - In: {in_degree}, Out: {out_degree}")

Accessing Graph Data
--------------------

Node and Edge Views
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access nodes and edges as views
   nodes = g.nodes
   edges = g.edges

   # Iterate over nodes
   for node_id in g.nodes:
       print(f"Node: {node_id}")

   # Iterate over edges  
   for source, target in g.edges:
       print(f"Edge: {source} -> {target}")

   # Get node attributes
   for node_id in g.nodes:
       attrs = g.nodes[node_id]
       print(f"{node_id}: {attrs}")

Filtering and Querying
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Filter nodes using proper filter syntax
   engineers = g.filter_nodes(gr.NodeFilter.attribute_equals("role", "engineer"))
   young_people = g.filter_nodes(gr.NodeFilter.attribute_filter("age", gr.AttributeFilter.less_than(30)))
   active_users = g.filter_nodes(gr.NodeFilter.attribute_equals("active", True))

   # Complex filters
   young_engineers = g.filter_nodes(gr.NodeFilter.and_filters([
       gr.NodeFilter.attribute_equals("role", "engineer"),
       gr.NodeFilter.attribute_filter("age", gr.AttributeFilter.less_than(35))
   ]))

   # Filter edges
   strong_connections = g.filter_edges(gr.EdgeFilter.attribute_filter("weight", gr.AttributeFilter.greater_than(0.7)))

Updating Graph Data
-------------------

Modifying Attributes
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Node attribute modification (limited support in current version)
   # Individual attribute setting may have type restrictions
   
   # Note: Direct attribute modification is limited in current release
   # Use node recreation for complex attribute changes
   
   # Access current attributes
   alice_data = g.nodes[alice]
   print(f"Current attributes: {alice_data.keys()}")
   
   # For bulk attribute updates, see storage views documentation

Removing Elements
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Remove individual node (also removes connected edges)
   g.remove_node(node_ids[2])  # charlie

   # Remove individual edge
   g.remove_edge(alice, bob)

   # Remove multiple nodes
   g.remove_nodes([alice, bob])

   # Remove multiple edges
   g.remove_edges([(alice, bob), (bob, node_ids[2])])

Graph Analysis
--------------

Connectivity
~~~~~~~~~~~~

.. code-block:: python

   # Find connected components
   components = g.analytics.connected_components()
   print(f"Number of components: {len(components)}")

   for i, component in enumerate(components):
       print(f"Component {i}: {len(component)} nodes")

Neighborhoods
~~~~~~~~~~~~~

.. code-block:: python

   # Get neighbors of a node
   alice_neighbors = g.neighbors(alice)
   print(f"Alice's neighbors: {alice_neighbors}")

   # For directed graphs
   if g.is_directed:
       predecessors = g.predecessors(alice)  # Incoming edges
       successors = g.successors(alice)     # Outgoing edges

Path Finding
~~~~~~~~~~~~

.. code-block:: python

   # Find shortest path
   try:
       path = g.analytics.shortest_path(alice, node_ids[2])  # alice to charlie
       print(f"Shortest path: {path}")
   except Exception:
       print("No path found")

   # Check if path exists
   has_path = g.analytics.has_path(alice, node_ids[2])

Traversal
~~~~~~~~~

.. code-block:: python

   # Breadth-first search
   bfs_visited = g.analytics.bfs(alice)
   print(f"BFS visited: {bfs_visited}")

   # Depth-first search
   dfs_visited = g.analytics.dfs(alice)
   print(f"DFS visited: {dfs_visited}")

Working with Subgraphs
----------------------

Creating Subgraphs
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create subgraph from node filter
   engineering_team = g.filter_nodes(gr.NodeFilter.attribute_equals("department", "engineering"))

   # Create subgraph from specific nodes
   core_team = g.subgraph([alice, bob])

   # Get largest connected component
   components = g.analytics.connected_components()
   largest_component = max(components, key=lambda c: len(c))

Subgraph Operations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Subgraphs have the same interface as full graphs
   print(f"Subgraph nodes: {engineering_team.node_count()}")
   print(f"Subgraph edges: {engineering_team.edge_count()}")

   # Convert subgraph to table for analysis
   team_table = engineering_team.nodes.table()
   print(f"Team size: {len(team_table)}")

Best Practices
--------------

Performance Tips
~~~~~~~~~~~~~~~~

1. **Use batch operations** for adding many nodes/edges
2. **Filter efficiently** with simple attribute comparisons  
3. **Cache results** of expensive operations
4. **Use appropriate data types** for node IDs

Memory Management
~~~~~~~~~~~~~~~~~

1. **Remove unused nodes/edges** to free memory
2. **Use views instead of copying** large datasets
3. **Process large graphs in chunks** when possible

Error Handling
~~~~~~~~~~~~~~

.. code-block:: python

   try:
       node_data = g.nodes[999]  # Nonexistent node ID
   except KeyError as e:
       print(f"Node not found: {e}")

   try:
       g.add_edge(alice, 999, weight=1.0)  # Nonexistent target
   except ValueError as e:
       print(f"Invalid edge: {e}")

Common Patterns
---------------

Loading Data
~~~~~~~~~~~~

.. code-block:: python

   import pandas as pd

   # Load from CSV
   nodes_df = pd.read_csv("nodes.csv")
   edges_df = pd.read_csv("edges.csv")

   # Convert to Groggy format
   nodes_data = nodes_df.to_dict('records')
   edges_data = edges_df.to_dict('records')

   # Build graph
   g = gr.Graph()
   g.add_nodes(nodes_data)
   g.add_edges(edges_data)

Analysis Pipeline
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # 1. Load data
   g = gr.Graph()
   g.add_nodes(node_data)
   g.add_edges(edge_data)

   # 2. Basic analysis
   print(f"Graph density: {g.density():.3f}")
   print(f"Average degree: {sum(g.degree().values()) / g.node_count():.2f}")

   # 3. Find important nodes
   components = g.analytics.connected_components()
   largest = max(components, key=lambda c: len(c))

   # 4. Extract insights from node table
   analysis_table = g.nodes.table()
   # Basic statistical analysis
   avg_age = analysis_table['age'].mean() if 'age' in analysis_table.columns else 0
   print(f"Average age: {avg_age}")

This covers the fundamental graph operations in Groggy. Next, explore :doc:`storage-views` for advanced data analysis capabilities.