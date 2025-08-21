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

   # Add a node with attributes
   g.add_node("alice", age=30, role="engineer", active=True)
   g.add_node("bob", age=25, role="designer", active=False)

   # Nodes can have any ID type
   g.add_node(1, name="Node 1", value=100)
   g.add_node("user_123", email="user@example.com")

Batch Node Addition
~~~~~~~~~~~~~~~~~~~

For better performance with large datasets:

.. code-block:: python

   # Prepare node data
   nodes = [
       {'id': 'alice', 'age': 30, 'department': 'engineering'},
       {'id': 'bob', 'age': 25, 'department': 'design'},
       {'id': 'charlie', 'age': 35, 'department': 'management'}
   ]

   # Add all nodes at once (much faster)
   g.add_nodes(nodes)

Adding Edges
------------

Individual Edges
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Add edges with attributes
   g.add_edge("alice", "bob", relationship="collaborates", weight=0.8)
   g.add_edge("bob", "charlie", relationship="reports_to", weight=1.0)

Batch Edge Addition
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   edges = [
       {'source': 'alice', 'target': 'bob', 'weight': 0.8, 'type': 'collaboration'},
       {'source': 'bob', 'target': 'charlie', 'weight': 1.0, 'type': 'reporting'},
       {'source': 'alice', 'target': 'charlie', 'weight': 0.6, 'type': 'communication'}
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
   print(f"Directed: {g.directed}")

   # Check if graph is connected
   print(f"Connected: {g.is_connected()}")

Node and Edge Queries
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Check if entities exist
   print(g.has_node("alice"))      # True
   print(g.has_edge("alice", "bob"))  # True

   # Get node/edge attributes
   alice_data = g.get_node("alice")
   print(alice_data)  # {'age': 30, 'role': 'engineer', 'active': True}

   edge_data = g.get_edge("alice", "bob")
   print(edge_data)   # {'relationship': 'collaborates', 'weight': 0.8}

Degree Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get degree for specific node
   alice_degree = g.degree("alice")
   print(f"Alice's degree: {alice_degree}")

   # Get degrees for all nodes
   all_degrees = g.degree()
   print(all_degrees)  # {'alice': 2, 'bob': 2, 'charlie': 1}

   # For directed graphs
   if g.directed:
       in_degree = g.in_degree("bob")
       out_degree = g.out_degree("bob")
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

   # Filter nodes by attributes
   engineers = g.nodes.filter("role == 'engineer'")
   young_people = g.nodes.filter("age < 30")
   active_users = g.nodes.filter("active == True")

   # Complex filters
   young_engineers = g.nodes.filter("role == 'engineer' AND age < 35")

   # Filter edges
   strong_connections = g.edges.filter("weight > 0.7")

Updating Graph Data
-------------------

Modifying Attributes
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Update node attributes
   g.update_node("alice", {"age": 31, "promoted": True})

   # Update edge attributes
   g.update_edge("alice", "bob", {"weight": 0.9, "last_interaction": "2025-08-21"})

   # Update multiple nodes
   updates = {
       "alice": {"salary": 95000},
       "bob": {"salary": 75000}
   }
   g.update_nodes(updates)

Removing Elements
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Remove individual node (also removes connected edges)
   g.remove_node("charlie")

   # Remove individual edge
   g.remove_edge("alice", "bob")

   # Remove multiple nodes
   g.remove_nodes(["alice", "bob"])

   # Remove multiple edges
   g.remove_edges([("alice", "bob"), ("bob", "charlie")])

Graph Analysis
--------------

Connectivity
~~~~~~~~~~~~

.. code-block:: python

   # Find connected components
   components = g.connected_components()
   print(f"Number of components: {len(components)}")

   for i, component in enumerate(components):
       print(f"Component {i}: {len(component.node_ids)} nodes")

Neighborhoods
~~~~~~~~~~~~~

.. code-block:: python

   # Get neighbors of a node
   alice_neighbors = g.neighbors("alice")
   print(f"Alice's neighbors: {alice_neighbors}")

   # For directed graphs
   if g.directed:
       predecessors = g.predecessors("alice")  # Incoming edges
       successors = g.successors("alice")     # Outgoing edges

Path Finding
~~~~~~~~~~~~

.. code-block:: python

   # Find shortest path
   try:
       path = g.shortest_path("alice", "charlie")
       print(f"Shortest path: {path}")
   except Exception:
       print("No path found")

   # Check if path exists
   has_path = g.has_path("alice", "charlie")

Traversal
~~~~~~~~~

.. code-block:: python

   # Breadth-first search
   bfs_visited = g.bfs(start_node="alice")
   print(f"BFS visited: {bfs_visited}")

   # Depth-first search
   dfs_visited = g.dfs(start_node="alice")
   print(f"DFS visited: {dfs_visited}")

Working with Subgraphs
----------------------

Creating Subgraphs
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create subgraph from node filter
   engineering_team = g.filter_nodes("department == 'engineering'")

   # Create subgraph from specific nodes
   core_team = g.subgraph(["alice", "bob"])

   # Get largest connected component
   largest_component = max(g.connected_components(), key=lambda c: len(c.node_ids))

Subgraph Operations
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Subgraphs have the same interface as full graphs
   print(f"Subgraph nodes: {len(engineering_team.node_ids)}")
   print(f"Subgraph edges: {len(engineering_team.edge_ids)}")

   # Convert subgraph to table for analysis
   team_table = engineering_team.table()
   print(team_table.describe())

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
       node_data = g.get_node("nonexistent")
   except KeyError as e:
       print(f"Node not found: {e}")

   try:
       g.add_edge("alice", "nonexistent", weight=1.0)
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
   components = g.connected_components()
   largest = max(components, key=lambda c: len(c.node_ids))

   # 4. Extract insights
   analysis_table = largest.table()
   summary = analysis_table.group_by('department').agg({
       'age': ['mean', 'count'],
       'salary': ['mean', 'std']
   })

This covers the fundamental graph operations in Groggy. Next, explore :doc:`storage-views` for advanced data analysis capabilities.