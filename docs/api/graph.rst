Graph API
=========

The Graph class is the primary interface for creating and manipulating graphs in Groggy.

Graph Constructor
-----------------

.. class:: Graph(directed=True)

   Create a new graph instance.

   :param bool directed: Whether the graph is directed (default: True)

   **Example:**

   .. code-block:: python

      import groggy as gr

      # Create directed graph
      g = gr.Graph(directed=True)

      # Create undirected graph  
      g_undirected = gr.Graph(directed=False)

Node Operations
---------------

Adding Nodes
~~~~~~~~~~~~

.. method:: Graph.add_node(node_id, **attributes)

   Add a single node with attributes.

   :param node_id: Unique identifier for the node
   :type node_id: str or int
   :param attributes: Node attributes as keyword arguments
   :raises ValueError: If node already exists

   **Example:**

   .. code-block:: python

      alice = g.add_node(name="Alice", age=30, role="engineer")
      node1 = g.add_node(name="Node 1", active=True)

.. method:: Graph.add_nodes(nodes)

   Add multiple nodes efficiently.

   :param list nodes: List of node dictionaries with 'id' key and attributes
   :raises ValueError: If any node already exists

   **Example:**

   .. code-block:: python

      nodes = [
          {'name': 'Alice', 'age': 30, 'role': 'engineer'},
          {'name': 'Bob', 'age': 25, 'role': 'designer'}
      ]
      node_ids = g.add_nodes(nodes)  # Returns list of node IDs

Querying Nodes
~~~~~~~~~~~~~~

.. method:: Graph.has_node(node_id)

   Check if a node exists.

   :param node_id: Node ID to check
   :type node_id: str or int
   :returns: True if node exists, False otherwise
   :rtype: bool

.. method:: Graph.node_count()

   Get the number of nodes in the graph.

   :returns: Number of nodes
   :rtype: int

   **Note:** To access node attributes, use ``g.nodes[node_id]`` instead of ``get_node()``.

Modifying Nodes
~~~~~~~~~~~~~~~

.. method:: Graph.set_node_attribute(node_id, name, value)

   Set a single node attribute.

   :param node_id: Node ID
   :type node_id: str or int  
   :param str name: Attribute name
   :param value: Attribute value
   :raises KeyError: If node doesn't exist

   **Note:** For batch updates, use ``g.set_node_attributes()`` instead of ``update_node()``.

.. method:: Graph.remove_node(node_id)

   Remove a node and all connected edges.

   :param node_id: ID of the node to remove
   :type node_id: str or int
   :raises KeyError: If node doesn't exist

.. method:: Graph.remove_nodes(node_ids)

   Remove multiple nodes and their edges.

   :param list node_ids: List of node IDs to remove
   :raises KeyError: If any node doesn't exist

Edge Operations
---------------

Adding Edges
~~~~~~~~~~~~

.. method:: Graph.add_edge(source, target, **attributes)

   Add a single edge with attributes.

   :param source: Source node ID
   :type source: str or int
   :param target: Target node ID
   :type target: str or int
   :param attributes: Edge attributes as keyword arguments
   :raises ValueError: If edge already exists or nodes don't exist

   **Example:**

   .. code-block:: python

      g.add_edge(alice, bob, weight=0.8, relationship="friend")

.. method:: Graph.add_edges(edges)

   Add multiple edges efficiently.

   :param list edges: List of edge dictionaries with 'source', 'target', and attributes
   :raises ValueError: If any edge already exists or nodes don't exist

   **Example:**

   .. code-block:: python

      edges = [
          (alice, bob, {'weight': 0.8}),
          (bob, charlie, {'weight': 0.6})
      ]
      g.add_edges(edges)

Querying Edges
~~~~~~~~~~~~~~

.. method:: Graph.has_edge(source, target)

   Check if an edge exists.

   :param source: Source node ID
   :type source: str or int
   :param target: Target node ID
   :type target: str or int
   :returns: True if edge exists, False otherwise
   :rtype: bool

.. method:: Graph.edge_count()

   Get the number of edges in the graph.

   :returns: Number of edges
   :rtype: int

   **Note:** To access edge attributes, use the edges table: ``g.edges.table()``

Modifying Edges
~~~~~~~~~~~~~~~

.. method:: Graph.set_edge_attribute(source, target, name, value)

   Set a single edge attribute.

   :param source: Source node ID
   :type source: str or int
   :param target: Target node ID  
   :type target: str or int
   :param str name: Attribute name
   :param value: Attribute value
   :raises KeyError: If edge doesn't exist

.. method:: Graph.remove_edge(source, target)

   Remove an edge.

   :param source: Source node ID
   :type source: str or int
   :param target: Target node ID
   :type target: str or int
   :raises KeyError: If edge doesn't exist

.. method:: Graph.remove_edges(edges)

   Remove multiple edges.

   :param list edges: List of (source, target) tuples
   :raises KeyError: If any edge doesn't exist

Graph Properties
----------------

.. attribute:: Graph.directed

   Whether the graph is directed.

   :type: bool

.. method:: Graph.density()

   Calculate graph density.

   :returns: Graph density (edges / possible_edges)
   :rtype: float

.. method:: Graph.is_connected()

   Check if the graph is connected.

   :returns: True if graph is connected, False otherwise
   :rtype: bool

Degree Operations
-----------------

.. method:: Graph.degree(node_id=None)

   Get degree(s) of node(s).

   :param node_id: Specific node ID, or None for all nodes
   :type node_id: str, int, or None
   :returns: Single degree or dictionary of node_id -> degree
   :rtype: int or dict

   **Example:**

   .. code-block:: python

      # Single node degree
      alice_degree = g.degree("alice")

      # All node degrees
      all_degrees = g.degree()

.. method:: Graph.in_degree(node_id=None)

   Get in-degree(s) for directed graphs.

   :param node_id: Specific node ID, or None for all nodes
   :type node_id: str, int, or None
   :returns: Single in-degree or dictionary of node_id -> in_degree
   :rtype: int or dict
   :raises ValueError: If graph is undirected

.. method:: Graph.out_degree(node_id=None)

   Get out-degree(s) for directed graphs.

   :param node_id: Specific node ID, or None for all nodes  
   :type node_id: str, int, or None
   :returns: Single out-degree or dictionary of node_id -> out_degree
   :rtype: int or dict
   :raises ValueError: If graph is undirected

Neighborhood Operations
-----------------------

.. method:: Graph.neighbors(node_id)

   Get neighbors of a node.

   :param node_id: Node ID
   :type node_id: str or int
   :returns: List of neighbor node IDs
   :rtype: list
   :raises KeyError: If node doesn't exist

.. method:: Graph.predecessors(node_id)

   Get predecessors of a node (for directed graphs).

   :param node_id: Node ID
   :type node_id: str or int
   :returns: List of predecessor node IDs
   :rtype: list
   :raises ValueError: If graph is undirected
   :raises KeyError: If node doesn't exist

.. method:: Graph.successors(node_id)

   Get successors of a node (for directed graphs).

   :param node_id: Node ID
   :type node_id: str or int
   :returns: List of successor node IDs
   :rtype: list
   :raises ValueError: If graph is undirected
   :raises KeyError: If node doesn't exist

Graph Views
-----------

.. attribute:: Graph.nodes

   Access to graph nodes.

   :type: NodeView

   **Example:**

   .. code-block:: python

      # Iterate over nodes
      for node_id in g.nodes:
          print(node_id)

      # Get node attributes
      alice_data = g.nodes["alice"]

      # Get nodes as table
      nodes_table = g.nodes.table()

.. attribute:: Graph.edges

   Access to graph edges.

   :type: EdgeView

   **Example:**

   .. code-block:: python

      # Iterate over edges
      for source, target in g.edges:
          print(f"{source} -> {target}")

      # Get edges as table
      edges_table = g.edges.table()

Storage Views
-------------

.. method:: Graph.adjacency(**kwargs)

   Get adjacency matrix representation.

   :param kwargs: Optional parameters for matrix construction
   :returns: Adjacency matrix of the graph
   :rtype: GraphMatrix

   **Example:**

   .. code-block:: python

      adj = g.adjacency()
      print(f"Matrix shape: {adj.shape}")
      print(f"Is sparse: {adj.is_sparse}")

.. method:: Graph.table(entity_type="nodes", attributes=None)

   Get tabular representation of graph data.

   :param str entity_type: "nodes" or "edges"
   :param list attributes: Specific attributes to include, or None for all
   :returns: Tabular view of graph entities
   :rtype: GraphTable

   **Example:**

   .. code-block:: python

      # All node data
      nodes_table = g.table("nodes")

      # Specific node attributes
      subset = g.table("nodes", ["age", "role"])

      # Edge data
      edges_table = g.table("edges")

Filtering and Querying
----------------------

.. method:: Graph.filter_nodes(condition)

   Filter nodes by condition.

   :param condition: String expression or callable predicate
   :type condition: str or callable
   :returns: Subgraph containing matching nodes
   :rtype: Subgraph

   **Example:**

   .. code-block:: python

      # String-based filtering
      engineers = g.filter_nodes("role == 'engineer'")
      young_people = g.filter_nodes("age < 30")

      # Complex conditions
      young_engineers = g.filter_nodes("role == 'engineer' AND age < 35")

.. method:: Graph.filter_edges(condition)

   Filter edges by condition.

   :param condition: String expression or callable predicate
   :type condition: str or callable
   :returns: Subgraph containing matching edges
   :rtype: Subgraph

Subgraph Operations
-------------------

.. method:: Graph.subgraph(node_ids)

   Create subgraph from specific nodes.

   :param list node_ids: List of node IDs to include
   :returns: Subgraph containing specified nodes and their edges
   :rtype: Subgraph

   **Example:**

   .. code-block:: python

      core_team = g.subgraph(["alice", "bob", "charlie"])

Path Operations
---------------

.. method:: Graph.shortest_path(source, target)

   Find shortest path between nodes.

   :param source: Source node ID
   :type source: str or int
   :param target: Target node ID
   :type target: str or int
   :returns: List of node IDs forming the shortest path
   :rtype: list
   :raises ValueError: If no path exists

.. method:: Graph.has_path(source, target)

   Check if a path exists between nodes.

   :param source: Source node ID
   :type source: str or int
   :param target: Target node ID
   :type target: str or int
   :returns: True if path exists, False otherwise
   :rtype: bool

Traversal Operations
--------------------

.. method:: Graph.bfs(start_node, visitor=None)

   Breadth-first search traversal.

   :param start_node: Starting node ID
   :type start_node: str or int
   :param callable visitor: Optional visitor function
   :returns: List of visited node IDs in BFS order
   :rtype: list

.. method:: Graph.dfs(start_node, visitor=None)

   Depth-first search traversal.

   :param start_node: Starting node ID
   :type start_node: str or int
   :param callable visitor: Optional visitor function
   :returns: List of visited node IDs in DFS order
   :rtype: list

Connectivity Analysis
---------------------

.. method:: Graph.connected_components()

   Find connected components.

   :returns: List of connected components as Subgraph objects
   :rtype: list

   **Example:**

   .. code-block:: python

      components = g.connected_components()
      print(f"Number of components: {len(components)}")
      
      largest = max(components, key=lambda c: len(c.node_ids))
      print(f"Largest component: {len(largest.node_ids)} nodes")

Analytics Module
----------------

.. attribute:: Graph.analytics

   Access to basic graph algorithms and analysis functions.

   **Available Methods:**

   - ``connected_components()`` - Find connected components
   - ``shortest_path(source, target)`` - Find shortest path between nodes
   - ``has_path(source, target)`` - Check if path exists
   - ``bfs(start_node)`` - Breadth-first search traversal
   - ``dfs(start_node)`` - Depth-first search traversal

   **Example:**

   .. code-block:: python

      components = g.analytics.connected_components()
      path = g.analytics.shortest_path(node1, node2)

   .. note::
      Advanced centrality measures and community detection algorithms 
      will be available in future releases.

Utility Methods
---------------

.. method:: Graph.copy()

   Create a deep copy of the graph.

   :returns: New graph instance with copied data
   :rtype: Graph

.. method:: Graph.clear()

   Remove all nodes and edges from the graph.

.. method:: Graph.memory_usage()

   Get memory usage of the graph.

   :returns: Memory usage in bytes
   :rtype: int

.. method:: Graph.summary()

   Get summary information about the graph.

   :returns: Dictionary with graph statistics
   :rtype: dict

   **Example:**

   .. code-block:: python

      summary = g.summary()
      print(f"Nodes: {summary['node_count']}")
      print(f"Edges: {summary['edge_count']}")
      print(f"Density: {summary['density']:.3f}")

Export and Integration
----------------------

.. method:: Graph.to_networkx()

   Convert to NetworkX graph.

   :returns: NetworkX Graph or DiGraph object
   :rtype: networkx.Graph or networkx.DiGraph

.. method:: Graph.to_dict()

   Export graph as dictionary.

   :returns: Dictionary representation of the graph
   :rtype: dict

.. method:: Graph.save(filename, format="json")

   Save graph to file.

   :param str filename: Output filename
   :param str format: Output format ("json", "csv", "graphml")

.. method:: Graph.load(filename, format="json")

   Load graph from file.

   :param str filename: Input filename
   :param str format: Input format ("json", "csv", "graphml")
   :returns: New graph instance
   :rtype: Graph

This API reference covers the core Graph functionality. See the other API sections for detailed information about storage views, analytics, and utilities.