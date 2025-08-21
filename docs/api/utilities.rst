Utilities API
=============

Groggy provides various utility functions and classes to support graph analysis workflows.

Data Loading and Creation
-------------------------

Graph Builders
~~~~~~~~~~~~~~

.. function:: groggy.random_graph(n_nodes, edge_probability=0.1, directed=True, seed=None)

   Generate a random Erdős-Rényi graph.

   :param int n_nodes: Number of nodes
   :param float edge_probability: Probability of edge between any two nodes
   :param bool directed: Whether to create directed graph
   :param int seed: Random seed for reproducibility
   :returns: Random graph
   :rtype: Graph

   **Example:**

   .. code-block:: python

      import groggy as gr
      
      # Create random graph
      g = gr.random_graph(100, edge_probability=0.05, seed=42)

.. function:: groggy.scale_free_graph(n_nodes, attachment_rate=3, directed=True, seed=None)

   Generate a scale-free graph using preferential attachment.

   :param int n_nodes: Number of nodes
   :param int attachment_rate: Number of edges added per new node
   :param bool directed: Whether to create directed graph
   :param int seed: Random seed for reproducibility
   :returns: Scale-free graph
   :rtype: Graph

.. function:: groggy.small_world_graph(n_nodes, k=4, rewire_probability=0.1, seed=None)

   Generate a small-world graph using Watts-Strogatz model.

   :param int n_nodes: Number of nodes
   :param int k: Initial degree (must be even)
   :param float rewire_probability: Probability of rewiring each edge
   :param int seed: Random seed for reproducibility
   :returns: Small-world graph
   :rtype: Graph

.. function:: groggy.complete_graph(n_nodes, directed=False)

   Generate a complete graph.

   :param int n_nodes: Number of nodes
   :param bool directed: Whether to create directed graph
   :returns: Complete graph
   :rtype: Graph

.. function:: groggy.star_graph(n_nodes)

   Generate a star graph (one central node connected to all others).

   :param int n_nodes: Total number of nodes
   :returns: Star graph
   :rtype: Graph

.. function:: groggy.cycle_graph(n_nodes, directed=False)

   Generate a cycle graph.

   :param int n_nodes: Number of nodes
   :param bool directed: Whether to create directed graph
   :returns: Cycle graph
   :rtype: Graph

.. function:: groggy.path_graph(n_nodes, directed=False)

   Generate a path graph.

   :param int n_nodes: Number of nodes
   :param bool directed: Whether to create directed graph
   :returns: Path graph
   :rtype: Graph

Data Import/Export
~~~~~~~~~~~~~~~~~~

.. function:: groggy.load_graph(filename, format='auto')

   Load graph from file.

   :param str filename: Path to input file
   :param str format: File format ('auto', 'json', 'csv', 'graphml', 'gexf')
   :returns: Loaded graph
   :rtype: Graph

   **Example:**

   .. code-block:: python

      # Auto-detect format from extension
      g = gr.load_graph('network.json')
      
      # Specify format explicitly
      g = gr.load_graph('edges.csv', format='csv')

.. function:: groggy.save_graph(graph, filename, format='auto')

   Save graph to file.

   :param Graph graph: Graph to save
   :param str filename: Output filename
   :param str format: File format ('auto', 'json', 'csv', 'graphml', 'gexf')

.. function:: groggy.from_networkx(nx_graph)

   Convert NetworkX graph to Groggy graph.

   :param nx_graph: NetworkX graph object
   :returns: Groggy graph
   :rtype: Graph

.. function:: groggy.to_networkx(groggy_graph)

   Convert Groggy graph to NetworkX graph.

   :param Graph groggy_graph: Groggy graph object
   :returns: NetworkX graph

.. function:: groggy.from_pandas_edgelist(df, source='source', target='target', edge_attr=None)

   Create graph from pandas DataFrame edge list.

   :param pandas.DataFrame df: DataFrame with edge data
   :param str source: Column name for source nodes
   :param str target: Column name for target nodes
   :param list edge_attr: Column names for edge attributes
   :returns: Graph created from edge list
   :rtype: Graph

   **Example:**

   .. code-block:: python

      import pandas as pd
      
      df = pd.DataFrame({
          'from': ['A', 'B', 'C'],
          'to': ['B', 'C', 'A'],
          'weight': [1.0, 2.0, 1.5]
      })
      
      g = gr.from_pandas_edgelist(df, source='from', target='to', edge_attr=['weight'])

Subgraph Utilities
------------------

.. class:: Subgraph

   Represents a subgraph view of a larger graph.

   .. attribute:: node_ids

      List of node IDs in the subgraph.

      :type: list

   .. attribute:: parent_graph

      Reference to the parent graph.

      :type: Graph

   .. method:: induced_subgraph()

      Create an induced subgraph with all edges between nodes.

      :returns: New independent graph
      :rtype: Graph

   .. method:: edge_count()

      Number of edges in the subgraph.

      :returns: Edge count
      :rtype: int

   .. method:: density()

      Density of the subgraph.

      :returns: Density value
      :rtype: float

   .. method:: to_graph()

      Convert subgraph to independent graph.

      :returns: New graph with copied data
      :rtype: Graph

   **Example:**

   .. code-block:: python

      # Create subgraph
      subgraph = g.subgraph(['alice', 'bob', 'charlie'])
      
      # Analyze subgraph
      print(f"Subgraph has {len(subgraph.node_ids)} nodes")
      print(f"Subgraph density: {subgraph.density():.3f}")
      
      # Convert to independent graph
      sub_g = subgraph.to_graph()

Graph Views
-----------

.. class:: NodeView

   Provides access to graph nodes.

   .. method:: __iter__()

      Iterate over node IDs.

      :returns: Iterator of node IDs

   .. method:: __len__()

      Number of nodes.

      :returns: Node count
      :rtype: int

   .. method:: __getitem__(node_id)

      Get node attributes.

      :param node_id: Node identifier
      :returns: Node attributes
      :rtype: dict

   .. method:: table(attributes=None)

      Get nodes as table.

      :param list attributes: Specific attributes to include
      :returns: Node data as table
      :rtype: GraphTable

   .. method:: sample(n=None, frac=None, seed=None)

      Random sample of nodes.

      :param int n: Number of nodes to sample
      :param float frac: Fraction of nodes to sample
      :param int seed: Random seed
      :returns: List of sampled node IDs
      :rtype: list

   .. method:: filter(predicate)

      Filter nodes by predicate.

      :param callable predicate: Function taking node_id and returning bool
      :returns: List of filtered node IDs
      :rtype: list

   **Example:**

   .. code-block:: python

      # Iterate over nodes
      for node_id in g.nodes:
          print(f"Node: {node_id}")
      
      # Get node attributes
      alice_attrs = g.nodes['alice']
      
      # Get as table
      nodes_table = g.nodes.table()
      
      # Sample nodes
      sample_nodes = g.nodes.sample(n=10, seed=42)

.. class:: EdgeView

   Provides access to graph edges.

   .. method:: __iter__()

      Iterate over edge tuples.

      :returns: Iterator of (source, target) tuples

   .. method:: __len__()

      Number of edges.

      :returns: Edge count
      :rtype: int

   .. method:: __getitem__(edge)

      Get edge attributes.

      :param tuple edge: (source, target) tuple
      :returns: Edge attributes
      :rtype: dict

   .. method:: table(attributes=None)

      Get edges as table.

      :param list attributes: Specific attributes to include
      :returns: Edge data as table
      :rtype: GraphTable

   .. method:: sample(n=None, frac=None, seed=None)

      Random sample of edges.

      :param int n: Number of edges to sample
      :param float frac: Fraction of edges to sample
      :param int seed: Random seed
      :returns: List of sampled (source, target) tuples
      :rtype: list

   .. method:: filter(predicate)

      Filter edges by predicate.

      :param callable predicate: Function taking (source, target) and returning bool
      :returns: List of filtered (source, target) tuples
      :rtype: list

Validation and Testing
----------------------

.. function:: groggy.is_valid_graph(graph)

   Check if graph is in a valid state.

   :param Graph graph: Graph to validate
   :returns: True if valid, False otherwise
   :rtype: bool

.. function:: groggy.graph_info(graph)

   Get comprehensive information about a graph.

   :param Graph graph: Graph to analyze
   :returns: Dictionary with graph statistics
   :rtype: dict

   **Example:**

   .. code-block:: python

      info = gr.graph_info(g)
      print(f"Nodes: {info['node_count']}")
      print(f"Edges: {info['edge_count']}")
      print(f"Density: {info['density']:.3f}")
      print(f"Connected: {info['is_connected']}")

.. function:: groggy.assert_graph_equal(graph1, graph2, check_attributes=True)

   Assert two graphs are equal (useful for testing).

   :param Graph graph1: First graph
   :param Graph graph2: Second graph
   :param bool check_attributes: Whether to compare attributes
   :raises AssertionError: If graphs are not equal

Performance Utilities
---------------------

.. function:: groggy.benchmark_operation(operation, *args, iterations=10, **kwargs)

   Benchmark a graph operation.

   :param callable operation: Function to benchmark
   :param args: Positional arguments for operation
   :param int iterations: Number of iterations to run
   :param kwargs: Keyword arguments for operation
   :returns: Dictionary with timing statistics
   :rtype: dict

   **Example:**

   .. code-block:: python

      # Benchmark PageRank
      stats = gr.benchmark_operation(g.centrality.pagerank, iterations=5)
      print(f"Average time: {stats['mean']:.3f}s")
      print(f"Standard deviation: {stats['std']:.3f}s")

.. function:: groggy.memory_usage(graph)

   Get memory usage of a graph.

   :param Graph graph: Graph to analyze
   :returns: Memory usage in bytes
   :rtype: int

.. function:: groggy.profile_graph_operation(operation, *args, **kwargs)

   Profile a graph operation for performance analysis.

   :param callable operation: Function to profile
   :param args: Positional arguments
   :param kwargs: Keyword arguments
   :returns: Profiling results
   :rtype: dict

Configuration and Settings
--------------------------

.. function:: groggy.set_parallel_threads(n_threads)

   Set number of threads for parallel operations.

   :param int n_threads: Number of threads (0 for auto-detect)

.. function:: groggy.get_parallel_threads()

   Get current thread count setting.

   :returns: Number of threads
   :rtype: int

.. function:: groggy.set_memory_limit(limit_mb)

   Set memory limit for operations.

   :param int limit_mb: Memory limit in megabytes

.. function:: groggy.enable_progress_bars(enabled=True)

   Enable/disable progress bars for long operations.

   :param bool enabled: Whether to show progress bars

.. function:: groggy.set_random_seed(seed)

   Set global random seed for reproducibility.

   :param int seed: Random seed value

Data Conversion Utilities
-------------------------

.. function:: groggy.normalize_node_ids(graph, mapping=None)

   Normalize node IDs to integers or strings.

   :param Graph graph: Graph to normalize
   :param dict mapping: Custom ID mapping (optional)
   :returns: New graph with normalized IDs and mapping
   :rtype: tuple[Graph, dict]

.. function:: groggy.relabel_nodes(graph, mapping)

   Relabel nodes according to mapping.

   :param Graph graph: Graph to relabel
   :param dict mapping: Mapping from old to new node IDs
   :returns: New graph with relabeled nodes
   :rtype: Graph

.. function:: groggy.largest_connected_component(graph)

   Extract the largest connected component.

   :param Graph graph: Input graph
   :returns: Subgraph of largest component
   :rtype: Subgraph

.. function:: groggy.remove_self_loops(graph)

   Remove self-loops from graph.

   :param Graph graph: Input graph
   :returns: New graph without self-loops
   :rtype: Graph

.. function:: groggy.remove_multi_edges(graph, keep='first')

   Remove multiple edges between same nodes.

   :param Graph graph: Input graph
   :param str keep: Which edge to keep ('first', 'last', 'max_weight')
   :returns: New graph without multi-edges
   :rtype: Graph

Graph Comparison
---------------

.. function:: groggy.graph_similarity(graph1, graph2, method='jaccard')

   Calculate similarity between two graphs.

   :param Graph graph1: First graph
   :param Graph graph2: Second graph
   :param str method: Similarity method ('jaccard', 'overlap', 'cosine')
   :returns: Similarity score (0 to 1)
   :rtype: float

.. function:: groggy.structural_similarity(graph1, graph2)

   Calculate structural similarity using graph features.

   :param Graph graph1: First graph
   :param Graph graph2: Second graph
   :returns: Structural similarity metrics
   :rtype: dict

Sampling Utilities
------------------

.. function:: groggy.random_walk_sample(graph, start_node, length)

   Perform random walk sampling.

   :param Graph graph: Graph to sample from
   :param start_node: Starting node
   :param int length: Length of walk
   :returns: List of visited nodes
   :rtype: list

.. function:: groggy.snowball_sample(graph, seed_nodes, k_steps)

   Perform snowball sampling.

   :param Graph graph: Graph to sample from
   :param list seed_nodes: Initial seed nodes
   :param int k_steps: Number of expansion steps
   :returns: Sampled subgraph
   :rtype: Subgraph

.. function:: groggy.degree_based_sample(graph, n_nodes, method='high_degree')

   Sample nodes based on degree.

   :param Graph graph: Graph to sample from
   :param int n_nodes: Number of nodes to sample
   :param str method: Sampling method ('high_degree', 'low_degree', 'proportional')
   :returns: List of sampled node IDs
   :rtype: list

Error Handling Utilities
------------------------

.. exception:: GraphError

   Base exception for graph-related errors.

.. exception:: NodeNotFoundError(GraphError)

   Raised when a node is not found in the graph.

.. exception:: EdgeNotFoundError(GraphError)

   Raised when an edge is not found in the graph.

.. exception:: InvalidGraphError(GraphError)

   Raised when graph is in an invalid state.

.. exception:: IncompatibleGraphError(GraphError)

   Raised when graphs are incompatible for an operation.

Context Managers
---------------

.. class:: temporary_graph_changes

   Context manager for temporary graph modifications.

   **Example:**

   .. code-block:: python

      with gr.temporary_graph_changes(g):
          # Add temporary nodes/edges
          g.add_node('temp_node')
          g.add_edge('alice', 'temp_node')
          
          # Analyze with temporary changes
          centrality = g.centrality.pagerank()
      
      # Changes are automatically reverted
      assert not g.has_node('temp_node')

.. class:: graph_transaction

   Context manager for atomic graph operations.

   **Example:**

   .. code-block:: python

      try:
          with gr.graph_transaction(g):
              g.add_nodes(large_node_list)
              g.add_edges(large_edge_list)
              # All changes committed together
      except Exception:
          # All changes rolled back on error
          pass

Debugging and Inspection
-----------------------

.. function:: groggy.debug_graph(graph, check_integrity=True)

   Debug and inspect graph for issues.

   :param Graph graph: Graph to debug
   :param bool check_integrity: Whether to run integrity checks
   :returns: Debug report
   :rtype: dict

.. function:: groggy.graph_statistics(graph, detailed=False)

   Calculate comprehensive graph statistics.

   :param Graph graph: Graph to analyze
   :param bool detailed: Include detailed statistics
   :returns: Statistics dictionary
   :rtype: dict

.. function:: groggy.find_problematic_nodes(graph)

   Find nodes that might cause issues (isolated, high degree, etc.).

   :param Graph graph: Graph to analyze
   :returns: Dictionary categorizing problematic nodes
   :rtype: dict

Best Practices
--------------

1. **Use appropriate graph generators** for testing and benchmarking
2. **Validate graphs** after major operations or imports
3. **Set memory limits** for large graph operations
4. **Use subgraphs** for focused analysis on parts of large graphs
5. **Profile operations** to identify performance bottlenecks
6. **Handle exceptions** appropriately in production code

**Example comprehensive workflow:**

.. code-block:: python

   import groggy as gr
   
   # Set up environment
   gr.set_parallel_threads(4)
   gr.set_random_seed(42)
   gr.enable_progress_bars(True)
   
   # Load and validate graph
   g = gr.load_graph('network.json')
   assert gr.is_valid_graph(g)
   
   # Get basic info
   info = gr.graph_info(g)
   print(f"Loaded graph: {info['node_count']} nodes, {info['edge_count']} edges")
   
   # Clean graph if needed
   if not info['is_connected']:
       g = gr.largest_connected_component(g).to_graph()
   
   # Remove problematic elements
   g = gr.remove_self_loops(g)
   g = gr.remove_multi_edges(g)
   
   # Analyze with benchmarking
   stats = gr.benchmark_operation(g.centrality.pagerank)
   print(f"PageRank took {stats['mean']:.3f}s on average")
   
   # Save results
   gr.save_graph(g, 'cleaned_network.json')

The utilities API provides essential tools for graph manipulation, analysis, and debugging in Groggy workflows.