Analytics API
=============

Groggy provides comprehensive graph analytics through specialized modules accessible via the Graph class.

Graph.centrality
-----------------

Centrality algorithms identify the most important nodes in a network.

.. attribute:: Graph.centrality

   Access to centrality algorithms.

   **Available Methods:**

Betweenness Centrality
~~~~~~~~~~~~~~~~~~~~~~

.. method:: Graph.centrality.betweenness(normalized=True, endpoints=False)

   Calculate betweenness centrality for all nodes.

   :param bool normalized: Normalize by maximum possible betweenness
   :param bool endpoints: Include endpoints in path counts
   :returns: Dictionary mapping node_id -> centrality score
   :rtype: dict

   **Example:**

   .. code-block:: python

      betweenness = g.centrality.betweenness()
      top_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5]

.. method:: Graph.centrality.betweenness_approximate(k=None, normalized=True)

   Approximate betweenness centrality using sampling.

   :param int k: Number of nodes to sample (default: sqrt(n))
   :param bool normalized: Normalize by maximum possible betweenness
   :returns: Dictionary mapping node_id -> centrality score
   :rtype: dict

Closeness Centrality
~~~~~~~~~~~~~~~~~~~~~

.. method:: Graph.centrality.closeness(normalized=True)

   Calculate closeness centrality for all nodes.

   :param bool normalized: Normalize by maximum possible closeness
   :returns: Dictionary mapping node_id -> centrality score
   :rtype: dict

   **Example:**

   .. code-block:: python

      closeness = g.centrality.closeness()
      most_central = max(closeness.items(), key=lambda x: x[1])

PageRank
~~~~~~~~

.. method:: Graph.centrality.pagerank(alpha=0.85, max_iter=100, tolerance=1e-6, personalization=None)

   Calculate PageRank for all nodes.

   :param float alpha: Damping parameter (probability of following link)
   :param int max_iter: Maximum number of iterations
   :param float tolerance: Convergence tolerance
   :param dict personalization: Personalization vector (optional)
   :returns: Dictionary mapping node_id -> PageRank score
   :rtype: dict

   **Example:**

   .. code-block:: python

      pagerank = g.centrality.pagerank(alpha=0.85)
      
      # Personalized PageRank
      personalized = g.centrality.pagerank(
          personalization={'alice': 0.5, 'bob': 0.5}
      )

Eigenvector Centrality
~~~~~~~~~~~~~~~~~~~~~~

.. method:: Graph.centrality.eigenvector(max_iter=100, tolerance=1e-6)

   Calculate eigenvector centrality for all nodes.

   :param int max_iter: Maximum number of iterations
   :param float tolerance: Convergence tolerance
   :returns: Dictionary mapping node_id -> centrality score
   :rtype: dict

   **Example:**

   .. code-block:: python

      eigenvector = g.centrality.eigenvector()

Degree Centrality
~~~~~~~~~~~~~~~~~

.. method:: Graph.centrality.degree(normalized=True)

   Calculate degree centrality for all nodes.

   :param bool normalized: Normalize by maximum possible degree
   :returns: Dictionary mapping node_id -> centrality score
   :rtype: dict

.. method:: Graph.centrality.in_degree(normalized=True)

   Calculate in-degree centrality (directed graphs only).

   :param bool normalized: Normalize by maximum possible in-degree
   :returns: Dictionary mapping node_id -> centrality score
   :rtype: dict

.. method:: Graph.centrality.out_degree(normalized=True)

   Calculate out-degree centrality (directed graphs only).

   :param bool normalized: Normalize by maximum possible out-degree
   :returns: Dictionary mapping node_id -> centrality score
   :rtype: dict

Graph.communities
------------------

Community detection algorithms identify groups of densely connected nodes.

.. attribute:: Graph.communities

   Access to community detection algorithms.

   **Available Methods:**

Louvain Algorithm
~~~~~~~~~~~~~~~~~

.. method:: Graph.communities.louvain(resolution=1.0, max_iter=100, random_state=None)

   Detect communities using the Louvain algorithm.

   :param float resolution: Resolution parameter (higher = more communities)
   :param int max_iter: Maximum number of iterations
   :param int random_state: Random seed for reproducibility
   :returns: List of communities (each community is a list of node IDs)
   :rtype: list

   **Example:**

   .. code-block:: python

      communities = g.communities.louvain(resolution=1.0)
      print(f"Found {len(communities)} communities")
      
      # Analyze community sizes
      sizes = [len(c) for c in communities]
      print(f"Largest community: {max(sizes)} nodes")

Leiden Algorithm
~~~~~~~~~~~~~~~~

.. method:: Graph.communities.leiden(resolution=1.0, max_iter=100, random_state=None)

   Detect communities using the Leiden algorithm.

   :param float resolution: Resolution parameter (higher = more communities)
   :param int max_iter: Maximum number of iterations
   :param int random_state: Random seed for reproducibility
   :returns: List of communities (each community is a list of node IDs)
   :rtype: list

   **Example:**

   .. code-block:: python

      leiden_communities = g.communities.leiden(resolution=1.0)

Modularity
~~~~~~~~~~

.. method:: Graph.communities.modularity(communities, resolution=1.0)

   Calculate modularity of a community partition.

   :param list communities: List of communities to evaluate
   :param float resolution: Resolution parameter
   :returns: Modularity score (-1 to 1, higher is better)
   :rtype: float

   **Example:**

   .. code-block:: python

      communities = g.communities.louvain()
      modularity = g.communities.modularity(communities)
      print(f"Modularity: {modularity:.3f}")

Label Propagation
~~~~~~~~~~~~~~~~~

.. method:: Graph.communities.label_propagation(max_iter=100, random_state=None)

   Detect communities using label propagation.

   :param int max_iter: Maximum number of iterations
   :param int random_state: Random seed for reproducibility
   :returns: List of communities
   :rtype: list

Graph.clustering
-----------------

Clustering coefficient and transitivity measures.

.. method:: Graph.clustering(nodes=None)

   Calculate average clustering coefficient.

   :param list nodes: Specific nodes to analyze (default: all)
   :returns: Average clustering coefficient
   :rtype: float

   **Example:**

   .. code-block:: python

      avg_clustering = g.clustering()
      print(f"Average clustering: {avg_clustering:.3f}")

.. method:: Graph.local_clustering(nodes=None)

   Calculate local clustering coefficient for each node.

   :param list nodes: Specific nodes to analyze (default: all)
   :returns: Dictionary mapping node_id -> clustering coefficient
   :rtype: dict

   **Example:**

   .. code-block:: python

      local_clustering = g.local_clustering()
      high_clustering = {node: c for node, c in local_clustering.items() if c > 0.5}

.. method:: Graph.transitivity()

   Calculate global transitivity (ratio of closed triplets).

   :returns: Transitivity value
   :rtype: float

Graph.assortativity
-------------------

Assortativity measures the tendency of nodes to connect to similar nodes.

.. attribute:: Graph.assortativity

   Access to assortativity algorithms.

   **Available Methods:**

Degree Assortativity
~~~~~~~~~~~~~~~~~~~~

.. method:: Graph.assortativity.degree()

   Calculate degree assortativity coefficient.

   :returns: Assortativity coefficient (-1 to 1)
   :rtype: float

   **Example:**

   .. code-block:: python

      degree_assort = g.assortativity.degree()
      if degree_assort > 0:
          print("High-degree nodes tend to connect to high-degree nodes")
      else:
          print("High-degree nodes tend to connect to low-degree nodes")

Attribute Assortativity
~~~~~~~~~~~~~~~~~~~~~~~

.. method:: Graph.assortativity.attribute(attribute_name)

   Calculate assortativity for a categorical attribute.

   :param str attribute_name: Node attribute to analyze
   :returns: Assortativity coefficient (-1 to 1)
   :rtype: float

   **Example:**

   .. code-block:: python

      dept_assort = g.assortativity.attribute('department')
      print(f"Department assortativity: {dept_assort:.3f}")

.. method:: Graph.assortativity.numeric_attribute(attribute_name)

   Calculate assortativity for a numeric attribute.

   :param str attribute_name: Node attribute to analyze
   :returns: Assortativity coefficient (-1 to 1)
   :rtype: float

Graph.motifs
------------

Motif analysis for small subgraph patterns.

.. attribute:: Graph.motifs

   Access to motif analysis algorithms.

   **Available Methods:**

Triangle Analysis
~~~~~~~~~~~~~~~~~

.. method:: Graph.triangles()

   Count total number of triangles in the graph.

   :returns: Number of triangles
   :rtype: int

.. method:: Graph.triangle_participation()

   Count triangles each node participates in.

   :returns: Dictionary mapping node_id -> triangle count
   :rtype: dict

   **Example:**

   .. code-block:: python

      triangles = g.triangles()
      participation = g.triangle_participation()
      
      avg_participation = sum(participation.values()) / len(participation)
      print(f"Total triangles: {triangles}")
      print(f"Average participation: {avg_participation:.1f}")

.. method:: Graph.motifs.count_motifs(size=3)

   Count motifs of specified size.

   :param int size: Motif size (3 or 4)
   :returns: Dictionary mapping motif_type -> count
   :rtype: dict

.. method:: Graph.motifs.node_motif_participation(node_id, size=3)

   Count motifs a specific node participates in.

   :param node_id: Node to analyze
   :param int size: Motif size (3 or 4)
   :returns: Dictionary mapping motif_type -> count
   :rtype: dict

Graph.structural
-----------------

Structural analysis algorithms.

.. attribute:: Graph.structural

   Access to structural analysis algorithms.

   **Available Methods:**

Core Decomposition
~~~~~~~~~~~~~~~~~~

.. method:: Graph.k_core_decomposition()

   Perform k-core decomposition.

   :returns: Dictionary mapping node_id -> core number
   :rtype: dict

   **Example:**

   .. code-block:: python

      k_cores = g.k_core_decomposition()
      max_k = max(k_cores.values())
      main_core = [node for node, k in k_cores.items() if k == max_k]

.. method:: Graph.k_core(k)

   Extract k-core subgraph.

   :param int k: Core number
   :returns: Subgraph containing k-core
   :rtype: Subgraph

Bridges and Articulation
~~~~~~~~~~~~~~~~~~~~~~~~~

.. method:: Graph.bridges()

   Find bridges (edges whose removal increases components).

   :returns: List of bridge edges as (source, target) tuples
   :rtype: list

.. method:: Graph.articulation_points()

   Find articulation points (nodes whose removal increases components).

   :returns: List of articulation point node IDs
   :rtype: list

   **Example:**

   .. code-block:: python

      bridges = g.bridges()
      articulation = g.articulation_points()
      
      print(f"Critical edges: {len(bridges)}")
      print(f"Critical nodes: {len(articulation)}")

Rich Club Analysis
~~~~~~~~~~~~~~~~~~

.. method:: Graph.structural.rich_club_coefficient(k=None)

   Calculate rich club coefficient.

   :param int k: Degree threshold (default: all degrees)
   :returns: Rich club coefficient or dict for all k values
   :rtype: float or dict

Graph.paths
-----------

Path analysis algorithms.

.. attribute:: Graph.paths

   Access to path analysis algorithms.

   **Available Methods:**

Path Statistics
~~~~~~~~~~~~~~~

.. method:: Graph.diameter()

   Calculate graph diameter (longest shortest path).

   :returns: Diameter value
   :rtype: int
   :raises ValueError: If graph is not connected

.. method:: Graph.average_path_length()

   Calculate average shortest path length.

   :returns: Average path length
   :rtype: float

.. method:: Graph.eccentricity(node_id=None)

   Calculate eccentricity (maximum distance to any other node).

   :param node_id: Specific node, or None for all nodes
   :returns: Eccentricity value or dictionary
   :rtype: int or dict

   **Example:**

   .. code-block:: python

      try:
          diameter = g.diameter()
          avg_path = g.average_path_length()
          print(f"Diameter: {diameter}")
          print(f"Average path length: {avg_path:.2f}")
      except ValueError:
          print("Graph is not connected")

Efficiency Measures
~~~~~~~~~~~~~~~~~~~

.. method:: Graph.paths.efficiency(global_=True)

   Calculate efficiency measures.

   :param bool global_: Calculate global efficiency (vs local)
   :returns: Efficiency value
   :rtype: float

.. method:: Graph.paths.node_efficiency()

   Calculate efficiency for each node.

   :returns: Dictionary mapping node_id -> efficiency
   :rtype: dict

Performance and Scaling
------------------------

Large Graph Approximations
~~~~~~~~~~~~~~~~~~~~~~~~~~~

For very large graphs, many algorithms provide approximation methods:

.. code-block:: python

   # Approximate algorithms for large graphs
   if g.node_count() > 100000:
       # Use sampling-based approximations
       betweenness = g.centrality.betweenness_approximate(k=1000)
       communities = g.communities.louvain()  # Still exact and efficient
   else:
       # Use exact algorithms
       betweenness = g.centrality.betweenness()
       communities = g.communities.leiden()

Parallel Processing
~~~~~~~~~~~~~~~~~~~

Some algorithms support parallel execution:

.. code-block:: python

   # Enable parallel processing for supported algorithms
   pagerank = g.centrality.pagerank(parallel=True)
   communities = g.communities.louvain(parallel=True)

Memory Optimization
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Memory-efficient analysis for large graphs
   def analyze_large_graph(g):
       # Use generators for memory efficiency
       components = g.connected_components()
       
       # Analyze largest component only
       largest = max(components, key=lambda c: len(c.node_ids))
       largest_subgraph = g.subgraph(largest.node_ids)
       
       # Run expensive algorithms on largest component
       centrality = largest_subgraph.centrality.betweenness()
       communities = largest_subgraph.communities.louvain()
       
       return centrality, communities

Integration with Storage Views
------------------------------

Analytics results integrate seamlessly with storage views:

.. code-block:: python

   # Calculate multiple centrality measures
   betweenness = g.centrality.betweenness()
   pagerank = g.centrality.pagerank()
   eigenvector = g.centrality.eigenvector()
   
   # Create analysis table
   nodes_table = g.nodes.table()
   
   # Add centrality as columns
   centrality_data = []
   for node_id in nodes_table['node_id']:
       centrality_data.append({
           'node_id': node_id,
           'betweenness': betweenness.get(node_id, 0),
           'pagerank': pagerank.get(node_id, 0),
           'eigenvector': eigenvector.get(node_id, 0)
       })
   
   centrality_table = gr.table(centrality_data)
   enriched = nodes_table.join(centrality_table, on='node_id')
   
   # Statistical analysis
   centrality_stats = enriched[['betweenness', 'pagerank', 'eigenvector']].describe()

Error Handling
--------------

Analytics algorithms provide clear error messages:

.. code-block:: python

   try:
       # This might fail for disconnected graphs
       diameter = g.diameter()
   except ValueError as e:
       print(f"Cannot compute diameter: {e}")
   
   try:
       # This might fail for undirected graphs
       in_degree = g.centrality.in_degree()
   except ValueError as e:
       print(f"In-degree not available: {e}")

Best Practices
--------------

1. **Start with basic measures** - degree, clustering before advanced centrality
2. **Check graph properties** - connectedness, directedness before running algorithms
3. **Use approximations for large graphs** - sampling-based methods for graphs > 100k nodes
4. **Combine multiple measures** - no single centrality captures all aspects of importance
5. **Validate results** - check if algorithmic results make sense for your domain
6. **Consider graph topology** - some algorithms work better on specific graph types

**Example comprehensive analysis:**

.. code-block:: python

   def comprehensive_analysis(g):
       results = {}
       
       # Basic properties
       results['node_count'] = g.node_count()
       results['edge_count'] = g.edge_count()
       results['density'] = g.density()
       results['is_connected'] = g.is_connected()
       
       # Centrality measures
       results['degree_centrality'] = g.centrality.degree()
       results['pagerank'] = g.centrality.pagerank()
       
       if g.node_count() < 10000:  # Only for smaller graphs
           results['betweenness'] = g.centrality.betweenness()
       
       # Community structure
       results['communities'] = g.communities.louvain()
       results['modularity'] = g.communities.modularity(results['communities'])
       
       # Clustering
       results['clustering'] = g.clustering()
       results['transitivity'] = g.transitivity()
       
       # Structural properties
       if g.is_connected():
           results['diameter'] = g.diameter()
           results['avg_path_length'] = g.average_path_length()
       
       return results

The analytics API provides powerful tools for understanding network structure and dynamics. Results integrate seamlessly with storage views for further analysis and visualization.