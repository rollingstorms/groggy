Graph Analytics
===============

Groggy provides a comprehensive suite of graph algorithms and analytical tools for understanding network structure and dynamics.

Graph Algorithms
----------------

Connectivity Analysis
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import groggy as gr

   g = gr.Graph()
   # ... build your graph ...

   # Find connected components
   components = g.connected_components()
   print(f"Number of components: {len(components)}")

   # Get the largest component
   largest = max(components, key=lambda c: len(c.node_ids))
   print(f"Largest component: {len(largest.node_ids)} nodes")

   # Check if graph is connected
   is_connected = g.is_connected()
   print(f"Graph is connected: {is_connected}")

Path Finding
~~~~~~~~~~~~

.. code-block:: python

   # Find shortest path between nodes
   try:
       path = g.shortest_path("alice", "bob")
       print(f"Shortest path: {' -> '.join(path)}")
       print(f"Path length: {len(path) - 1}")
   except ValueError:
       print("No path exists between alice and bob")

   # Check if path exists
   has_path = g.has_path("alice", "bob")

   # Find all simple paths (limited length)
   all_paths = g.all_simple_paths("alice", "bob", max_length=5)
   print(f"Found {len(all_paths)} simple paths")

Graph Traversal
~~~~~~~~~~~~~~~

.. code-block:: python

   # Breadth-first search
   bfs_order = g.bfs(start_node="alice")
   print(f"BFS traversal: {bfs_order}")

   # Depth-first search  
   dfs_order = g.dfs(start_node="alice")
   print(f"DFS traversal: {dfs_order}")

   # Custom traversal with visitor pattern
   visited_nodes = []
   def visitor(node_id):
       visited_nodes.append(node_id)
       return True  # Continue traversal

   g.bfs(start_node="alice", visitor=visitor)

Centrality Measures
-------------------

Centrality measures identify the most important or influential nodes in a network.

Degree Centrality
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic degree centrality
   degrees = g.degree()
   
   # Normalize by maximum possible degree
   n = g.node_count()
   normalized_degrees = {node: deg/(n-1) for node, deg in degrees.items()}

   # For directed graphs
   if g.directed:
       in_degrees = g.in_degree()
       out_degrees = g.out_degree()

Betweenness Centrality
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate betweenness centrality
   betweenness = g.centrality.betweenness()
   
   # Find most central nodes
   sorted_nodes = sorted(betweenness.items(), key=lambda x: x[1], reverse=True)
   print("Most central nodes (betweenness):")
   for node, centrality in sorted_nodes[:5]:
       print(f"  {node}: {centrality:.3f}")

Closeness Centrality
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate closeness centrality
   closeness = g.centrality.closeness()
   
   # Analyze results
   max_closeness = max(closeness.values())
   most_central = [node for node, c in closeness.items() if c == max_closeness]
   print(f"Most central nodes (closeness): {most_central}")

PageRank
~~~~~~~~

.. code-block:: python

   # Calculate PageRank
   pagerank = g.centrality.pagerank(alpha=0.85, max_iter=100, tolerance=1e-6)
   
   # Find top-ranked nodes
   top_nodes = sorted(pagerank.items(), key=lambda x: x[1], reverse=True)[:10]
   print("Top PageRank nodes:")
   for node, score in top_nodes:
       print(f"  {node}: {score:.4f}")

Eigenvector Centrality
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate eigenvector centrality
   eigenvector = g.centrality.eigenvector(max_iter=100, tolerance=1e-6)
   
   # Compare with other centrality measures
   import pandas as pd
   centrality_df = pd.DataFrame({
       'betweenness': betweenness,
       'closeness': closeness,
       'pagerank': pagerank,
       'eigenvector': eigenvector
   })
   
   print(centrality_df.corr())

Community Detection
-------------------

Community detection algorithms identify groups of densely connected nodes.

Louvain Algorithm
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find communities using Louvain algorithm
   communities = g.communities.louvain(resolution=1.0)
   
   print(f"Found {len(communities)} communities")
   for i, community in enumerate(communities):
       print(f"Community {i}: {len(community)} nodes")
       
   # Analyze community sizes
   sizes = [len(community) for community in communities]
   print(f"Average community size: {sum(sizes) / len(sizes):.1f}")
   print(f"Largest community: {max(sizes)} nodes")

Modularity
~~~~~~~~~~

.. code-block:: python

   # Calculate modularity of detected communities
   modularity = g.communities.modularity(communities)
   print(f"Modularity: {modularity:.3f}")
   
   # Compare different resolutions
   resolutions = [0.5, 1.0, 1.5, 2.0]
   for res in resolutions:
       comms = g.communities.louvain(resolution=res)
       mod = g.communities.modularity(comms)
       print(f"Resolution {res}: {len(comms)} communities, modularity {mod:.3f}")

Leiden Algorithm
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use Leiden algorithm for higher quality communities
   leiden_communities = g.communities.leiden(resolution=1.0)
   leiden_modularity = g.communities.modularity(leiden_communities)
   
   print(f"Leiden: {len(leiden_communities)} communities, modularity {leiden_modularity:.3f}")

Network Properties
------------------

Density and Clustering
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Graph density
   density = g.density()
   print(f"Graph density: {density:.3f}")
   
   # Clustering coefficient
   clustering = g.clustering()
   print(f"Average clustering: {clustering:.3f}")
   
   # Local clustering for each node
   local_clustering = g.local_clustering()
   high_clustering = {node: c for node, c in local_clustering.items() if c > 0.5}

Assortativity
~~~~~~~~~~~~~

.. code-block:: python

   # Degree assortativity
   degree_assortativity = g.assortativity.degree()
   print(f"Degree assortativity: {degree_assortativity:.3f}")
   
   # Attribute assortativity
   if 'department' in g.nodes.table().columns:
       dept_assortativity = g.assortativity.attribute('department')
       print(f"Department assortativity: {dept_assortativity:.3f}")

Diameter and Paths
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Graph diameter (longest shortest path)
   try:
       diameter = g.diameter()
       print(f"Graph diameter: {diameter}")
   except ValueError:
       print("Graph is not connected - no diameter")
       
   # Average path length
   avg_path_length = g.average_path_length()
   print(f"Average path length: {avg_path_length:.2f}")

Structural Analysis
-------------------

Bridges and Articulation Points
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find bridges (edges whose removal increases components)
   bridges = g.bridges()
   print(f"Number of bridges: {len(bridges)}")
   
   # Find articulation points (nodes whose removal increases components)
   articulation_points = g.articulation_points()
   print(f"Number of articulation points: {len(articulation_points)}")

Core Decomposition
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # K-core decomposition
   k_cores = g.k_core_decomposition()
   max_k = max(k_cores.values())
   print(f"Maximum k-core: {max_k}")
   
   # Nodes in the main core
   main_core_nodes = [node for node, k in k_cores.items() if k == max_k]
   print(f"Main core size: {len(main_core_nodes)}")

Motif Analysis
~~~~~~~~~~~~~~

.. code-block:: python

   # Count triangles
   triangles = g.triangles()
   print(f"Number of triangles: {triangles}")
   
   # Triangle participation per node
   triangle_participation = g.triangle_participation()
   high_participation = {node: t for node, t in triangle_participation.items() if t > 5}

Advanced Analytics
------------------

Multi-Layer Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze graph at different scales
   # 1. Node level
   node_metrics = {
       'degree': g.degree(),
       'clustering': g.local_clustering(),
       'betweenness': g.centrality.betweenness()
   }
   
   # 2. Community level
   communities = g.communities.louvain()
   community_sizes = [len(c) for c in communities]
   
   # 3. Global level
   global_metrics = {
       'density': g.density(),
       'clustering': g.clustering(),
       'modularity': g.communities.modularity(communities)
   }

Temporal Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # If your graph has temporal information
   if 'timestamp' in g.edges.table().columns:
       # Analyze edge creation over time
       edges_table = g.edges.table()
       
       # Group by time periods
       temporal_analysis = edges_table.group_by('timestamp').agg({
           'weight': ['mean', 'count'],
           'source': 'nunique',
           'target': 'nunique'
       })
       
       print("Temporal edge patterns:")
       print(temporal_analysis)

Similarity and Recommendation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Node similarity based on common neighbors
   def jaccard_similarity(g, node1, node2):
       neighbors1 = set(g.neighbors(node1))
       neighbors2 = set(g.neighbors(node2))
       
       intersection = len(neighbors1 & neighbors2)
       union = len(neighbors1 | neighbors2)
       
       return intersection / union if union > 0 else 0

   # Find similar nodes to Alice
   alice_similarities = {}
   for node in g.nodes:
       if node != 'alice':
           sim = jaccard_similarity(g, 'alice', node)
           if sim > 0:
               alice_similarities[node] = sim
   
   # Top similar nodes
   similar = sorted(alice_similarities.items(), key=lambda x: x[1], reverse=True)[:5]

Integration with Storage Views
------------------------------

Combining Graph Algorithms with Tabular Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate centrality measures
   betweenness = g.centrality.betweenness()
   pagerank = g.centrality.pagerank()
   
   # Create analysis table
   nodes_table = g.nodes.table()
   
   # Add centrality as new columns
   centrality_data = []
   for node_id in nodes_table['node_id']:
       centrality_data.append({
           'node_id': node_id,
           'betweenness': betweenness.get(node_id, 0),
           'pagerank': pagerank.get(node_id, 0)
       })
   
   centrality_table = gr.table(centrality_data)
   
   # Join with node attributes
   enriched = nodes_table.join(centrality_table, on='node_id')

Community-Based Analysis
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Detect communities
   communities = g.communities.louvain()
   
   # Create community membership table
   membership_data = []
   for i, community in enumerate(communities):
       for node in community:
           membership_data.append({'node_id': node, 'community': i})
   
   membership_table = gr.table(membership_data)
   
   # Analyze communities with node attributes
   community_analysis = enriched.join(membership_table, on='node_id')
   
   # Group by community and analyze
   community_stats = community_analysis.group_by('community').agg({
       'age': ['mean', 'std'],
       'department': 'nunique',
       'betweenness': 'mean',
       'pagerank': 'mean'
   })

Visualization Integration
-------------------------

Preparing Data for Visualization
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare node data for visualization
   viz_nodes = enriched.to_pandas()
   
   # Prepare edge data
   edges_table = g.edges.table()
   viz_edges = edges_table.to_pandas()
   
   # Export for external visualization tools
   viz_nodes.to_csv('nodes_for_viz.csv', index=False)
   viz_edges.to_csv('edges_for_viz.csv', index=False)

Performance Optimization
------------------------

Large Graph Analysis
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # For very large graphs, process in chunks
   if g.node_count() > 100000:
       # Sample for exploratory analysis
       sample_nodes = g.nodes.sample(10000)
       sample_subgraph = g.subgraph(sample_nodes)
       
       # Run algorithms on sample
       sample_communities = sample_subgraph.communities.louvain()
       
       # Use results to guide full analysis
       
   # Use approximation algorithms for very large graphs
   approx_betweenness = g.centrality.betweenness_approximate(k=1000)

Best Practices
--------------

1. **Start Simple**: Begin with basic measures (degree, clustering) before advanced algorithms
2. **Validate Results**: Check if algorithmic results make sense for your domain
3. **Combine Measures**: Use multiple centrality measures for robust identification of important nodes
4. **Consider Scale**: Choose algorithms appropriate for your graph size
5. **Iterate**: Use initial results to refine analysis and ask better questions

The analytics capabilities in Groggy provide powerful tools for understanding network structure. Next, explore :doc:`performance` for optimization techniques and :doc:`integration` for working with other libraries.