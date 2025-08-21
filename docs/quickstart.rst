Quick Start Guide
=================

This guide will get you up and running with Groggy in just a few minutes.

Basic Graph Creation
--------------------

Let's start by creating a simple graph:

.. code-block:: python

   import groggy as gr

   # Create a new graph
   g = gr.Graph()

   # Add some nodes with attributes
   g.add_node("alice", age=30, role="engineer", salary=95000)
   g.add_node("bob", age=25, role="designer", salary=75000) 
   g.add_node("charlie", age=35, role="manager", salary=120000)

   # Add edges with attributes
   g.add_edge("alice", "bob", relationship="collaborates", strength=0.8)
   g.add_edge("charlie", "alice", relationship="manages", strength=0.9)
   g.add_edge("charlie", "bob", relationship="manages", strength=0.7)

   print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

Working with Storage Views
--------------------------

Groggy's power comes from its unified storage views that let you seamlessly switch between graph and tabular representations.

**Convert to Table**

.. code-block:: python

   # Get all nodes as a table
   nodes_table = g.nodes.table()
   print(nodes_table.head())

   # Get specific attributes only
   employee_data = g.nodes.table(attributes=["age", "role", "salary"])
   print(employee_data)

**Statistical Analysis**

.. code-block:: python

   # Basic statistics
   print(nodes_table.describe())

   # Group by operations
   role_analysis = nodes_table.group_by('role').agg({
       'salary': ['mean', 'min', 'max'],
       'age': ['mean', 'std']
   })
   print(role_analysis)

   # Individual column statistics  
   salary_stats = nodes_table['salary']
   print(f"Average salary: ${salary_stats.mean():,.2f}")
   print(f"Salary range: ${salary_stats.min():,} - ${salary_stats.max():,}")

**Matrix Operations**

.. code-block:: python

   # Get adjacency matrix
   adj_matrix = g.adjacency()
   print(f"Adjacency matrix shape: {adj_matrix.shape}")
   print(f"Is sparse: {adj_matrix.is_sparse}")

   # Matrix powers (for path counting)
   paths_2 = adj_matrix.power(2)  # 2-step paths
   paths_3 = adj_matrix.power(3)  # 3-step paths

   # Row/column operations
   out_degrees = adj_matrix.sum_axis(axis=1)  # Sum each row
   in_degrees = adj_matrix.sum_axis(axis=0)   # Sum each column

Batch Operations
----------------

For better performance, use batch operations when adding multiple entities:

.. code-block:: python

   # Batch add nodes
   new_nodes = [
       {'id': f'user_{i}', 'score': i * 10, 'active': i % 2 == 0}
       for i in range(100)
   ]
   g.add_nodes(new_nodes)

   # Batch add edges
   new_edges = [
       {'source': f'user_{i}', 'target': f'user_{i+1}', 'weight': 0.5}
       for i in range(99)
   ]
   g.add_edges(new_edges)

   print(f"Graph now has {g.node_count()} nodes and {g.edge_count()} edges")

Advanced Analytics
------------------

**Graph-Aware Table Operations**

.. code-block:: python

   # Get neighborhood information as a table
   alice_neighbors = gr.GraphTable.neighborhood_table(
       g, "alice", ["age", "role", "salary"]
   )
   print("Alice's neighbors:")
   print(alice_neighbors)

   # K-hop neighborhoods
   extended_network = gr.GraphTable.k_hop_neighborhood_table(
       g, "alice", k=2, ["role", "salary"]
   )
   print("Alice's 2-hop network:")
   print(extended_network)

**Multi-Table Operations**

.. code-block:: python

   # Create another table
   performance_data = gr.table({
       'name': ['alice', 'bob', 'charlie'],
       'performance_score': [8.5, 7.2, 9.1],
       'projects_completed': [12, 8, 15]
   })

   # Join with node data
   employee_table = g.nodes.table(attributes=['name', 'age', 'role', 'salary'])
   complete_data = employee_table.join(performance_data, on='name', how='inner')
   
   print("Complete employee data:")
   print(complete_data)

   # Advanced analysis
   high_performers = complete_data.filter_rows(
       lambda row: row['performance_score'] > 8.0
   )
   print("High performers:")
   print(high_performers)

Graph Algorithms
----------------

**Centrality Measures**

.. code-block:: python

   # Calculate different centrality measures
   betweenness = g.centrality.betweenness()
   pagerank = g.centrality.pagerank()
   closeness = g.centrality.closeness()

   # Create a table with all centrality measures
   centrality_table = gr.table({
       'node': list(betweenness.keys()),
       'betweenness': list(betweenness.values()),
       'pagerank': list(pagerank.values()),
       'closeness': list(closeness.values())
   })
   
   # Sort by PageRank
   top_nodes = centrality_table.sort_by('pagerank', ascending=False)
   print("Most central nodes:")
   print(top_nodes.head())

**Community Detection**

.. code-block:: python

   # Find communities
   communities = g.communities.louvain()
   
   # Analyze community structure
   community_sizes = {}
   for i, community in enumerate(communities):
       community_sizes[f'community_{i}'] = len(community)
   
   print("Community sizes:", community_sizes)

Working with Real Data
----------------------

**Loading from CSV**

.. code-block:: python

   import pandas as pd

   # Load nodes from CSV
   nodes_df = pd.read_csv('nodes.csv')
   nodes_table = gr.table.from_pandas(nodes_df)
   
   # Convert to graph (assuming 'id' column exists)
   g_from_csv = gr.Graph()
   for row in nodes_table.to_dict():
       node_id = row.pop('id')  # Remove 'id' from attributes
       g_from_csv.add_node(node_id, **row)

**Integration with NetworkX**

.. code-block:: python

   import networkx as nx

   # Convert Groggy graph to NetworkX
   nx_graph = g.to_networkx()
   
   # Use NetworkX algorithms
   nx_communities = nx.community.greedy_modularity_communities(nx_graph)
   
   # Convert NetworkX graph to Groggy
   g_from_nx = gr.from_networkx(nx_graph)

Memory Efficiency and Performance
---------------------------------

**Lazy Evaluation**

.. code-block:: python

   # These operations are lazy - no computation until needed
   large_table = g.nodes.table()  # Instant
   filtered_view = large_table.filter_rows(lambda r: r['age'] > 25)  # Still lazy
   
   # Computation happens here
   result = filtered_view.head(10)  # Only computes first 10 matching rows

**Sparse Matrices**

.. code-block:: python

   # Groggy automatically uses sparse representation for large, sparse graphs
   adj = g.adjacency()
   print(f"Matrix is sparse: {adj.is_sparse}")
   print(f"Matrix density: {adj.density():.4f}")
   
   # Convert between sparse and dense as needed
   dense_adj = adj.to_dense()
   sparse_adj = dense_adj.to_sparse()

**Memory Usage**

.. code-block:: python

   # Check memory usage
   print(f"Graph memory usage: {g.memory_usage():,} bytes")
   print(f"Table memory usage: {nodes_table.memory_usage():,} bytes")

Next Steps
----------

Now that you've seen the basics, explore these topics:

- :doc:`user-guide/storage-views` - Deep dive into Arrays, Matrices, and Tables
- :doc:`user-guide/analytics` - Advanced graph algorithms and analysis
- :doc:`user-guide/performance` - Optimization techniques for large graphs
- :doc:`examples/index` - More comprehensive examples and use cases

For complete API documentation, see :doc:`api/index`.