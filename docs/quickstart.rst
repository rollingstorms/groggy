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

   # Add some nodes with attributes (returns numeric IDs)
   alice = g.add_node(name="Alice", age=30, role="engineer", salary=95000)
   bob = g.add_node(name="Bob", age=25, role="designer", salary=75000) 
   charlie = g.add_node(name="Charlie", age=35, role="manager", salary=120000)

   # Add edges with attributes
   g.add_edge(alice, bob, relationship="collaborates", strength=0.8)
   g.add_edge(charlie, alice, relationship="manages", strength=0.9)
   g.add_edge(charlie, bob, relationship="manages", strength=0.7)

   print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")

Working with Storage Views
--------------------------

Groggy's power comes from its unified storage views that let you seamlessly switch between graph and tabular representations.

**Convert to Table**

.. code-block:: python

   # Get all nodes as a table
   nodes_table = g.nodes.table()
   print(nodes_table.head())

   # Access specific columns from the table
   employee_data = g.nodes.table()
   print(employee_data[["age", "role", "salary"]])

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

   # Batch add nodes (note: returns list of node IDs)
   new_nodes = [
       {'score': i * 10, 'active': i % 2 == 0, 'user_id': f'user_{i}'}
       for i in range(100)
   ]
   node_ids = g.add_nodes(new_nodes)

   # Batch add edges using returned node IDs
   new_edges = [
       (node_ids[i], node_ids[i+1], {'weight': 0.5})
       for i in range(99)
   ]
   g.add_edges(new_edges)

   print(f"Graph now has {g.node_count()} nodes and {g.edge_count()} edges")

Advanced Analytics
------------------

**Graph-Aware Table Operations**

.. code-block:: python

   # Get neighbors using graph methods
   alice_neighbors = g.neighbors(alice)
   print(f"Alice's neighbors: {alice_neighbors}")
   
   # Get neighbor data as table
   neighbor_data = []
   for neighbor_id in alice_neighbors:
       neighbor = g.nodes[neighbor_id]
       neighbor_data.append({
           'id': neighbor_id,
           'age': neighbor['age'],
           'role': neighbor['role'],
           'salary': neighbor['salary']
       })
   neighbors_table = gr.table(neighbor_data)
   print("Alice's neighbors data:")
   print(neighbors_table)

**Multi-Table Operations**

.. code-block:: python

   # Create another table
   performance_data = gr.table({
       'name': ['Alice', 'Bob', 'Charlie'],
       'performance_score': [8.5, 7.2, 9.1],
       'projects_completed': [12, 8, 15]
   })

   # Join with node data
   employee_table = g.nodes.table()
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

**Basic Network Analysis**

.. code-block:: python

   # Calculate basic centrality measures (degree centrality)
   degrees = g.degree()
   
   # Create a table with node information
   nodes_table = g.nodes.table()
   nodes_with_degree = []
   
   for i, degree in enumerate(degrees):
       if i < len(nodes_table):
           node_data = nodes_table[i].to_dict() if hasattr(nodes_table[i], 'to_dict') else dict(nodes_table[i])
           node_data['degree'] = degree
           nodes_with_degree.append(node_data)
   
   centrality_table = gr.table(nodes_with_degree)
   
   # Sort by degree (simple centrality measure)
   top_nodes = centrality_table.sort_by('degree', ascending=False)
   print("Most connected nodes:")
   print(top_nodes.head())

**Connected Components Analysis**

.. code-block:: python

   # Find connected components (basic community structure)
   components = g.analytics.connected_components()
   
   # Analyze component structure
   component_sizes = {}
   for i, component in enumerate(components):
       component_sizes[f'component_{i}'] = len(component)
   
   print("Component sizes:", component_sizes)
   print(f"Number of components: {len(components)}")
   print(f"Graph is connected: {g.is_connected()}")

Working with Real Data
----------------------

**Loading from CSV**

.. code-block:: python

   import pandas as pd

   # Load nodes from CSV
   nodes_df = pd.read_csv('nodes.csv')
   nodes_table = gr.table.from_pandas(nodes_df)
   
   # Convert to graph
   g_from_csv = gr.Graph()
   for i in range(len(nodes_table)):
       row_data = nodes_table[i]
       if hasattr(row_data, 'to_dict'):
           attrs = row_data.to_dict()
       else:
           attrs = dict(row_data)
       g_from_csv.add_node(**attrs)

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