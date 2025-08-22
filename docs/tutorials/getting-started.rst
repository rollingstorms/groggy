Getting Started Tutorial
========================

This tutorial will guide you through your first steps with Groggy, from installation to performing basic graph analysis.

Installation
-----------

Installing Groggy
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install from PyPI (recommended)
   pip install groggy

   # Install with optional dependencies
   pip install groggy[visualization,data]

   # Install development version
   pip install git+https://github.com/groggy-dev/groggy.git

Verifying Installation
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import groggy as gr
   print(f"Groggy version: {gr.__version__}")
   
   # Create a simple test graph
   g = gr.Graph()
   g.add_node("test")
   print("Installation successful!")

Your First Graph
---------------

Creating a Graph
~~~~~~~~~~~~~~~

.. code-block:: python

   import groggy as gr

   # Create an empty directed graph
   g = gr.Graph(directed=True)

   # Create an undirected graph
   g_undirected = gr.Graph(directed=False)

   print(f"Graph is directed: {g.directed}")
   print(f"Graph is undirected: {g_undirected.directed}")

Adding Nodes
~~~~~~~~~~~

.. code-block:: python

   # Add individual nodes
   g.add_node("alice")
   g.add_node("bob")
   g.add_node("charlie")

   # Add nodes with attributes
   g.add_node("diana", age=25, role="engineer", department="AI")

   # Add multiple nodes at once
   nodes_to_add = [
       {"id": "eve", "age": 30, "role": "manager"},
       {"id": "frank", "age": 28, "role": "designer"}
   ]
   g.add_nodes(nodes_to_add)

   print(f"Number of nodes: {g.node_count()}")
   print(f"Nodes in graph: {list(g.nodes)}")

Adding Edges
~~~~~~~~~~~

.. code-block:: python

   # Add individual edges
   g.add_edge("alice", "bob")
   g.add_edge("bob", "charlie")
   g.add_edge("charlie", "diana")

   # Add edges with attributes
   g.add_edge("diana", "eve", weight=0.8, relationship="collaborates")

   # Add multiple edges at once
   edges_to_add = [
       {"source": "eve", "target": "frank", "weight": 0.9},
       {"source": "frank", "target": "alice", "weight": 0.7}
   ]
   g.add_edges(edges_to_add)

   print(f"Number of edges: {g.edge_count()}")
   print(f"Graph density: {g.density():.3f}")

Exploring Your Graph
-------------------

Basic Properties
~~~~~~~~~~~~~~~

.. code-block:: python

   # Graph statistics
   print(f"Nodes: {g.node_count()}")
   print(f"Edges: {g.edge_count()}")
   print(f"Density: {g.density():.3f}")
   print(f"Is connected: {g.is_connected()}")

   # Node and edge information
   print(f"Node with highest degree: {max(g.degree().items(), key=lambda x: x[1])}")
   
   if g.directed:
       print(f"Node with highest in-degree: {max(g.in_degree().items(), key=lambda x: x[1])}")
       print(f"Node with highest out-degree: {max(g.out_degree().items(), key=lambda x: x[1])}")

Accessing Node and Edge Data
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Access node attributes
   alice_data = g.nodes["alice"]
   print(f"Alice's data: {alice_data}")

   # Check if node exists
   if g.has_node("diana"):
       diana_attrs = g.get_node("diana")
       print(f"Diana's age: {diana_attrs.get('age', 'Unknown')}")

   # Access edge attributes
   if g.has_edge("diana", "eve"):
       edge_data = g.get_edge("diana", "eve")
       print(f"Diana-Eve relationship: {edge_data}")

   # Get neighbors
   alice_neighbors = g.neighbors("alice")
   print(f"Alice's neighbors: {alice_neighbors}")

Working with Storage Views
-------------------------

Array View
~~~~~~~~~

.. code-block:: python

   # Create a table view of nodes
   nodes_table = g.nodes.table()
   print(f"Node table shape: {nodes_table.shape}")
   print(f"Columns: {nodes_table.columns}")

   # Get age as an array
   if 'age' in nodes_table.columns:
       ages = nodes_table['age']
       print(f"Age statistics:")
       print(f"  Mean: {ages.mean():.1f}")
       print(f"  Min: {ages.min()}")
       print(f"  Max: {ages.max()}")
       print(f"  Std Dev: {ages.std():.1f}")

Table View
~~~~~~~~~

.. code-block:: python

   # Work with the full table
   print("First few rows of node data:")
   print(nodes_table.head())

   # Filter nodes
   if 'age' in nodes_table.columns:
       young_employees = nodes_table.filter_rows(lambda row: row.get('age', 0) < 30)
       print(f"Young employees: {len(young_employees)} out of {len(nodes_table)}")

   # Group by department (if column exists)
   if 'department' in nodes_table.columns:
       dept_stats = nodes_table.group_by('department').agg({'age': 'mean'})
       print("Average age by department:")
       print(dept_stats)

Basic Graph Algorithms
---------------------

Centrality Analysis
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Calculate different centrality measures
   print("Centrality Analysis:")

   # PageRank (works for all graph types)
   pagerank = g.centrality.pagerank()
   print(f"PageRank scores:")
   for node, score in sorted(pagerank.items(), key=lambda x: x[1], reverse=True):
       print(f"  {node}: {score:.3f}")

   # Degree centrality
   degree_centrality = g.centrality.degree()
   print(f"\nDegree centrality:")
   for node, score in sorted(degree_centrality.items(), key=lambda x: x[1], reverse=True):
       print(f"  {node}: {score:.3f}")

   # Betweenness centrality (may be slow for large graphs)
   if g.node_count() <= 100:  # Only for small graphs in this tutorial
       betweenness = g.centrality.betweenness()
       print(f"\nBetweenness centrality:")
       for node, score in sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:3]:
           print(f"  {node}: {score:.3f}")

Community Detection
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find communities in the graph
   if g.node_count() >= 3:  # Need at least 3 nodes for meaningful communities
       print("\nCommunity Detection:")
       
       # Louvain algorithm
       communities = g.communities.louvain()
       print(f"Found {len(communities)} communities:")
       
       for i, community in enumerate(communities):
           print(f"  Community {i+1}: {community}")
       
       # Calculate modularity
       modularity = g.communities.modularity(communities)
       print(f"Modularity score: {modularity:.3f}")

Path Analysis
~~~~~~~~~~~~

.. code-block:: python

   # Find shortest paths
   print("\nPath Analysis:")

   try:
       # Shortest path between two nodes
       path = g.shortest_path("alice", "diana")
       print(f"Shortest path from Alice to Diana: {' -> '.join(path)}")
       print(f"Path length: {len(path) - 1}")
   except ValueError:
       print("No path found between Alice and Diana")

   # Check connectivity
   connected_pairs = []
   nodes = list(g.nodes)
   for i, node1 in enumerate(nodes):
       for node2 in nodes[i+1:]:
           if g.has_path(node1, node2):
               connected_pairs.append((node1, node2))
   
   print(f"Connected node pairs: {len(connected_pairs)} out of {len(nodes) * (len(nodes) - 1) // 2}")

Working with Real Data
--------------------

Loading from External Sources
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Example: Creating a graph from edge list data
   def create_collaboration_network():
       """Create a sample collaboration network"""
       
       # Sample data: employee collaborations
       collaborations = [
           ("Alice", "Bob", {"project": "Project X", "frequency": 5}),
           ("Bob", "Charlie", {"project": "Project Y", "frequency": 3}),
           ("Charlie", "Diana", {"project": "Project X", "frequency": 8}),
           ("Diana", "Eve", {"project": "Project Z", "frequency": 2}),
           ("Eve", "Alice", {"project": "Project Y", "frequency": 4}),
           ("Bob", "Diana", {"project": "Project Z", "frequency": 6})
       ]
       
       # Employee information
       employees = [
           {"id": "Alice", "department": "Engineering", "seniority": 5},
           {"id": "Bob", "department": "Engineering", "seniority": 3},
           {"id": "Charlie", "department": "Design", "seniority": 4},
           {"id": "Diana", "department": "Product", "seniority": 6},
           {"id": "Eve", "department": "Product", "seniority": 2}
       ]
       
       # Build graph
       g = gr.Graph(directed=False)  # Undirected for collaboration
       
       # Add employees as nodes
       for emp in employees:
           g.add_node(emp["id"], 
                     department=emp["department"], 
                     seniority=emp["seniority"])
       
       # Add collaborations as edges
       for source, target, attrs in collaborations:
           g.add_edge(source, target, **attrs)
       
       return g

   # Create and analyze the collaboration network
   collab_graph = create_collaboration_network()
   print(f"\nCollaboration Network:")
   print(f"Employees: {collab_graph.node_count()}")
   print(f"Collaborations: {collab_graph.edge_count()}")

Analyzing the Collaboration Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Department-based analysis
   nodes_table = collab_graph.nodes.table()
   
   print("\nDepartment Analysis:")
   dept_stats = nodes_table.group_by('department').agg({
       'seniority': ['mean', 'count']
   })
   print(dept_stats)

   # Find key collaborators (high degree centrality)
   degree_centrality = collab_graph.centrality.degree()
   key_collaborators = sorted(degree_centrality.items(), 
                             key=lambda x: x[1], reverse=True)
   
   print(f"\nKey Collaborators:")
   for person, centrality in key_collaborators:
       person_data = collab_graph.nodes[person]
       print(f"  {person} ({person_data['department']}): {centrality:.3f}")

   # Project frequency analysis
   edges_table = collab_graph.edges.table()
   if 'frequency' in edges_table.columns:
       avg_frequency = edges_table['frequency'].mean()
       high_freq_collabs = edges_table.filter_rows(
           lambda row: row['frequency'] > avg_frequency
       )
       
       print(f"\nHigh-frequency collaborations ({len(high_freq_collabs)} out of {len(edges_table)}):")
       for _, row in high_freq_collabs.iterrows():
           print(f"  {row['source']} <-> {row['target']}: {row['frequency']} times")

Visualization and Export
-----------------------

Converting to Other Formats
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to pandas for external analysis
   nodes_df = collab_graph.nodes.table().to_pandas()
   edges_df = collab_graph.edges.table().to_pandas()

   print(f"\nData shapes:")
   print(f"Nodes DataFrame: {nodes_df.shape}")
   print(f"Edges DataFrame: {edges_df.shape}")

   # Export to files (optional)
   # nodes_df.to_csv('nodes.csv', index=False)
   # edges_df.to_csv('edges.csv', index=False)

Integration with NetworkX
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Convert to NetworkX for visualization (if networkx is installed)
   try:
       import networkx as nx
       import matplotlib.pyplot as plt
       
       # Convert to NetworkX
       nx_graph = collab_graph.to_networkx()
       
       # Simple visualization
       plt.figure(figsize=(10, 8))
       pos = nx.spring_layout(nx_graph)
       
       # Draw nodes colored by department
       node_colors = []
       dept_color_map = {'Engineering': 'lightblue', 'Design': 'lightgreen', 'Product': 'lightcoral'}
       
       for node in nx_graph.nodes():
           dept = collab_graph.nodes[node]['department']
           node_colors.append(dept_color_map.get(dept, 'lightgray'))
       
       nx.draw(nx_graph, pos, node_color=node_colors, 
               with_labels=True, node_size=1000, font_size=10)
       
       plt.title("Employee Collaboration Network")
       plt.show()
       
   except ImportError:
       print("NetworkX not installed. Skipping visualization.")

Performance Tips
---------------

For Large Graphs
~~~~~~~~~~~~~~~

.. code-block:: python

   # Tips for working with larger graphs
   
   def analyze_large_graph(g):
       """Efficient analysis for large graphs"""
       
       # Use sampling for expensive algorithms
       if g.node_count() > 10000:
           print("Large graph detected - using optimized algorithms")
           
           # Sample for community detection
           sample_nodes = g.nodes.sample(n=min(5000, g.node_count() // 2))
           sample_subgraph = g.subgraph(sample_nodes)
           communities = sample_subgraph.communities.louvain()
           
           # Use approximate centrality
           pagerank = g.centrality.pagerank(max_iter=50, tolerance=1e-4)
           
       else:
           # Use exact algorithms for smaller graphs
           communities = g.communities.louvain()
           pagerank = g.centrality.pagerank()
       
       return {
           'communities': len(communities),
           'top_pagerank': max(pagerank.items(), key=lambda x: x[1])
       }

Memory Management
~~~~~~~~~~~~~~~

.. code-block:: python

   # Monitor memory usage
   import psutil
   import os

   def get_memory_usage():
       process = psutil.Process(os.getpid())
       return process.memory_info().rss / 1024 / 1024  # MB

   print(f"Initial memory usage: {get_memory_usage():.1f} MB")

   # Create a moderately large graph
   large_g = gr.random_graph(10000, edge_probability=0.001)
   print(f"After creating graph: {get_memory_usage():.1f} MB")

   # Perform analysis
   result = large_g.centrality.pagerank()
   print(f"After PageRank: {get_memory_usage():.1f} MB")

   # Clean up
   del large_g, result
   print(f"After cleanup: {get_memory_usage():.1f} MB")

Next Steps
---------

Now that you've learned the basics, explore these areas:

1. **Advanced Algorithms**: Try betweenness centrality, eigenvector centrality, and advanced community detection
2. **Storage Views**: Learn about GraphMatrix for linear algebra operations
3. **Performance Optimization**: Explore parallel processing and approximation algorithms
4. **Custom Analysis**: Create your own analysis pipelines
5. **Integration**: Connect Groggy with your data sources and visualization tools

Key Takeaways
~~~~~~~~~~~~

- Groggy provides both directed and undirected graph support
- Storage views (Array, Table, Matrix) offer different perspectives on your data
- Built-in algorithms are optimized for performance
- The library integrates well with the Python data science ecosystem
- Memory efficiency and performance scale well with graph size

Continue with the User Guide sections to dive deeper into specific topics, or check out the API Reference for detailed documentation of all available functions and methods.