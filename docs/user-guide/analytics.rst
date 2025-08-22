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
   components = g.analytics.connected_components()
   print(f"Number of components: {len(components)}")

   # Get the largest component
   largest = max(components, key=lambda c: len(c))
   print(f"Largest component: {len(largest)} nodes")

   # Check if graph is connected
   is_connected = g.is_connected()
   print(f"Graph is connected: {is_connected}")

Path Finding
~~~~~~~~~~~~

.. code-block:: python

   # Find shortest path between nodes (using node IDs)
   # Assuming alice and bob are node IDs from previous examples
   try:
       path = g.analytics.shortest_path(alice, bob)
       print(f"Shortest path: {path}")
       print(f"Path length: {len(path) - 1}")
   except ValueError:
       print("No path exists between nodes")

   # Check if path exists
   has_path = g.analytics.has_path(alice, bob)
   print(f"Path exists: {has_path}")

   # Note: all_simple_paths not available in current implementation
   # Will be added in future releases

Graph Traversal
~~~~~~~~~~~~~~~

.. code-block:: python

   # Breadth-first search
   bfs_order = g.analytics.bfs(alice)
   print(f"BFS traversal: {bfs_order}")

   # Depth-first search  
   dfs_order = g.analytics.dfs(alice)
   print(f"DFS traversal: {dfs_order}")

   # Note: Custom visitor patterns not available in current implementation
   # Basic traversal orders are returned as lists

Basic Network Analysis
----------------------

The current release focuses on fundamental graph operations. Advanced centrality measures will be added in future releases.

Degree Analysis
~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic degree centrality (available now)
   degrees = g.degree()
   print(f"Node degrees: {degrees}")
   
   # Normalize by maximum possible degree
   n = g.node_count()
   if isinstance(degrees, dict):
       normalized_degrees = {node: deg/(n-1) for node, deg in degrees.items()}
   else:
       # degrees is a list/array
       max_possible = n - 1
       normalized_degrees = [deg/max_possible for deg in degrees]

   # For directed graphs
   if g.is_directed:
       in_degrees = g.in_degree()
       out_degrees = g.out_degree()
       print(f"In-degrees: {in_degrees}")
       print(f"Out-degrees: {out_degrees}")

Node Importance by Degree
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Identify highly connected nodes
   degrees = g.degree()
   
   # Get nodes table for analysis
   nodes_table = g.nodes.table()
   
   # Combine with degree information
   degree_analysis = []
   for i, degree in enumerate(degrees):
       if i < len(nodes_table):
           node_data = nodes_table[i].to_dict() if hasattr(nodes_table[i], 'to_dict') else dict(nodes_table[i])
           node_data['degree'] = degree
           node_data['node_id'] = i
           degree_analysis.append(node_data)
   
   # Create analysis table
   degree_table = gr.table(degree_analysis)
   
   # Sort by degree to find most connected nodes
   top_nodes = degree_table.sort_by('degree', ascending=False)
   print("Most connected nodes:")
   print(top_nodes.head())

Advanced Centrality Measures
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Advanced centrality measures (betweenness, PageRank, eigenvector centrality) 
   will be available in the next major release as part of the analytics module expansion.

Connected Components Analysis
------------------------------

The current release provides connected component analysis. Advanced community detection algorithms will be added in future releases.

Component Detection
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Find connected components (basic clustering)
   components = g.analytics.connected_components()
   
   print(f"Found {len(components)} connected components")
   for i, component in enumerate(components):
       print(f"Component {i}: {len(component)} nodes")
       
   # Analyze component sizes
   sizes = [len(component) for component in components]
   if sizes:
       print(f"Average component size: {sum(sizes) / len(sizes):.1f}")
       print(f"Largest component: {max(sizes)} nodes")

Component Analysis
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze the largest component
   if components:
       largest_component = max(components, key=lambda c: len(c))
       
       # Create subgraph of largest component
       largest_subgraph = g.subgraph(largest_component)
       
       # Analyze the largest component
       print(f"Largest component analysis:")
       print(f"  Nodes: {largest_subgraph.node_count()}")
       print(f"  Edges: {largest_subgraph.edge_count()}")
       print(f"  Density: {largest_subgraph.density():.3f}")

Advanced Community Detection
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Advanced community detection algorithms (Louvain, Leiden, modularity optimization) 
   will be available in the next major release.

Basic Network Properties
------------------------

The current release provides fundamental network properties. Advanced metrics will be available in future releases.

Graph Density
~~~~~~~~~~~~~

.. code-block:: python

   # Graph density (available now)
   density = g.density()
   print(f"Graph density: {density:.3f}")
   
   # Basic connectivity
   is_connected = g.is_connected()
   print(f"Graph is connected: {is_connected}")
   
   # Size information
   print(f"Nodes: {g.node_count()}")
   print(f"Edges: {g.edge_count()}")

Degree Distribution Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze degree distribution
   degrees = g.degree()
   
   if degrees:
       if isinstance(degrees, dict):
           degree_values = list(degrees.values())
       else:
           degree_values = degrees
           
       avg_degree = sum(degree_values) / len(degree_values)
       max_degree = max(degree_values)
       min_degree = min(degree_values)
       
       print(f"Average degree: {avg_degree:.2f}")
       print(f"Degree range: {min_degree} - {max_degree}")

Path Analysis
~~~~~~~~~~~~~

.. code-block:: python

   # Basic path analysis (for connected graphs)
   if g.is_connected():
       # Sample path analysis between two nodes
       nodes_table = g.nodes.table()
       if len(nodes_table) >= 2:
           # Get first two node IDs
           node1 = 0 if 0 < g.node_count() else None
           node2 = 1 if 1 < g.node_count() else None
           
           if node1 is not None and node2 is not None:
               try:
                   path = g.analytics.shortest_path(node1, node2)
                   print(f"Sample shortest path length: {len(path) - 1}")
               except:
                   print("Could not compute sample path")

Advanced Network Metrics
~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Advanced network metrics (clustering coefficients, assortativity, diameter) 
   will be available in the next major release.

Practical Graph Analysis
------------------------

Working with Real Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Comprehensive graph analysis workflow
   def analyze_graph(g):
       """Basic analysis of graph structure"""
       
       print("=== Graph Overview ===")
       print(f"Nodes: {g.node_count()}")
       print(f"Edges: {g.edge_count()}")
       print(f"Directed: {g.is_directed}")
       print(f"Density: {g.density():.4f}")
       print(f"Connected: {g.is_connected()}")
       
       print("\n=== Connectivity Analysis ===")
       components = g.analytics.connected_components()
       print(f"Connected components: {len(components)}")
       
       if components:
           component_sizes = [len(comp) for comp in components]
           print(f"Largest component: {max(component_sizes)} nodes")
           print(f"Average component size: {sum(component_sizes)/len(component_sizes):.1f}")
       
       print("\n=== Degree Analysis ===")
       degrees = g.degree()
       if degrees:
           if isinstance(degrees, dict):
               degree_vals = list(degrees.values())
           else:
               degree_vals = degrees
           
           print(f"Average degree: {sum(degree_vals)/len(degree_vals):.2f}")
           print(f"Max degree: {max(degree_vals)}")
           print(f"Min degree: {min(degree_vals)}")
       
       return {
           'nodes': g.node_count(),
           'edges': g.edge_count(),
           'density': g.density(),
           'connected': g.is_connected(),
           'components': len(components) if components else 0
       }
   
   # Run analysis
   results = analyze_graph(g)

Advanced Structural Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. note::
   Advanced structural analysis (bridges, articulation points, k-core decomposition, 
   motif analysis) will be available in the next major release.

Data Integration and Analysis
-----------------------------

Combining Graph and Table Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Leverage Groggy's unified storage views
   # 1. Node-level analysis
   node_metrics = []
   degrees = g.degree()
   nodes_table = g.nodes.table()
   
   for i, degree in enumerate(degrees):
       if i < len(nodes_table):
           node_data = nodes_table[i].to_dict() if hasattr(nodes_table[i], 'to_dict') else dict(nodes_table[i])
           node_data['degree'] = degree
           node_data['node_id'] = i
           node_metrics.append(node_data)
   
   # Create analysis table
   analysis_table = gr.table(node_metrics)
   
   # 2. Component-level analysis
   components = g.analytics.connected_components()
   component_info = [{'component_id': i, 'size': len(comp)} for i, comp in enumerate(components)]
   component_table = gr.table(component_info)
   
   # 3. Global-level metrics
   global_metrics = {
       'density': g.density(),
       'node_count': g.node_count(),
       'edge_count': g.edge_count(),
       'component_count': len(components),
       'is_connected': g.is_connected()
   }

Working with Edge Attributes
~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze edge attributes if available
   edges_table = g.edges.table()
   print(f"Edge table columns: {edges_table.columns}")
   
   # Example: analyze edge weights if they exist
   if 'weight' in edges_table.columns:
       weights = edges_table['weight']
       print(f"Average edge weight: {weights.mean():.3f}")
       print(f"Weight range: {weights.min():.3f} - {weights.max():.3f}")
   
   # Example: analyze edge types if they exist
   if 'type' in edges_table.columns:
       edge_types = edges_table['type'].value_counts()
       print("Edge type distribution:")
       for edge_type, count in edge_types.items():
           print(f"  {edge_type}: {count}")

Node Similarity Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Simple node similarity based on common neighbors
   def jaccard_similarity(g, node1, node2):
       """Calculate Jaccard similarity between two nodes"""
       try:
           neighbors1 = set(g.neighbors(node1))
           neighbors2 = set(g.neighbors(node2))
           
           intersection = len(neighbors1 & neighbors2)
           union = len(neighbors1 | neighbors2)
           
           return intersection / union if union > 0 else 0
       except:
           return 0

   # Example: find similar nodes
   if g.node_count() >= 2:
       node1, node2 = 0, 1  # First two nodes
       similarity = jaccard_similarity(g, node1, node2)
       print(f"Similarity between nodes {node1} and {node2}: {similarity:.3f}")

Working with Storage Views
--------------------------

Combining Graph and Table Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Groggy's strength: seamless graph-table integration
   
   # 1. Start with graph analysis
   degrees = g.degree()
   components = g.analytics.connected_components()
   
   # 2. Get tabular view of nodes
   nodes_table = g.nodes.table()
   
   # 3. Enrich with graph metrics
   enriched_data = []
   for i, degree in enumerate(degrees):
       if i < len(nodes_table):
           node_data = nodes_table[i].to_dict() if hasattr(nodes_table[i], 'to_dict') else dict(nodes_table[i])
           node_data['degree'] = degree
           
           # Add component membership
           for comp_id, component in enumerate(components):
               if i in component:
                   node_data['component'] = comp_id
                   break
           else:
               node_data['component'] = -1  # Isolated node
           
           enriched_data.append(node_data)
   
   # 4. Create enriched analysis table
   enriched_table = gr.table(enriched_data)
   
   # 5. Perform tabular analysis
   print("Analysis by component:")
   for comp_id in set(item['component'] for item in enriched_data):
       comp_nodes = [item for item in enriched_data if item['component'] == comp_id]
       if comp_nodes:
           avg_degree = sum(node['degree'] for node in comp_nodes) / len(comp_nodes)
           print(f"  Component {comp_id}: {len(comp_nodes)} nodes, avg degree {avg_degree:.2f}")

Export for External Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare data for external tools
   
   # Export node data with graph metrics
   export_data = []
   degrees = g.degree()
   nodes_table = g.nodes.table()
   
   for i, degree in enumerate(degrees):
       if i < len(nodes_table):
           node_data = nodes_table[i].to_dict() if hasattr(nodes_table[i], 'to_dict') else dict(nodes_table[i])
           node_data.update({
               'node_id': i,
               'degree': degree,
               'graph_density': g.density(),
               'total_nodes': g.node_count(),
               'total_edges': g.edge_count()
           })
           export_data.append(node_data)
   
   # Create export table
   export_table = gr.table(export_data)
   
   # Can convert to pandas for external analysis
   # pandas_df = export_table.to_pandas()
   
   print(f"Export table ready with {len(export_table)} rows")

Next Steps and Future Features
-----------------------------

Current Capabilities Summary
~~~~~~~~~~~~~~~~~~~~~~~~~~~

The current release provides a solid foundation for graph analysis:

- **Core Graph Operations**: Node/edge management, filtering, subgraphs
- **Basic Analytics**: Connectivity, components, degree analysis, shortest paths
- **Storage Views**: Seamless graph-table integration
- **Data Export**: NetworkX compatibility, table exports

Roadmap for Advanced Analytics
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Future releases will include:

- **Advanced Centrality**: Betweenness, PageRank, eigenvector centrality
- **Community Detection**: Louvain, Leiden algorithms with modularity optimization
- **Network Metrics**: Clustering coefficients, assortativity, diameter calculations
- **Structural Analysis**: Bridges, articulation points, k-core decomposition
- **Visualization Engine**: Built-in graph visualization and plotting
- **Linear Algebra Module**: Advanced matrix operations for graph algorithms

Best Practices for Current Release
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Leverage Storage Views**: Use graph-table integration for analysis workflows
2. **Start with Basics**: Degree analysis and connectivity provide rich insights
3. **Use Filtering**: Create focused subgraphs for detailed analysis  
4. **Export When Needed**: Use NetworkX integration for advanced algorithms
5. **Build Incrementally**: Combine basic operations for complex analysis

Example Complete Workflow
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complete analysis workflow with current capabilities
   def complete_graph_analysis(g):
       """Comprehensive analysis using available features"""
       
       print("=== GRAPH ANALYSIS REPORT ===")
       
       # Basic properties
       print(f"Nodes: {g.node_count()}, Edges: {g.edge_count()}")
       print(f"Density: {g.density():.4f}")
       print(f"Directed: {g.is_directed}")
       
       # Connectivity
       print(f"Connected: {g.is_connected()}")
       components = g.analytics.connected_components()
       print(f"Components: {len(components)}")
       
       # Degree analysis
       degrees = g.degree()
       if degrees:
           degree_vals = degrees if not isinstance(degrees, dict) else list(degrees.values())
           print(f"Avg degree: {sum(degree_vals)/len(degree_vals):.2f}")
           print(f"Max degree: {max(degree_vals)}")
       
       # Create analysis table
       nodes_table = g.nodes.table()
       enriched_data = []
       for i, degree in enumerate(degrees):
           if i < len(nodes_table):
               node_data = {'node_id': i, 'degree': degree}
               enriched_data.append(node_data)
       
       analysis_table = gr.table(enriched_data)
       print(f"Analysis table created with {len(analysis_table)} rows")
       
       return analysis_table

This foundation provides everything needed for robust graph analysis workflows. Future releases will add advanced algorithms while maintaining the same intuitive API.