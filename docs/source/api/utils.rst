utils module
============

.. automodule:: groggy.utils
   :members:
   :undoc-members:
   :show-inheritance:

Utility Functions
-----------------

Graph Creation
~~~~~~~~~~~~~~

.. autofunction:: groggy.utils.create_random_graph

   Create a random graph for testing and benchmarking.

   **Parameters:**
   
   * ``n_nodes`` (int): Number of nodes to create
   * ``edge_probability`` (float): Probability of edge between any two nodes
   * ``use_rust`` (bool, optional): Force backend selection
   
   **Returns:**
   
   * ``Graph``: A new graph with random structure

   **Example:**

   .. code-block:: python

      from groggy.utils import create_random_graph
      
      # Create small random graph
      g = create_random_graph(100, 0.1)
      
      # Create dense graph
      dense_g = create_random_graph(50, 0.8)
      
      # Force Python backend
      py_g = create_random_graph(100, 0.1, use_rust=False)

Workflow Utilities
~~~~~~~~~~~~~~~~~~

.. autofunction:: groggy.utils.create_clustering_workflow

   Set up branches for different clustering algorithms.

.. autofunction:: groggy.utils.create_subgraph_branch

   Create a branch from a subgraph for isolated processing.

Performance Testing
-------------------

The utils module provides functions specifically designed for performance testing:

.. code-block:: python

   import time
   from groggy.utils import create_random_graph
   
   # Benchmark graph creation
   start = time.time()
   large_graph = create_random_graph(10000, 0.01)
   creation_time = time.time() - start
   
   print(f"Created {large_graph.node_count()} nodes in {creation_time:.3f}s")
   print(f"Graph has {large_graph.edge_count()} edges")

Best Practices
--------------

**Random Graph Generation**

* Use lower edge probabilities for large graphs to avoid memory issues
* Consider using Rust backend for graphs with >1000 nodes
* Test with different random seeds for reproducible results

**Workflow Management**

* Use descriptive branch names for different analysis stages
* Create subgraph branches for isolated processing
* Clean up temporary branches after analysis

.. code-block:: python

   # Good practice: structured workflow
   from groggy.utils import create_clustering_workflow, create_random_graph
   
   # Create base graph
   graph = create_random_graph(1000, 0.05)
   
   # Set up analysis workflow
   clustering_branches = create_clustering_workflow(
       store, graph, 
       algorithms=['kmeans', 'spectral', 'dbscan']
   )
   
   # Process each algorithm in parallel
   for branch in clustering_branches:
       # Switch to branch and run algorithm
       pass
