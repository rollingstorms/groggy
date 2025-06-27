graph module
============

.. automodule:: gli.graph
   :members:
   :undoc-members:
   :show-inheritance:

Graph Class
-----------

The Graph class is the main interface for graph operations in GLI. It provides a high-level Python API backed by a high-performance Rust implementation.

.. autoclass:: gli.graph.core.Graph
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Creation Methods
~~~~~~~~~~~~~~~~

.. automethod:: gli.graph.core.Graph.from_edge_list
.. automethod:: gli.graph.core.Graph.from_node_list

Core Operations
~~~~~~~~~~~~~~~

Node Operations
^^^^^^^^^^^^^^^

.. automethod:: gli.graph.core.Graph.add_node
.. automethod:: gli.graph.core.Graph.get_node
.. automethod:: gli.graph.core.Graph.set_node_attribute
.. automethod:: gli.graph.core.Graph.set_node_attributes
.. automethod:: gli.graph.core.Graph.get_node_attributes

Edge Operations
^^^^^^^^^^^^^^^

.. automethod:: gli.graph.core.Graph.add_edge
.. automethod:: gli.graph.core.Graph.get_edge
.. automethod:: gli.graph.core.Graph.set_edge_attribute
.. automethod:: gli.graph.core.Graph.set_edge_attributes
.. automethod:: gli.graph.core.Graph.get_edge_attributes

Batch Operations
~~~~~~~~~~~~~~~~

.. automethod:: gli.graph.core.Graph.set_nodes_attributes_batch
.. automethod:: gli.graph.core.Graph.set_edges_attributes_batch

Graph Queries
~~~~~~~~~~~~~

.. automethod:: gli.graph.core.Graph.get_neighbors
.. automethod:: gli.graph.core.Graph.node_count
.. automethod:: gli.graph.core.Graph.edge_count
.. automethod:: gli.graph.core.Graph.filter_nodes
.. automethod:: gli.graph.core.Graph.filter_edges

Iteration
~~~~~~~~~

.. automethod:: gli.graph.Graph.nodes
.. automethod:: gli.graph.Graph.edges
.. automethod:: gli.graph.Graph.edge_pairs

Factory Methods
~~~~~~~~~~~~~~~

.. automethod:: gli.graph.Graph.empty
.. automethod:: gli.graph.Graph.from_edge_list

Batch Operations
----------------

High-performance batch operations for efficient bulk processing.

Batch Filtering
~~~~~~~~~~~~~~~

.. automethod:: gli.graph.Graph.batch_filter_nodes
.. automethod:: gli.graph.Graph.batch_filter_edges

Batch Attribute Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automethod:: gli.graph.Graph.batch_get_node_attributes
.. automethod:: gli.graph.Graph.batch_set_node_attributes
.. automethod:: gli.graph.Graph.batch_update_node_attributes
.. automethod:: gli.graph.Graph.batch_set_edge_attributes

Subgraph Operations
~~~~~~~~~~~~~~~~~~~

.. automethod:: gli.graph.Graph.create_subgraph
.. automethod:: gli.graph.Graph.create_subgraph_fast
.. automethod:: gli.graph.Graph.get_connected_component
.. automethod:: gli.graph.Graph.get_k_hop_neighborhood

Batch Operation Context
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: gli.graph.BatchOperationContext
   :members:
   :undoc-members:
   :show-inheritance:

Performance Features
--------------------

The Graph class automatically selects the best available backend:

* **Rust Backend**: High-performance implementation for large graphs
* **Python Backend**: Fallback implementation with full feature compatibility

Backend selection can be controlled globally or per-instance:

.. code-block:: python

   from gli import Graph, set_backend
   
   # Global backend setting  
   set_backend('rust')
   g1 = Graph()  # Uses Rust backend
   
   # Per-instance backend
   g2 = Graph(backend='python')  # Forces Python backend
