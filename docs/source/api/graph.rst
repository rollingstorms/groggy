graph module
============

.. automodule:: groggy.graph
   :members:
   :undoc-members:
   :show-inheritance:

Graph Class
-----------

The Graph class is the main interface for graph operations in Groggy. It provides a high-level Python API backed by a high-performance Rust implementation.

.. autoclass:: groggy.graph.core.Graph
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

Creation Methods
~~~~~~~~~~~~~~~~

.. automethod:: groggy.graph.core.Graph.empty
.. automethod:: groggy.graph.core.Graph.from_edge_list
.. automethod:: groggy.graph.core.Graph.from_node_list

Core Operations
~~~~~~~~~~~~~~~

Node Operations
^^^^^^^^^^^^^^^

.. automethod:: groggy.graph.core.Graph.add_node
.. automethod:: groggy.graph.core.Graph.get_node
.. automethod:: groggy.graph.core.Graph.get_node_ids
.. automethod:: groggy.graph.core.Graph.remove_node
.. automethod:: groggy.graph.core.Graph.node_count

Edge Operations
^^^^^^^^^^^^^^^

.. automethod:: groggy.graph.core.Graph.add_edge
.. automethod:: groggy.graph.core.Graph.get_edge
.. automethod:: groggy.graph.core.Graph.remove_edge
.. automethod:: groggy.graph.core.Graph.edge_count

Attribute Operations
~~~~~~~~~~~~~~~~~~~~

Single Attribute Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: groggy.graph.core.Graph.set_node_attribute
.. automethod:: groggy.graph.core.Graph.set_edge_attribute
.. automethod:: groggy.graph.core.Graph.get_node_attributes
.. automethod:: groggy.graph.core.Graph.get_edge_attributes

Multiple Attribute Operations
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: groggy.graph.core.Graph.set_node_attributes
.. automethod:: groggy.graph.core.Graph.set_edge_attributes
.. automethod:: groggy.graph.core.Graph.set_nodes_attributes_batch
.. automethod:: groggy.graph.core.Graph.set_edges_attributes_batch

Batch Operations
~~~~~~~~~~~~~~~~

New Simplified Batch API
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: groggy.graph.core.Graph.add_nodes
.. automethod:: groggy.graph.core.Graph.add_edges
.. automethod:: groggy.graph.core.Graph.update_node
.. automethod:: groggy.graph.core.Graph.update_edge
.. automethod:: groggy.graph.core.Graph.update_nodes
.. automethod:: groggy.graph.core.Graph.remove_nodes
.. automethod:: groggy.graph.core.Graph.remove_edges

Graph Queries and Navigation
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Neighbor Queries
^^^^^^^^^^^^^^^^

.. automethod:: groggy.graph.core.Graph.get_neighbors
.. automethod:: groggy.graph.core.Graph.get_outgoing_neighbors
.. automethod:: groggy.graph.core.Graph.get_incoming_neighbors
.. automethod:: groggy.graph.core.Graph.get_all_neighbors

Filtering Operations
^^^^^^^^^^^^^^^^^^^^

.. automethod:: groggy.graph.core.Graph.filter_nodes
.. automethod:: groggy.graph.core.Graph.filter_edges

Legacy Compatibility
^^^^^^^^^^^^^^^^^^^^

.. automethod:: groggy.graph.core.Graph.filter_nodes_by_attributes
.. automethod:: groggy.graph.core.Graph.filter_edges_by_attributes

Batch Operation Context
~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: groggy.graph.batch.BatchOperationContext
   :members:
   :undoc-members:
   :show-inheritance:

.. automethod:: groggy.graph.core.Graph.batch_operations

Performance Features
--------------------

The Graph class automatically selects the best available backend:

* **Rust Backend**: High-performance implementation for large graphs
* **Python Backend**: Fallback implementation with full feature compatibility

Backend selection can be controlled globally or per-instance:

.. code-block:: python

   from groggy import Graph, set_backend
   
   # Global backend setting  
   set_backend('rust')
   g1 = Graph()  # Uses Rust backend
   
   # Per-instance backend
   g2 = Graph(backend='python')  # Forces Python backend

API Examples
------------

Basic Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   from groggy import Graph
   
   # Create a new graph
   g = Graph()
   
   # Add nodes with attributes
   g.add_node("alice", age=30, role="engineer")
   g.add_node("bob", age=25, role="designer")
   
   # Add edges with attributes
   g.add_edge("alice", "bob", relationship="colleague", since=2020)
   
   # Query the graph
   print(f"Nodes: {g.node_count()}")
   print(f"Edges: {g.edge_count()}")
   print(f"Alice's neighbors: {g.get_neighbors('alice')}")

Batch Operations
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Efficient bulk operations
   nodes = [
       {'id': 'user_1', 'age': 25, 'role': 'engineer'},
       {'id': 'user_2', 'age': 30, 'role': 'manager'},
       {'id': 'user_3', 'age': 28, 'role': 'designer'}
   ]
   g.add_nodes(nodes)
   
   edges = [
       {'source': 'user_1', 'target': 'user_2', 'relationship': 'reports_to'},
       {'source': 'user_2', 'target': 'user_3', 'relationship': 'manages'}
   ]
   g.add_edges(edges)

Update Operations
~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Update single node/edge
   g.update_node("alice", age=31, department="engineering")
   g.update_edge("alice", "bob", strength=0.8, last_contact="2024-01-15")
   
   # Batch updates
   updates = {
       'user_1': {'salary': 80000, 'level': 'senior'},
       'user_2': {'salary': 90000, 'department': 'engineering'}
   }
   g.update_nodes(updates)

Filtering Operations
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Filter by function
   senior_employees = g.filter_nodes(
       lambda node_id, attrs: attrs.get('age', 0) > 30
   )
   
   # Filter by attributes
   managers = g.filter_nodes({'role': 'manager'})
   
   # Filter by query string
   high_earners = g.filter_nodes("salary > 85000 and role == 'engineer'")
   
   # Return subgraph instead of node list
   manager_subgraph = g.filter_nodes({'role': 'manager'}, return_graph=True)

Context Manager for Batch Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Use context manager for optimal performance
   with g.batch_operations() as batch:
       for i in range(1000):
           batch.add_node(f"node_{i}", value=i)
           if i > 0:
               batch.add_edge(f"node_{i-1}", f"node_{i}", weight=1.0)
       
       # Bulk attribute updates within context
       node_updates = {f"node_{i}": {"processed": True} for i in range(100)}
       batch.set_node_attributes(node_updates)

Migration from Old API
~~~~~~~~~~~~~~~~~~~~~~

The following methods have been simplified (removed ``_bulk`` suffix):

.. code-block:: python

   # Old API (deprecated)
   g.add_nodes_bulk(nodes_data)
   g.add_edges_bulk(edges_data)
   g.update_nodes_bulk(updates)
   
   # New API (recommended)
   g.add_nodes(nodes_data)
   g.add_edges(edges_data)
   g.update_nodes(updates)

The old method names are still supported for backward compatibility but will be removed in future versions.
