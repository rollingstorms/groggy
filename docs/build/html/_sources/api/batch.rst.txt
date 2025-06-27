batch module
============

.. automodule:: gli.graph.batch
   :members:
   :undoc-members:
   :show-inheritance:

BatchOperationContext
----------------------

The BatchOperationContext provides a context manager for efficient bulk operations.

.. autoclass:: gli.graph.batch.BatchOperationContext
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__
   .. automethod:: __enter__
   .. automethod:: __exit__

Batch Operations
~~~~~~~~~~~~~~~~

Node Operations
^^^^^^^^^^^^^^^

.. automethod:: gli.graph.batch.BatchOperationContext.add_node
.. automethod:: gli.graph.batch.BatchOperationContext.set_node_attributes

Edge Operations
^^^^^^^^^^^^^^^

.. automethod:: gli.graph.batch.BatchOperationContext.add_edge
.. automethod:: gli.graph.batch.BatchOperationContext.set_edge_attributes

Usage Example
~~~~~~~~~~~~~

.. code-block:: python

   from gli import Graph

   g = Graph()
   
   # Using batch context for efficient bulk operations
   with g.batch() as batch:
       # Add many nodes efficiently
       for i in range(1000):
           batch.add_node(f"node_{i}", value=i, category=f"cat_{i%10}")
       
       # Bulk attribute updates
       node_updates = {f"node_{i}": {"processed": True} for i in range(100)}
       batch.set_node_attributes(node_updates)

Performance Notes
~~~~~~~~~~~~~~~~~

* Batch operations are 10-100x faster than individual operations for large datasets
* The Rust backend optimizes memory allocation and reduces Python-Rust boundary crossings
* Use batch operations when adding/updating more than ~100 nodes/edges at once
