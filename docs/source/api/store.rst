store module
============

.. automodule:: groggy.store
   :members:
   :undoc-members:
   :show-inheritance:

GraphStore Class
----------------

The GraphStore class provides persistent storage capabilities for graphs.

.. autoclass:: groggy.store.GraphStore
   :members:
   :undoc-members:
   :show-inheritance:

Storage Operations
~~~~~~~~~~~~~~~~~~

.. automethod:: groggy.store.GraphStore.save
.. automethod:: groggy.store.GraphStore.load
.. automethod:: groggy.store.GraphStore.exists

Serialization
~~~~~~~~~~~~~

GLI supports multiple serialization formats:

* **JSON**: Human-readable format, good for small graphs
* **Binary**: Compact format for large graphs
* **Custom**: Optimized format preserving all GLI features

.. code-block:: python

   from groggy import Graph, GraphStore
   
   # Create and populate graph
   g = Graph()
   alice = g.add_node(name="Alice", age=30)
   bob = g.add_node(name="Bob", age=25)
   g.add_edge(alice, bob, relationship="friends")
   
   # Save to different formats
   store = GraphStore()
   store.save(g, "graph.json", format="json")
   store.save(g, "graph.bin", format="binary")
   
   # Load graph
   loaded_graph = store.load("graph.json")
