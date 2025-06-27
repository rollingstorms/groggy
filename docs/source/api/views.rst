views module
============

.. automodule:: groggy.views
   :members:
   :undoc-members:
   :show-inheritance:

Lazy Data Structures
--------------------

GLI provides memory-efficient lazy views that avoid copying data when possible.

LazyDict Class
~~~~~~~~~~~~~~

.. autoclass:: groggy.views.LazyDict
   :members:
   :undoc-members:
   :show-inheritance:

   Zero-copy dictionary view that combines base data with delta changes.

   .. automethod:: __init__

   **Usage Example**

   .. code-block:: python

      base_data = {"a": 1, "b": 2, "c": 3}
      
      # Create lazy view with modifications
      lazy_dict = LazyDict(
          base_dict=base_data,
          delta_added={"d": 4},
          delta_removed={"b"},
          delta_modified={"a": 10}
      )
      
      print(lazy_dict["a"])   # 10 (modified)
      print(lazy_dict["d"])   # 4 (added)
      # lazy_dict["b"]        # KeyError (removed)
      print(lazy_dict["c"])   # 3 (from base)

NodeView Class
~~~~~~~~~~~~~~

.. autoclass:: groggy.views.NodeView
   :members:
   :undoc-members:
   :show-inheritance:

   Provides a lazy view over graph nodes with efficient iteration.

EdgeView Class
~~~~~~~~~~~~~~

.. autoclass:: groggy.views.EdgeView
   :members:
   :undoc-members:
   :show-inheritance:

   Provides a lazy view over graph edges with efficient iteration.

Performance Benefits
-------------------

Lazy views provide several performance advantages:

**Memory Efficiency**
   - No data copying for read operations
   - Delta-based change tracking
   - Shared base data across views

**Computational Efficiency**
   - Deferred computation until needed
   - Efficient iteration patterns
   - Minimal memory allocations

**Use Cases**
   - Graph snapshots and versioning
   - Temporary modifications
   - Filter operations
   - Subgraph views

.. code-block:: python

   # Efficient subgraph filtering
   large_graph = Graph()
   # ... populate with millions of nodes ...
   
   # Create filtered view without copying data
   filtered_nodes = NodeView(
       base_nodes=large_graph.nodes,
       filter_func=lambda node: node.get("category") == "important"
   )
