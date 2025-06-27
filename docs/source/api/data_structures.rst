data_structures module
====================

.. automodule:: gli.data_structures
   :members:
   :undoc-members:
   :show-inheritance:

Core Data Structures
--------------------

Node Class
~~~~~~~~~~

.. autoclass:: gli.data_structures.Node
   :members:
   :undoc-members:
   :show-inheritance:

   The Node class represents a graph vertex with associated attributes.

   .. automethod:: __init__

   **Attribute Access**

   Nodes support dict-like attribute access:

   .. code-block:: python

      node = Node("alice", {"age": 30, "city": "NYC"})
      
      # Dict-like access
      print(node["age"])          # 30
      print(node.get("job", ""))  # "" (default)
      node["job"] = "Engineer"
      
      # Iteration
      for key in node:
          print(f"{key}: {node[key]}")

Edge Class
~~~~~~~~~~

.. autoclass:: gli.data_structures.Edge
   :members:
   :undoc-members:
   :show-inheritance:

   The Edge class represents a graph edge connecting two nodes.

   .. automethod:: __init__

   **Usage Example**

   .. code-block:: python

      edge = Edge("alice", "bob", {
          "relationship": "friends",
          "since": 2020,
          "strength": 0.9
      })
      
      print(edge.source)          # "alice"
      print(edge.target)          # "bob"
      print(edge["relationship"]) # "friends"

Attribute Management
-------------------

Both Node and Edge classes provide consistent attribute management:

**Common Methods**

.. automethod:: gli.data_structures.Node.get_attribute
.. automethod:: gli.data_structures.Node.set_attribute

**Dict-like Interface**

All data structures support Python's dict protocol:

* ``obj[key]`` - Get attribute value
* ``obj[key] = value`` - Set attribute value  
* ``key in obj`` - Check if attribute exists
* ``obj.get(key, default)`` - Get with default
* ``obj.keys()`` - Get all attribute names
* ``obj.values()`` - Get all attribute values
* ``obj.items()`` - Get key-value pairs

**Immutability**

Data structures support immutable updates:

.. code-block:: python

   node = Node("alice", {"age": 30})
   new_node = node.set_attribute("age", 31)  # Returns new instance
   # original node unchanged
