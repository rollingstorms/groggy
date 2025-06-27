delta module
============

.. automodule:: groggy.delta
   :members:
   :undoc-members:
   :show-inheritance:

Delta Operations
---------------

GLI's delta system provides efficient change tracking and manipulation.

Classes
~~~~~~~

.. autoclass:: groggy.delta.Delta
   :members:
   :undoc-members:
   :show-inheritance:

   Represents a set of changes to a graph.

.. autoclass:: groggy.delta.Operation
   :members:
   :undoc-members:
   :show-inheritance:

   Represents a single operation in a delta.

Core Concepts
~~~~~~~~~~~~~

**Delta Composition**

Deltas can be composed to represent complex changes:

.. code-block:: python

   from groggy.delta import Delta
   
   # Create individual deltas
   delta1 = Delta()
   delta1.add_node("alice", {"name": "Alice"})
   
   delta2 = Delta()  
   delta2.add_node("bob", {"name": "Bob"})
   delta2.add_edge("alice", "bob", {"relationship": "friends"})
   
   # Compose deltas
   combined = delta1.compose(delta2)
   
   # Apply to graph
   g.apply_delta(combined)

**Delta Inversion**

Deltas can be inverted to undo changes:

.. code-block:: python

   # Create and apply delta
   delta = Delta()
   delta.add_node("temp", {"temporary": True})
   g.apply_delta(delta)
   
   # Invert to undo
   inverse_delta = delta.invert()
   g.apply_delta(inverse_delta)  # Removes the temporary node

**Delta Serialization**

Deltas can be serialized for persistence or network transfer:

.. code-block:: python

   # Serialize delta
   delta_data = delta.serialize()
   
   # Deserialize delta
   restored_delta = Delta.deserialize(delta_data)

Advanced Operations
------------------

**Conflict Detection**

.. autofunction:: groggy.delta.detect_conflicts

**Delta Optimization**

.. autofunction:: groggy.delta.optimize_delta

**Patch Application**

.. code-block:: python

   # Create patch from two graph states
   patch = g1.diff(g2)
   
   # Apply patch to transform g1 into g2
   g1.apply_delta(patch)
   
   # Verify transformation
   assert g1.equals(g2)

Performance Considerations
-------------------------

**Efficient Delta Operations**

* Use batch operations when possible
* Compose deltas before applying
* Optimize deltas to remove redundant operations

**Memory Management**

* Large deltas can consume significant memory
* Consider streaming application for very large changes
* Use delta compression for long-term storage

.. code-block:: python

   # Efficient batch delta creation
   with g.batch_operations() as batch:
       delta = Delta()
       for i in range(1000):
           delta.add_node(f"node_{i}", {"index": i})
       batch.apply_delta(delta)

Use Cases
---------

**Graph Versioning**

Track changes over time:

.. code-block:: python

   # Track all changes
   changes = []
   
   # Make changes and record deltas
   delta1 = g.add_node("alice")
   changes.append(delta1)
   
   delta2 = g.add_edge("alice", "bob")
   changes.append(delta2)
   
   # Replay changes
   new_graph = Graph()
   for delta in changes:
       new_graph.apply_delta(delta)

**Distributed Graph Processing**

Synchronize changes across systems:

.. code-block:: python

   # System A creates delta
   local_changes = Delta()
   local_changes.add_node("new_node")
   
   # Serialize and send to System B
   delta_data = local_changes.serialize()
   send_to_system_b(delta_data)
   
   # System B applies changes
   remote_delta = Delta.deserialize(delta_data)
   system_b_graph.apply_delta(remote_delta)
