state module
============

.. automodule:: gli.graph.state
   :members:
   :undoc-members:
   :show-inheritance:

State Management
----------------

GLI provides Git-like state management for graph versioning and branching, backed by the high-performance Rust backend.

StateMixin Class
~~~~~~~~~~~~~~~~

.. autoclass:: gli.graph.state.StateMixin
   :members:
   :undoc-members:
   :show-inheritance:

State Operations
~~~~~~~~~~~~~~~~

Saving and Loading States
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automethod:: gli.graph.state.StateMixin.save_state
.. automethod:: gli.graph.state.StateMixin.load_state
.. automethod:: gli.graph.state.StateMixin.commit
.. automethod:: gli.graph.state.StateMixin.get_state_info

Branch Management
^^^^^^^^^^^^^^^^^

.. automethod:: gli.graph.state.StateMixin.create_branch
.. automethod:: gli.graph.state.StateMixin.switch_branch

Storage Statistics
^^^^^^^^^^^^^^^^^^

.. automethod:: gli.graph.state.StateMixin.get_storage_stats

Lazy Properties
~~~~~~~~~~~~~~~

State Information
^^^^^^^^^^^^^^^^^

The ``states`` property provides lazy-loaded access to state information:

.. code-block:: python

   # Access state information
   print(f"Total states: {len(g.states['state_hashes'])}")
   print(f"Current state: {g.states['current_hash']}")
   print(f"Auto states: {g.states['auto_states']}")

Branch Information
^^^^^^^^^^^^^^^^^^

The ``branches`` property provides lazy-loaded access to branch information:

.. code-block:: python

   # Access branch information
   print(f"Available branches: {list(g.branches.keys())}")
   for branch_name, state_hash in g.branches.items():
       print(f"Branch '{branch_name}' points to state {state_hash}")

Usage Examples
~~~~~~~~~~~~~~

**Basic State Management**

.. code-block:: python

   from gli import Graph
   
   g = Graph()
   g.add_node("alice", name="Alice", age=30)
   g.add_node("bob", name="Bob", age=25)
   g.add_edge("alice", "bob", relationship="friends")
   
   # Save initial state
   initial_hash = g.save_state("Initial graph with Alice and Bob")
   
   # Make changes
   g.add_node("charlie", name="Charlie", age=35)
   g.save_state("Added Charlie")
   
   # Load previous state
   g.load_state(initial_hash)
   print(f"Nodes after loading: {len(g.nodes)}")  # Should be 2

**Branch Management**

.. code-block:: python

   # Create and work with branches
   g.create_branch("feature/social_network")
   g.switch_branch("feature/social_network")
   
   # Add features in the branch
   g.add_node("diana", name="Diana", role="admin")
   g.save_state("Added admin user")
   
   # Switch back to main
   g.switch_branch("main")
   
   # Merge changes (manual for now)
   g.switch_branch("feature/social_network")
   diana_attrs = g.get_node_attributes("diana")
   g.switch_branch("main")
   g.add_node("diana", **diana_attrs)

**Performance Monitoring**

.. code-block:: python

   # Get storage statistics
   stats = g.get_storage_stats()
   print(f"Total states: {stats.get('total_states', 0)}")
   print(f"Memory usage: {stats.get('memory_usage', 'N/A')}")

Performance Notes
~~~~~~~~~~~~~~~~~

* State operations are backed by the Rust backend for optimal performance
* Branch switching typically takes 0.1-0.2 seconds for graphs with 100K+ nodes
* States use content-addressed storage with deduplication for memory efficiency
* Lazy-loaded properties minimize overhead when accessing state information
   snapshot = g.snapshot()
   
   # Continue modifying original
   g.add_node(name="Bob")
   
   # Snapshot remains unchanged
   print(snapshot.node_count())  # 1
   print(g.node_count())         # 2

**Branching Workflow**

.. code-block:: python

   # Create branch for experimental feature
   experimental_branch = g.create_branch("feature_experiment")
   
   # Work on experimental branch
   g.switch_branch("feature_experiment")
   g.add_node(name="Experimental")
   
   # Switch back to main
   g.switch_branch("main")
   # Experimental changes not visible

**Merging Changes**

.. code-block:: python

   # Merge experimental branch back
   g.merge_branch("feature_experiment")
   
   # Now experimental changes are in main
   print("Experimental" in g.nodes)  # True

State Persistence
-----------------

Graph state can be persisted and restored:

.. code-block:: python

   # Save current state
   state_data = g.export_state()
   
   # Restore from saved state
   restored_graph = Graph.from_state(state_data)

Advanced Features
-----------------

**Conflict Resolution**

When merging branches, GLI provides conflict resolution strategies:

.. code-block:: python

   # Merge with conflict resolution
   g.merge_branch("feature_branch", 
                  strategy="latest_wins")  # or "manual", "merge_attributes"

**State Inspection**

.. code-block:: python

   # Get current branch info
   current_branch = g.current_branch()
   print(f"On branch: {current_branch.name}")
   
   # List all branches
   for branch in g.list_branches():
       print(f"Branch: {branch.name}, Commits: {len(branch.commits)}")
   
   # Get commit history
   for commit in g.get_commit_history():
       print(f"Commit: {commit.id}, Message: {commit.message}")
