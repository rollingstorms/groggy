state module
============

.. automodule:: gli.state
   :members:
   :undoc-members:
   :show-inheritance:

State Management
---------------

GLI provides sophisticated state management for graph versioning and branching.

Classes
~~~~~~~

.. autoclass:: gli.state.GraphState
   :members:
   :undoc-members:
   :show-inheritance:

   Manages graph state with support for branching and versioning.

.. autoclass:: gli.state.Branch
   :members:
   :undoc-members:
   :show-inheritance:

   Represents a branch in the graph's version history.

.. autoclass:: gli.state.Commit
   :members:
   :undoc-members:
   :show-inheritance:

   Represents a commit in the graph's version history.

Versioning Operations
~~~~~~~~~~~~~~~~~~~~

**Creating Snapshots**

.. code-block:: python

   from gli import Graph
   
   g = Graph()
   g.add_node(name="Alice")
   
   # Create snapshot
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
