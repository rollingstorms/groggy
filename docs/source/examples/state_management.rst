State Management and Branching
==============================

This section demonstrates Groggy's powerful state management and branching capabilities, similar to Git for graphs.

Basic State Management
----------------------

.. code-block:: python

   from groggy import Graph
   
   # Create a graph and add initial data
   g = Graph()
   g.add_node("alice", name="Alice", department="Engineering")
   g.add_node("bob", name="Bob", department="Design")
   g.add_edge("alice", "bob", relationship="collaborates")
   
   # Save the initial state
   initial_state = g.save_state("Initial team structure")
   print(f"Saved initial state: {initial_state}")
   
   # Make some changes
   g.add_node("charlie", name="Charlie", department="Engineering")
   g.add_edge("alice", "charlie", relationship="mentors")
   
   # Save the updated state
   updated_state = g.save_state("Added Charlie to engineering team")
   
   # View state information
   print(f"Total states: {len(g.states['state_hashes'])}")
   print(f"Current state: {g.states['current_hash']}")
   
   # Go back to initial state
   g.load_state(initial_state)
   print(f"Nodes after rollback: {len(g.nodes)}")  # Should be 2

Branch Management
-----------------

Create and manage different versions of your graph:

.. code-block:: python

   from groggy import Graph
   
   # Start with a base graph
   g = Graph()
   g.add_node("main_server", type="server", status="running")
   g.add_node("db", type="database", status="active")
   g.add_edge("main_server", "db", connection="tcp")
   
   # Save main branch state
   main_state = g.save_state("Production infrastructure")
   
   # Create development branch
   g.create_branch("development")
   g.switch_branch("development")
   
   # Add development-specific changes
   g.add_node("test_server", type="server", status="testing")
   g.add_node("cache", type="redis", status="active")
   g.add_edge("test_server", "cache", connection="redis")
   g.save_state("Added test infrastructure")
   
   # Create feature branch from development
   g.create_branch("feature/monitoring", switch=True)
   g.add_node("monitor", type="monitoring", status="active")
   g.add_edge("monitor", "main_server", monitors=True)
   g.add_edge("monitor", "db", monitors=True)
   g.save_state("Added monitoring system")
   
   # View all branches
   print("Available branches:")
   for branch, state_hash in g.branches.items():
       print(f"  {branch}: {state_hash}")

Advanced Branching Workflows
-----------------------------

Implement complex workflows with multiple branches:

.. code-block:: python

   from groggy import Graph
   import time
   
   # Initialize project graph
   project = Graph()
   
   # Main branch: stable production code
   project.add_node("core", component="main", version="1.0.0")
   project.add_node("api", component="interface", version="1.0.0")
   project.add_edge("api", "core", depends_on=True)
   
   main_state = project.save_state("Release 1.0.0")
   
   # Development branch: new features
   project.create_branch("development", switch=True)
   project.add_node("analytics", component="analysis", version="1.1.0-dev")
   project.add_edge("analytics", "core", depends_on=True)
   dev_state = project.save_state("Added analytics module")
   
   # Feature branch: specific feature development
   project.create_branch("feature/user-auth", switch=True)
   project.add_node("auth", component="authentication", version="1.1.0-dev")
   project.add_edge("auth", "api", integrates_with=True)
   project.add_edge("auth", "core", depends_on=True)
   feature_state = project.save_state("Implemented user authentication")
   
   # Hotfix branch: critical fixes
   project.switch_branch("main")
   project.create_branch("hotfix/security-patch", switch=True)
   project.set_node_attribute("core", "version", "1.0.1")
   project.set_node_attribute("core", "security_patch", True)
   hotfix_state = project.save_state("Security patch 1.0.1")
   
   # Demonstrate rapid switching between contexts
   contexts = ["main", "development", "feature/user-auth", "hotfix/security-patch"]
   
   print("Testing rapid context switching:")
   start_time = time.time()
   
   for i in range(10):
       for context in contexts:
           project.switch_branch(context)
           nodes_count = len(project.nodes)
           print(f"  {context}: {nodes_count} nodes")
   
   switch_time = time.time() - start_time
   total_switches = 10 * len(contexts)
   print(f"\\nCompleted {total_switches} branch switches in {switch_time:.3f}s")
   print(f"Average: {switch_time/total_switches:.3f}s per switch")

State Persistence and Recovery
------------------------------

Handle state persistence and recovery scenarios:

.. code-block:: python

   from groggy import Graph
   
   # Create a complex graph with multiple states
   g = Graph()
   
   # Build initial network
   nodes = [(f"node_{i}", {"value": i, "layer": i // 10}) for i in range(100)]
   edges = [(f"node_{i}", f"node_{i+1}", {"weight": i}) for i in range(99)]
   
   # Use efficient batch operations
   node_attrs = {node_id: attrs for node_id, attrs in nodes}
   for node_id in node_attrs:
       g.add_node(node_id)
   g.set_nodes_attributes_batch(node_attrs)
   
   for source, target, attrs in edges:
       g.add_edge(source, target, **attrs)
   
   # Create multiple save points
   save_points = []
   
   # Save every 10 modifications
   for i in range(0, 100, 10):
       # Modify some nodes
       updates = {f"node_{j}": {"modified": True, "iteration": i} 
                 for j in range(i, min(i+10, 100))}
       g.set_nodes_attributes_batch(updates)
       
       state_hash = g.save_state(f"Batch modification {i//10 + 1}")
       save_points.append(state_hash)
       print(f"Saved checkpoint at modification {i}: {state_hash}")
   
   # Demonstrate recovery to any save point
   print(f"\\nTotal save points: {len(save_points)}")
   
   # Go back to middle save point
   middle_point = save_points[len(save_points)//2]
   g.load_state(middle_point)
   print(f"Recovered to middle save point: {middle_point}")
   
   # Verify state
   modified_nodes = g.filter_nodes({"modified": True})
   print(f"Modified nodes after recovery: {len(modified_nodes)}")

Performance Monitoring
-----------------------

Monitor state management performance:

.. code-block:: python

   from groggy import Graph
   import time
   
   # Create test graph
   g = Graph()
   
   # Add substantial data
   print("Creating large graph...")
   start = time.time()
   
   # 10K nodes with attributes
   for i in range(10000):
       g.add_node(f"n_{i}", value=i, category=f"cat_{i%10}")
   
   # 10K edges
   for i in range(10000):
       source = f"n_{i}"
       target = f"n_{(i+1)%10000}"
       g.add_edge(source, target, weight=i%100)
   
   creation_time = time.time() - start
   print(f"Graph creation: {creation_time:.2f}s")
   
   # Test state operations performance
   print("\\nTesting state operations:")
   
   # Save state
   save_start = time.time()
   state_hash = g.save_state("Large graph state")
   save_time = time.time() - save_start
   print(f"Save state: {save_time:.3f}s")
   
   # Create branch
   branch_start = time.time()
   g.create_branch("performance_test")
   branch_time = time.time() - branch_start
   print(f"Create branch: {branch_time:.3f}s")
   
   # Switch branch
   switch_start = time.time()
   g.switch_branch("performance_test")
   switch_time = time.time() - switch_start
   print(f"Switch branch: {switch_time:.3f}s")
   
   # Load state
   load_start = time.time()
   g.load_state(state_hash)
   load_time = time.time() - load_start
   print(f"Load state: {load_time:.3f}s")
   
   # Get storage stats
   stats = g.get_storage_stats()
   print(f"\\nStorage statistics:")
   for key, value in stats.items():
       print(f"  {key}: {value}")

Tips and Best Practices
------------------------

**State Management Tips:**

1. **Regular Checkpoints**: Save states at logical milestones
2. **Descriptive Messages**: Use clear, descriptive commit messages
3. **Branch Naming**: Use consistent branch naming conventions
4. **Performance**: Batch operations before saving states for efficiency

**Branching Strategies:**

1. **Feature Branches**: Create branches for major features
2. **Hotfix Branches**: Use dedicated branches for critical fixes
3. **Development Branch**: Maintain a stable development branch
4. **Testing**: Use branches for experimental changes

**Performance Optimization:**

1. **Batch Updates**: Use batch operations for multiple changes
2. **Strategic Saves**: Don't save state after every small change
3. **Memory Management**: Monitor storage stats for large graphs
4. **Clean Branches**: Remove unused branches to save memory

.. code-block:: python

   # Example: Efficient workflow
   g = Graph()
   
   # 1. Batch initial setup
   with g.batch() as batch:
       for i in range(1000):
           batch.add_node(f"node_{i}", initial=True)
   
   # 2. Save after major changes
   g.save_state("Initial setup complete")
   
   # 3. Use branches for experiments
   g.create_branch("experiment", switch=True)
   
   # 4. Batch modifications
   updates = {f"node_{i}": {"processed": True} for i in range(100)}
   g.set_nodes_attributes_batch(updates)
   
   # 5. Save milestone
   g.save_state("Processed first 100 nodes")
