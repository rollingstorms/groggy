Quick Start Guide
=================

This guide will get you up and running with GLI in just a few minutes.

Basic Graph Operations
----------------------

Creating a Graph
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gli import Graph
   
   # Create a new graph
   g = Graph()
   
   # Check which backend is being used
   print(f"Using backend: {g.backend}")

Adding Nodes
~~~~~~~~~~~~

.. code-block:: python

   # Add single nodes with attributes
   alice = g.add_node(name="Alice", age=30, city="New York")
   bob = g.add_node(name="Bob", age=25, city="Boston")
   charlie = g.add_node(name="Charlie", age=35, city="Chicago")
   
   print(f"Added nodes: {[alice, bob, charlie]}")

   # Add multiple nodes efficiently with add_nodes()
   team_data = [
       {'id': 'emp001', 'name': 'Diana', 'age': 28, 'role': 'engineer'},
       {'id': 'emp002', 'name': 'Eve', 'age': 32, 'role': 'designer'},
       {'id': 'emp003', 'name': 'Frank', 'age': 29, 'role': 'analyst'}
   ]
   g.add_nodes(team_data)  # Much faster for many nodes

Adding Edges
~~~~~~~~~~~~

.. code-block:: python

   # Add single edges with attributes
   friendship1 = g.add_edge(alice, bob, 
                           relationship="friends",
                           since=2020,
                           strength=0.9)
   
   friendship2 = g.add_edge(bob, charlie,
                           relationship="colleagues", 
                           since=2019,
                           strength=0.7)

   # Add multiple edges efficiently with add_edges()
   connections = [
       {'source': alice, 'target': 'emp001', 'relationship': 'mentor', 'frequency': 'weekly'},
       {'source': bob, 'target': 'emp002', 'relationship': 'collaborator', 'frequency': 'daily'},
       {'source': charlie, 'target': 'emp003', 'relationship': 'manager', 'frequency': 'weekly'}
   ]
   g.add_edges(connections)  # Much faster for many edges

Querying the Graph
~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get node attributes
   alice_node = g.get_node(alice)
   print(f"Alice's data: {alice_node}")
   print(f"Alice's age: {alice_node['age']}")
   
   # Get neighbors
   alice_neighbors = g.get_neighbors(alice)
   print(f"Alice's neighbors: {alice_neighbors}")
   
   # Get edge attributes
   edge_data = g.get_edge(alice, bob)
   print(f"Alice-Bob relationship: {edge_data}")

Graph Statistics
~~~~~~~~~~~~~~~~

.. code-block:: python

   # Basic statistics
   print(f"Number of nodes: {g.node_count()}")
   print(f"Number of edges: {g.edge_count()}")
   
   # Get neighbors to calculate degree manually
   alice_neighbors = g.get_neighbors(alice)
   print(f"Alice's degree: {len(alice_neighbors)}")

Modern Update API
~~~~~~~~~~~~~~~~~

GLI provides multiple ways to update node and edge attributes with a clean, intuitive interface:

.. code-block:: python

   # Update single node - multiple ways
   g.update_node(alice, age=31, title="Senior Engineer")  # Keyword args
   g.update_node(alice, {"salary": 85000, "department": "AI Research"})  # Dict
   g.update_node(alice, {"level": "L5"}, bonus_eligible=True)  # Both combined!

   # Update single edge - same flexibility  
   g.update_edge(alice, bob, strength=0.95, last_contact="2024-01-15")
   g.update_edge(alice, bob, {"duration": "4 years"}, active=True)

   # Efficient bulk updates for large operations
   salary_updates = {
       alice: {"salary": 90000, "promotion": "2024-01"},
       bob: {"salary": 75000, "department": "UX Design"}, 
       charlie: {"salary": 95000, "title": "Engineering Manager"}
   }
   g.update_nodes(salary_updates)  # Update thousands of nodes efficiently

   # Backwards compatibility - old methods still work
   g.set_node_attribute(alice, "status", "active")
   g.set_node_attributes(bob, {"team": "core", "level": "senior"})

High-Performance Batch Operations
---------------------------------

For large graphs, GLI provides efficient batch operations that are **10-100x faster** than individual operations.

Efficient Filtering Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a larger graph for demonstration
   import random
   g = Graph(backend='rust')  # Use Rust backend for performance
   
   # Add many people using batch operations
   people_data = []
   cities = ['New York', 'Boston', 'Chicago', 'San Francisco']
   occupations = ['Engineer', 'Teacher', 'Doctor', 'Artist']
   
   for i in range(1000):
       people_data.append({
           'id': f"person_{i}",
           'name': f"Person_{i}",
           'age': random.randint(20, 60),
           'city': random.choice(cities),
           'occupation': random.choice(occupations)
       })
   
   # Add all people efficiently
   g.add_nodes(people_data)
   
   # Efficient filtering using the new API
   engineers = g.filter_nodes({'occupation': 'Engineer'})
   ny_residents = g.filter_nodes({'city': 'New York'})
   senior_engineers = g.filter_nodes(
       lambda node_id, attrs: attrs.get('occupation') == 'Engineer' and attrs.get('age', 0) > 40
   )
   
   print(f"Found {len(engineers)} engineers")
   print(f"Found {len(ny_residents)} New York residents")
   print(f"Found {len(senior_engineers)} senior engineers")

Bulk Attribute Updates
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Prepare bulk updates for multiple nodes
   updates = {}
   for node_id in engineers[:10]:  # Update first 10 engineers
       updates[node_id] = {
           'status': 'active', 
           'last_updated': '2025-01-15',
           'department': 'engineering'
       }
   
   # Apply all updates efficiently
   g.update_nodes(updates)

Performance Benefits
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   # Compare individual vs batch operations
   # Create sample data
   many_updates = {}
   for i, node_id in enumerate(list(g.get_node_ids())[:100]):
       many_updates[node_id] = {'processed': True, 'batch_id': i}
   
   # Individual operations (slower)
   start = time.time()
   for node_id, attrs in many_updates.items():
       g.update_node(node_id, attrs)
   individual_time = time.time() - start
   
   # Batch operations (much faster)
   start = time.time()
   g.update_nodes(many_updates)
   batch_time = time.time() - start
   
   print(f"Individual: {individual_time:.4f}s")
   print(f"Batch: {batch_time:.4f}s")
   print(f"Speedup: {individual_time/batch_time:.1f}x")

Working with Attributes
-----------------------

Complex Attributes
~~~~~~~~~~~~~~~~~~

GLI supports complex nested attributes:

.. code-block:: python

   # Add node with complex attributes
   person = g.add_node(
       name="David",
       contact={
           "email": "david@example.com",
           "phone": "+1-555-0123"
       },
       skills=["Python", "Rust", "Graph Theory"],
       metadata={
           "created_at": "2025-01-01",
           "source": "manual_entry"
       }
   )
   
   # Access nested attributes
   node_data = g.get_node(person)
   print(f"Email: {node_data['contact']['email']}")
   print(f"Skills: {node_data['skills']}")

Updating Attributes
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Update node attributes
   g.update_node(alice, location="San Francisco", age=31)
   
   # Update edge attributes  
   g.update_edge(alice, bob, strength=0.95, last_contact="2025-01-15")

Backend Selection
-----------------

Choosing Backends
~~~~~~~~~~~~~~~~~

.. code-block:: python

   from gli import Graph, set_backend, get_available_backends
   
   # Check available backends
   print(f"Available backends: {get_available_backends()}")
   
   # Set global backend preference
   set_backend('rust')  # or 'python'
   
   # Create graph with specific backend
   g_rust = Graph(backend='rust')
   g_python = Graph(backend='python')

Performance Comparison
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   # Test with smaller graph sizes for demo
   def time_graph_creation(backend, num_nodes=100, num_edges=200):
       start = time.time()
       g = Graph(backend=backend)
       
       # Add nodes
       nodes_data = [{'id': f'node_{i}', 'value': i} for i in range(num_nodes)]
       g.add_nodes(nodes_data)
       
       # Add edges
       import random
       edges_data = []
       for _ in range(num_edges):
           source = f'node_{random.randint(0, num_nodes-1)}'
           target = f'node_{random.randint(0, num_nodes-1)}'
           if source != target:
               edges_data.append({'source': source, 'target': target, 'weight': random.random()})
       g.add_edges(edges_data)
       
       return time.time() - start
   
   rust_time = time_graph_creation('rust')
   python_time = time_graph_creation('python')
   
   print(f"Rust backend: {rust_time:.3f}s")
   print(f"Python backend: {python_time:.3f}s")
   print(f"Rust is {python_time/rust_time:.1f}x faster")

Batch Operations
----------------

For better performance when adding many nodes/edges:

.. code-block:: python

   # Use batch operations for efficiency
   with g.batch_operations() as batch:
       for i in range(1000):
           node_id = batch.add_node(value=i, category="batch")
           if i > 0:
               batch.add_edge(f"node_{i-1}", node_id, weight=1.0)

Graph Iteration
---------------

Iterating Over Nodes
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Iterate over all nodes
   for node_id in g.nodes:
       node_data = g.get_node(node_id)
       print(f"Node {node_id}: {node_data}")

Iterating Over Edges
~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Iterate over all edges using the edge view
   for edge_id, edge in g.edges.items():
       print(f"Edge {edge_id}: {edge.source} -> {edge.target}, attrs: {edge.attributes}")
   
   # Get all edge IDs and iterate
   for edge_id in g.edges:
       edge = g.edges[edge_id]
       print(f"{edge.source} -> {edge.target}: {edge.attributes}")

Error Handling
--------------

GLI provides comprehensive error handling:

.. code-block:: python

   try:
       # This will raise an error if node doesn't exist
       node_data = g.get_node("nonexistent_node")
   except KeyError as e:
       print(f"Node not found: {e}")
   
   try:
       # This will raise an error if edge already exists
       g.add_edge(alice, bob)  # assuming this edge already exists
   except ValueError as e:
       print(f"Edge creation failed: {e}")

Next Steps
----------

Now that you've learned the basics, explore:

- :doc:`api/index` - Complete API reference
- :doc:`examples/index` - More complex examples
- :doc:`performance` - Performance optimization tips
- :doc:`architecture` - Understanding GLI's architecture
