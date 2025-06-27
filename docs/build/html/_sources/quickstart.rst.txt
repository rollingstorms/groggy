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

   # Add a node with attributes
   alice = g.add_node(name="Alice", age=30, city="New York")
   bob = g.add_node(name="Bob", age=25, city="Boston")
   charlie = g.add_node(name="Charlie", age=35, city="Chicago")
   
   print(f"Added nodes: {[alice, bob, charlie]}")

Adding Edges
~~~~~~~~~~~~

.. code-block:: python

   # Add edges with attributes
   friendship1 = g.add_edge(alice, bob, 
                           relationship="friends",
                           since=2020,
                           strength=0.9)
   
   friendship2 = g.add_edge(bob, charlie,
                           relationship="colleagues", 
                           since=2019,
                           strength=0.7)

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
   
   # Node degrees
   alice_degree = g.degree(alice)
   print(f"Alice's degree: {alice_degree}")

High-Performance Batch Operations
---------------------------------

For large graphs, GLI provides efficient batch operations that are **30-40x faster** than individual operations.

Batch Filtering
~~~~~~~~~~~~~~~

.. code-block:: python

   # Create a larger graph for demonstration
   g = Graph(backend='rust')  # Use Rust backend for performance
   
   # Add many people
   people = []
   cities = ['New York', 'Boston', 'Chicago', 'San Francisco']
   occupations = ['Engineer', 'Teacher', 'Doctor', 'Artist']
   
   for i in range(1000):
       person_id = g.add_node(
           name=f"Person_{i}",
           age=random.randint(20, 60),
           city=random.choice(cities),
           occupation=random.choice(occupations)
       )
       people.append(person_id)
   
   # Efficient batch filtering
   engineers = g.batch_filter_nodes(occupation='Engineer')
   ny_residents = g.batch_filter_nodes(city='New York')
   
   print(f"Found {len(engineers)} engineers")
   print(f"Found {len(ny_residents)} New York residents")

Batch Attribute Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Get attributes for multiple nodes efficiently
   engineer_attrs = g.batch_get_node_attributes(engineers[:10])
   
   # Show details
   for i, attrs in enumerate(engineer_attrs):
       print(f"{attrs['name']}: age {attrs['age']}, lives in {attrs['city']}")
   
   # Bulk attribute updates
   updates = {
       person_id: {'status': 'active', 'last_updated': '2025-01-15'}
       for person_id in engineers[:5]
   }
   g.batch_set_node_attributes(updates)

Performance Benefits
~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   # Compare individual vs batch operations
   target_nodes = list(g.nodes)[:100]
   
   # Individual operations (slower)
   start = time.time()
   individual_results = []
   for node_id in target_nodes:
       node = g.get_node(node_id)
       if node.get('occupation') == 'Engineer':
           individual_results.append(node_id)
   individual_time = time.time() - start
   
   # Batch operations (much faster)
   start = time.time()
   batch_results = g.batch_filter_nodes(occupation='Engineer')
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
   from gli import create_random_graph
   
   # Create test graphs with different backends
   start = time.time()
   g_rust = create_random_graph(1000, 5000, backend='rust')
   rust_time = time.time() - start
   
   start = time.time()  
   g_python = create_random_graph(1000, 5000, backend='python')
   python_time = time.time() - start
   
   print(f"Rust backend: {rust_time:.3f}s")
   print(f"Python backend: {python_time:.3f}s")

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

   # Iterate over all edges
   for edge_id in g.edges:
       edge_data = g.get_edge_by_id(edge_id)
       print(f"Edge {edge_id}: {edge_data}")
   
   # Iterate over edges with source/target
   for source, target in g.edge_pairs():
       edge_data = g.get_edge(source, target)
       print(f"{source} -> {target}: {edge_data}")

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
