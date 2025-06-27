Basic Usage Examples
====================

This section demonstrates basic Groggy operations with practical examples using the new high-performance API.

Creating Your First Graph
--------------------------

.. code-block:: python

   from groggy import Graph
   
   # Create a new graph (automatically uses Rust backend)
   g = Graph()
   
   # Add some people (supports both string and integer IDs)
   alice = g.add_node("alice", name="Alice", age=30, occupation="Engineer")
   bob = g.add_node(1, name="Bob", age=25, occupation="Designer") 
   charlie = g.add_node("charlie", name="Charlie", age=35, occupation="Manager")
   
   # Add relationships with the new API
   g.add_edge(alice, bob, relationship="colleagues", since=2020)
   g.add_edge(bob, charlie, relationship="friends", since=2018)
   g.add_edge(alice, charlie, relationship="reports_to", since=2021)
   
   print(f"Created graph with {g.node_count()} nodes and {g.edge_count()} edges")
   
   # Access nodes and edges with lazy loading
   print(f"Alice's details: {g.nodes[alice].attributes}")
   
   # Get edge with new (source, target) API
   relationship = g.get_edge(alice, bob)
   print(f"Alice-Bob relationship: {relationship.attributes}")

Working with Mixed ID Types
----------------------------

Groggy seamlessly handles both string and integer node IDs:

.. code-block:: python

   from groggy import Graph
   
   g = Graph()
   
   # Mix string and integer IDs freely
   user_alice = g.add_node("user_alice", name="Alice", type="user")
   server_1 = g.add_node(1001, name="Database Server", type="server")
   process_42 = g.add_node(42, name="Auth Process", type="process")
   
   # Connect different ID types
   g.add_edge("user_alice", 1001, action="connects_to")
   g.add_edge(1001, 42, action="spawns")
   
   # Query works with any ID type
   neighbors = g.get_neighbors("user_alice")
   print(f"Alice connects to: {neighbors}")

Social Network Analysis
-----------------------

Building a social network and analyzing connections:

.. code-block:: python

   from groggy import Graph
   
   # Create social network
   social_net = Graph()
   
   # Add users with profiles
   users = [
       ("alice", {"name": "Alice Smith", "age": 28, "city": "NYC", "interests": ["tech", "art"]}),
       ("bob", {"name": "Bob Jones", "age": 32, "city": "SF", "interests": ["music", "sports"]}),
       ("carol", {"name": "Carol Brown", "age": 26, "city": "LA", "interests": ["art", "travel"]}),
       ("dave", {"name": "Dave Wilson", "age": 30, "city": "NYC", "interests": ["tech", "music"]}),
   ]
   
   # Efficient batch addition using new batch API
   user_attrs = {user_id: profile for user_id, profile in users}
   
   # Add all nodes first
   for user_id in user_attrs:
       social_net.add_node(user_id)
   
   # Then set all attributes in batch (much faster)
   social_net.set_nodes_attributes_batch(user_attrs)
           batch.add_node(user_id, **profile)
   
   # Add friendships with metadata
   friendships = [
       ("alice", "bob", {"type": "close_friend", "met": "college", "strength": 0.9}),
       ("alice", "carol", {"type": "friend", "met": "work", "strength": 0.7}),
       ("bob", "dave", {"type": "friend", "met": "online", "strength": 0.6}),
       ("carol", "dave", {"type": "acquaintance", "met": "party", "strength": 0.4}),
   ]
   
   with social_net.batch_operations() as batch:
       for user1, user2, metadata in friendships:
           batch.add_edge(user1, user2, **metadata)
   
   # Analyze the network
   print("Social Network Analysis:")
   print(f"Total users: {social_net.node_count()}")
   print(f"Total connections: {social_net.edge_count()}")
   
   # Find most connected users
   for user_id in social_net.nodes:
       degree = social_net.degree(user_id)
       user_data = social_net.get_node(user_id)
       print(f"{user_data['name']} has {degree} connections")
   
   # Find users with common interests
   tech_users = []
   for user_id in social_net.nodes:
       user_data = social_net.get_node(user_id)
       if "tech" in user_data.get("interests", []):
           tech_users.append(user_data["name"])
   
   print(f"Tech enthusiasts: {', '.join(tech_users)}")

Knowledge Graph Example
-----------------------

Building a simple knowledge graph:

.. code-block:: python

   from groggy import Graph
   
   # Create knowledge graph
   kg = Graph()
   
   # Add entities with types
   entities = [
       ("python", {"type": "programming_language", "created": 1991, "creator": "Guido van Rossum"}),
       ("guido", {"type": "person", "name": "Guido van Rossum", "nationality": "Dutch"}),
       ("rust", {"type": "programming_language", "created": 2010, "creator": "Mozilla"}),
       ("mozilla", {"type": "organization", "founded": 1998, "type": "non-profit"}),
       ("web_dev", {"type": "domain", "name": "Web Development"}),
       ("systems_prog", {"type": "domain", "name": "Systems Programming"}),
   ]
   
   # Add entities
   for entity_id, attributes in entities:
       kg.add_node(entity_id, **attributes)
   
   # Add relationships
   relationships = [
       ("guido", "python", {"relationship": "created", "year": 1991}),
       ("mozilla", "rust", {"relationship": "sponsors", "since": 2010}),
       ("python", "web_dev", {"relationship": "used_for", "popularity": "high"}),
       ("rust", "systems_prog", {"relationship": "used_for", "popularity": "growing"}),
       ("python", "rust", {"relationship": "alternative_to", "context": "some_domains"}),
   ]
   
   for source, target, metadata in relationships:
       kg.add_edge(source, target, **metadata)
   
   # Query the knowledge graph
   print("Knowledge Graph Queries:")
   
   # What did Guido create?
   guido_creations = kg.get_neighbors("guido")
   for creation in guido_creations:
       creation_data = kg.get_node(creation)
       print(f"Guido created: {creation} ({creation_data['type']})")
   
   # What languages are used for web development?
   web_dev_neighbors = kg.get_neighbors("web_dev")
   for lang in web_dev_neighbors:
       lang_data = kg.get_node(lang)
       if lang_data["type"] == "programming_language":
           edge_data = kg.get_edge(lang, "web_dev")
           print(f"{lang} is used for web dev (popularity: {edge_data['popularity']})")

Working with Complex Attributes
-------------------------------

Handling nested and complex data structures:

.. code-block:: python

   from groggy import Graph
   import json
   from datetime import datetime
   
   # Create graph for complex data
   complex_graph = Graph()
   
   # Add node with deeply nested attributes
   company_id = complex_graph.add_node(
       name="TechCorp Inc.",
       founded=2015,
       headquarters={
           "address": {
               "street": "123 Tech Street",
               "city": "San Francisco",
               "state": "CA",
               "zip": "94105"
           },
           "coordinates": {"lat": 37.7749, "lng": -122.4194}
       },
       employees=[
           {"id": 1, "name": "Alice", "role": "CTO", "salary": 200000},
           {"id": 2, "name": "Bob", "role": "Engineer", "salary": 150000},
           {"id": 3, "name": "Carol", "role": "Designer", "salary": 120000}
       ],
       financial_data={
           "revenue": [1000000, 2500000, 5000000],  # Last 3 years
           "funding_rounds": [
               {"round": "Seed", "amount": 500000, "date": "2016-01-15"},
               {"round": "Series A", "amount": 5000000, "date": "2018-06-20"},
               {"round": "Series B", "amount": 15000000, "date": "2020-09-10"}
           ]
       },
       metadata={
           "last_updated": datetime.now().isoformat(),
           "data_source": "company_database",
           "confidence": 0.95
       }
   )
   
   # Retrieve and work with complex data
   company_data = complex_graph.get_node(company_id)
   
   # Access nested data
   hq_city = company_data["headquarters"]["address"]["city"]
   print(f"Company headquarters: {hq_city}")
   
   # Calculate average salary
   employees = company_data["employees"]
   avg_salary = sum(emp["salary"] for emp in employees) / len(employees)
   print(f"Average salary: ${avg_salary:,.2f}")
   
   # Latest funding round
   funding_rounds = company_data["financial_data"]["funding_rounds"]
   latest_funding = max(funding_rounds, key=lambda x: x["date"])
   print(f"Latest funding: {latest_funding['round']} - ${latest_funding['amount']:,}")
   
   # Update complex attributes
   # Add new employee
   updated_employees = company_data["employees"].copy()
   updated_employees.append({
       "id": 4, 
       "name": "Dave", 
       "role": "Marketing", 
       "salary": 110000
   })
   
   complex_graph.update_node(company_id, employees=updated_employees)
   
   # Verify update
   updated_company = complex_graph.get_node(company_id)
   print(f"Employee count after update: {len(updated_company['employees'])}")

Error Handling
--------------

Proper error handling in Groggy operations:

.. code-block:: python

   from groggy import Graph
   
   g = Graph()
   
   # Safe node operations
   try:
       # This will work
       alice = g.add_node(name="Alice")
       print(f"Added node: {alice}")
       
       # This will raise KeyError
       nonexistent = g.get_node("does_not_exist")
   except KeyError as e:
       print(f"Node not found: {e}")
   
   # Safe edge operations
   try:
       bob = g.add_node(name="Bob")
       
       # This will work
       edge_id = g.add_edge(alice, bob, relationship="friends")
       print(f"Added edge: {edge_id}")
       
       # This will raise ValueError (edge already exists)
       duplicate_edge = g.add_edge(alice, bob, relationship="colleagues")
   except ValueError as e:
       print(f"Edge creation failed: {e}")
   
   # Safe attribute access
   alice_data = g.get_node(alice)
   
   # Safe way to access potentially missing attributes
   age = alice_data.get("age", "unknown")
   city = alice_data.get("city", "not specified")
   
   print(f"Alice's age: {age}, city: {city}")
   
   # Check existence before operations
   if g.has_node("charlie"):
       charlie_data = g.get_node("charlie")
   else:
       print("Charlie node does not exist")
   
   if g.has_edge(alice, bob):
       edge_data = g.get_edge(alice, bob)
       print(f"Alice-Bob relationship: {edge_data.get('relationship', 'unknown')}")

Graph Iteration Patterns
-------------------------

Efficient ways to iterate through graphs:

.. code-block:: python

   from groggy import Graph
   
   # Create sample graph
   g = Graph()
   people = ["Alice", "Bob", "Charlie", "Diana", "Eve"]
   
   # Add nodes with attributes
   node_ids = {}
   for person in people:
       node_ids[person] = g.add_node(
           name=person, 
           age=20 + len(person),  # Simple age assignment
           active=True
       )
   
   # Add some edges
   connections = [
       ("Alice", "Bob"), ("Bob", "Charlie"), 
       ("Charlie", "Diana"), ("Diana", "Eve"),
       ("Alice", "Charlie"), ("Bob", "Diana")
   ]
   
   for person1, person2 in connections:
       g.add_edge(node_ids[person1], node_ids[person2], 
                 weight=1.0, created="2025-01-01")
   
   print("=== Node Iteration ===")
   # Iterate over all nodes
   for node_id in g.nodes:
       node_data = g.get_node(node_id)
       print(f"{node_data['name']} (age {node_data['age']})")
   
   print("\\n=== Edge Iteration ===")
   # Iterate over all edges
   for source, target in g.edge_pairs():
       edge_data = g.get_edge(source, target)
       source_name = g.get_node(source)['name']
       target_name = g.get_node(target)['name']
       print(f"{source_name} -> {target_name} (weight: {edge_data['weight']})")
   
   print("\\n=== Neighbor Analysis ===")
   # Analyze each node's connections
   for node_id in g.nodes:
       node_data = g.get_node(node_id)
       neighbors = g.get_neighbors(node_id)
       neighbor_names = [g.get_node(n)['name'] for n in neighbors]
       print(f"{node_data['name']} is connected to: {', '.join(neighbor_names)}")
       
   print("\\n=== Filtered Iteration ===")
   # Iterate with filtering
   young_people = []
   for node_id in g.nodes:
       node_data = g.get_node(node_id)
       if node_data['age'] < 25:
           young_people.append(node_data['name'])
   
   print(f"Young people (age < 25): {', '.join(young_people)}")
   
   # Find high-degree nodes
   high_degree_nodes = []
   for node_id in g.nodes:
       if g.degree(node_id) >= 2:
           node_data = g.get_node(node_id)
           high_degree_nodes.append(node_data['name'])
   
   print(f"Well-connected people (degree >= 2): {', '.join(high_degree_nodes)}")
