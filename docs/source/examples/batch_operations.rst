Batch Operations Examples
=========================

This guide demonstrates GLI's high-performance batch operations for efficient large-scale graph processing.

Overview
--------

GLI's batch operations provide significant performance improvements over individual operations:

- **30-40x faster** filtering and querying
- **Memory efficient** bulk processing
- **Identical API** across Python and Rust backends
- **Linear scaling** to millions of nodes

Social Network Analysis
-----------------------

Building and Analyzing Large Social Networks
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import gli
   import random
   import time
   
   # Use Rust backend for maximum performance
   gli.set_backend('rust')
   g = gli.Graph()
   
   print("Creating social network...")
   start_time = time.time()
   
   # Sample demographic data
   cities = ['New York', 'Los Angeles', 'Chicago', 'Houston', 'Phoenix']
   occupations = ['Engineer', 'Teacher', 'Doctor', 'Artist', 'Manager']
   names = ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve', 'Frank']
   
   # Create 10,000 people efficiently
   node_ids = []
   for i in range(10000):
       person_id = f"person_{i}"
       g.add_node(person_id,
                  name=f"{random.choice(names)}_{i}",
                  age=random.randint(18, 65),
                  city=random.choice(cities),
                  occupation=random.choice(occupations),
                  active=random.choice([True, False]))
       node_ids.append(person_id)
   
   # Add friendships
   for _ in range(5000):
       person1 = random.choice(node_ids)
       person2 = random.choice(node_ids)
       if person1 != person2:
           g.add_edge(person1, person2,
                      relationship='friend',
                      strength=random.uniform(0.3, 1.0))
   
   creation_time = time.time() - start_time
   print(f"Created {g.node_count():,} people in {creation_time:.2f} seconds")

Efficient Filtering and Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Batch filtering - much faster than individual queries
   print("\\nPerforming batch analysis...")
   
   # Find people by city
   ny_people = g.batch_filter_nodes(city='New York')
   la_people = g.batch_filter_nodes(city='Los Angeles')
   
   # Find people by occupation
   engineers = g.batch_filter_nodes(occupation='Engineer')
   teachers = g.batch_filter_nodes(occupation='Teacher')
   
   # Find active users
   active_users = g.batch_filter_nodes(active=True)
   
   print(f"New York residents: {len(ny_people):,}")
   print(f"Los Angeles residents: {len(la_people):,}")
   print(f"Engineers: {len(engineers):,}")
   print(f"Teachers: {len(teachers):,}")
   print(f"Active users: {len(active_users):,}")

Combined Filtering for Complex Queries
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Complex analysis: young professionals in major cities
   young_professionals = []
   
   # Get all engineers first
   all_engineers = g.batch_filter_nodes(occupation='Engineer')
   
   # Get their attributes efficiently
   engineer_attrs = g.batch_get_node_attributes(all_engineers)
   
   # Filter by age and city
   target_cities = {'New York', 'Los Angeles', 'Chicago'}
   for i, attrs in enumerate(engineer_attrs):
       if (25 <= attrs['age'] <= 35 and 
           attrs['city'] in target_cities):
           young_professionals.append(all_engineers[i])
   
   print(f"Young engineers in major cities: {len(young_professionals)}")
   
   # Show examples
   sample_attrs = g.batch_get_node_attributes(young_professionals[:5])
   for i, attrs in enumerate(sample_attrs):
       print(f"  {attrs['name']}: age {attrs['age']}, {attrs['city']}")

Supply Chain Network Analysis
-----------------------------

Enterprise-Scale Graph Processing
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create supply chain network
   supply_chain = gli.Graph(backend='rust')
   
   # Company types and attributes
   company_types = ['Supplier', 'Manufacturer', 'Distributor', 'Retailer']
   industries = ['Electronics', 'Automotive', 'Textiles', 'Food', 'Pharma']
   regions = ['North America', 'Europe', 'Asia', 'South America']
   
   # Create companies
   companies = []
   for i in range(5000):
       company_id = f"company_{i}"
       supply_chain.add_node(company_id,
                             name=f"Company_{i}",
                             type=random.choice(company_types),
                             industry=random.choice(industries),
                             region=random.choice(regions),
                             size=random.choice(['Small', 'Medium', 'Large']),
                             sustainability_score=random.uniform(1.0, 10.0),
                             financial_rating=random.choice(['A', 'B', 'C', 'D']))
       companies.append(company_id)
   
   # Create supply relationships
   for _ in range(8000):
       supplier = random.choice(companies)
       customer = random.choice(companies)
       if supplier != customer:
           supply_chain.add_edge(supplier, customer,
                                relationship='supplies',
                                volume=random.randint(1000, 100000),
                                contract_value=random.randint(10000, 5000000),
                                risk_level=random.choice(['Low', 'Medium', 'High']))

Supply Chain Risk Analysis
~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Risk analysis using batch operations
   print("\\nSupply Chain Risk Analysis:")
   
   # Find high-risk suppliers
   high_risk_edges = supply_chain.batch_filter_edges(risk_level='High')
   print(f"High-risk supply relationships: {len(high_risk_edges)}")
   
   # Find companies by sustainability score
   sustainable_companies = []
   all_companies = list(supply_chain.nodes)
   company_attrs = supply_chain.batch_get_node_attributes(all_companies)
   
   for i, attrs in enumerate(company_attrs):
       if attrs['sustainability_score'] >= 8.0:
           sustainable_companies.append(all_companies[i])
   
   print(f"Highly sustainable companies: {len(sustainable_companies)}")
   
   # Regional analysis
   for region in regions:
       regional_companies = supply_chain.batch_filter_nodes(region=region)
       regional_attrs = supply_chain.batch_get_node_attributes(regional_companies)
       
       avg_sustainability = sum(a['sustainability_score'] for a in regional_attrs) / len(regional_attrs)
       large_companies = sum(1 for a in regional_attrs if a['size'] == 'Large')
       
       print(f"{region}: {len(regional_companies)} companies, "
             f"avg sustainability: {avg_sustainability:.2f}, "
             f"{large_companies} large companies")

Bulk Attribute Updates
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Quarterly updates - efficient bulk operations
   quarterly_updates = {}
   
   # Update all companies in electronics industry
   electronics_companies = supply_chain.batch_filter_nodes(industry='Electronics')
   for company_id in electronics_companies:
       quarterly_updates[company_id] = {
           'last_audit': '2024-Q1',
           'market_trend': 'growing'
       }
   
   # Apply all updates at once
   supply_chain.batch_set_node_attributes(quarterly_updates)
   
   # Functional updates based on current attributes
   large_company_updates = {}
   large_companies = supply_chain.batch_filter_nodes(size='Large')
   
   for company_id in large_companies:
       large_company_updates[company_id] = lambda attrs: {
           **attrs,
           'market_cap': attrs.get('market_cap', 1000000) * 1.1,
           'employees': attrs.get('employees', 100) + random.randint(10, 50)
       }
   
   supply_chain.batch_update_node_attributes(large_company_updates)

Knowledge Graph Processing
-------------------------

Academic Collaboration Network
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Create academic collaboration network
   academic_graph = gli.Graph(backend='rust')
   
   # Academic data
   fields = ['Computer Science', 'Biology', 'Physics', 'Chemistry', 'Mathematics']
   institutions = ['MIT', 'Stanford', 'Harvard', 'Caltech', 'Berkeley']
   positions = ['Professor', 'Associate Professor', 'Assistant Professor', 'Postdoc']
   
   # Create researchers
   researchers = []
   for i in range(3000):
       researcher_id = f"researcher_{i}"
       academic_graph.add_node(researcher_id,
                               name=f"Dr. {random.choice(['Smith', 'Johnson', 'Williams', 'Brown'])}_{i}",
                               field=random.choice(fields),
                               institution=random.choice(institutions),
                               position=random.choice(positions),
                               h_index=random.randint(5, 100),
                               publications=random.randint(10, 200),
                               years_active=random.randint(5, 30))
       researchers.append(researcher_id)
   
   # Create collaborations
   for _ in range(5000):
       researcher1 = random.choice(researchers)
       researcher2 = random.choice(researchers)
       if researcher1 != researcher2:
           academic_graph.add_edge(researcher1, researcher2,
                                   relationship='collaboration',
                                   papers_coauthored=random.randint(1, 20),
                                   collaboration_strength=random.uniform(0.1, 1.0),
                                   active=random.choice([True, False]))

Research Impact Analysis
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   # Analyze research network using batch operations
   print("\\nAcademic Network Analysis:")
   
   # Find top researchers by field
   for field in fields:
       field_researchers = academic_graph.batch_filter_nodes(field=field)
       field_attrs = academic_graph.batch_get_node_attributes(field_researchers)
       
       # Sort by h-index
       top_researchers = sorted(
           zip(field_researchers, field_attrs),
           key=lambda x: x[1]['h_index'],
           reverse=True
       )[:5]
       
       print(f"\\nTop {field} researchers:")
       for researcher_id, attrs in top_researchers:
           print(f"  {attrs['name']}: h-index {attrs['h_index']}, "
                 f"{attrs['publications']} publications")
   
   # Cross-institutional collaborations
   active_collabs = academic_graph.batch_filter_edges(active=True)
   cross_institutional = []
   
   for source, target in active_collabs:
       source_attrs = academic_graph.get_node(source)
       target_attrs = academic_graph.get_node(target)
       if source_attrs['institution'] != target_attrs['institution']:
           cross_institutional.append((source, target))
   
   print(f"\\nCross-institutional collaborations: {len(cross_institutional)}")

Performance Monitoring
---------------------

Benchmarking Batch Operations
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

   import time
   
   def benchmark_operations(graph, description):
       \"\"\"Benchmark various batch operations\"\"\"
       print(f"\\n{description}")
       print("-" * len(description))
       
       # Benchmark filtering
       start = time.time()
       filtered_nodes = graph.batch_filter_nodes(active=True)
       filter_time = time.time() - start
       print(f"Batch filter: {len(filtered_nodes)} nodes in {filter_time:.4f}s")
       
       # Benchmark attribute retrieval
       start = time.time()
       attrs = graph.batch_get_node_attributes(filtered_nodes[:1000])
       attr_time = time.time() - start
       print(f"Batch attributes: 1000 nodes in {attr_time:.4f}s")
       
       # Benchmark updates
       updates = {node_id: {'last_accessed': '2024-01-15'} 
                  for node_id in filtered_nodes[:500]}
       start = time.time()
       graph.batch_set_node_attributes(updates)
       update_time = time.time() - start
       print(f"Batch updates: 500 nodes in {update_time:.4f}s")
   
   # Test with different backends
   gli.set_backend('rust')
   rust_graph = gli.Graph()
   # ... populate graph ...
   benchmark_operations(rust_graph, "Rust Backend Performance")
   
   gli.set_backend('python')
   python_graph = gli.Graph()
   # ... populate graph ...
   benchmark_operations(python_graph, "Python Backend Performance")

Best Practices Summary
---------------------

Performance Optimization Guidelines
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

1. **Use Batch Operations**: Always prefer batch methods for multiple operations

   .. code-block:: python
   
      # ✅ Good
      filtered = g.batch_filter_nodes(category='important')
      attributes = g.batch_get_node_attributes(filtered)
      
      # ❌ Avoid
      filtered = [n for n in g.nodes if g.get_node(n)['category'] == 'important']

2. **Choose the Right Backend**: Use Rust for large graphs, Python for development

   .. code-block:: python
   
      # Production/large graphs
      gli.set_backend('rust')
      
      # Development/debugging
      gli.set_backend('python')

3. **Batch Attribute Operations**: Update multiple nodes efficiently

   .. code-block:: python
   
      # ✅ Efficient bulk updates
      g.batch_set_node_attributes({
          'node1': {'status': 'active'},
          'node2': {'status': 'inactive'}
      })

4. **Memory-Efficient Processing**: Use subgraphs for focused analysis

   .. code-block:: python
   
      # Process subsets efficiently
      important_subgraph = g.create_subgraph_fast(priority='high')
      analysis_results = analyze_network(important_subgraph)

Expected Performance Gains
~~~~~~~~~~~~~~~~~~~~~~~~~~

With proper use of batch operations, expect:

- **30-40x faster** filtering operations
- **Linear scaling** to millions of nodes  
- **50-70% less memory** usage with Rust backend
- **Sub-millisecond** query times for common operations

These optimizations make GLI suitable for real-time applications and large-scale graph analysis.
