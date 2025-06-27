GLI - Graph Language Interface Documentation
=============================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/rust-1.70+-orange.svg
   :target: https://www.rust-lang.org/
   :alt: Rust 1.70+

**A high-performance graph library with Rust backend for efficient graph operations, state management, and branching at scale.**

GLI (Graph Language Interface) is a powerful graph manipulation library designed for both rapid prototyping and production-scale applications. It features a high-performance Rust backend with a Python interface for maximum performance and usability.

Features
--------

* **High-Performance Rust Backend**: Handle 1M+ nodes/edges efficiently with ~85MB per million nodes
* **Comprehensive State Management**: Save, restore, and branch graph states with version control-like functionality
* **Batch Operations**: Optimized bulk operations for nodes and edges with 10-100x performance improvements
* **Flexible Node/Edge IDs**: Support both string and integer identifiers with automatic conversion
* **Rich Attributes**: Complex nested data structures on nodes and edges with efficient batch updates
* **Intuitive API**: Clean, Pythonic interface with lazy-loaded properties and smart caching
* **Memory Efficient**: Content-addressed storage with deduplication and garbage collection
* **Branch Management**: Create, switch, and manage multiple graph states like Git branches
* **Developer Friendly**: Comprehensive error handling, type hints, and extensive documentation

Quick Start
-----------

.. code-block:: python

   from gli import Graph

   # Create a graph (uses high-performance Rust backend)
   g = Graph()

   # Add nodes with attributes (supports int and str IDs)
   alice = g.add_node(label="Alice", age=30, city="New York")
   bob = g.add_node(1, label="Bob", age=25, city="Boston")  # Mixed ID types

   # Add edges with attributes
   friendship = g.add_edge(alice, bob, 
                          relationship="friends", 
                          since=2020, 
                          strength=0.9)

   # State management and branching
   initial_state = g.save_state("Initial graph")
   g.create_branch("development")
   g.switch_branch("development")

   # Efficient batch operations
   node_updates = {
       alice: {"department": "Engineering", "salary": 100000},
       bob: {"department": "Design", "salary": 85000}
   }
   g.set_nodes_attributes_batch(node_updates)

   # Query the graph with new API
   neighbors = g.get_neighbors(alice)
   edge = g.get_edge(alice, bob)  # Takes (source, target) parameters
   
   # Lazy-loaded properties
   print(f"Total states: {len(g.states['state_hashes'])}")
   print(f"Available branches: {list(g.branches.keys())}")

   # High-performance filtering
   engineers = g.filter_nodes(lambda node_id, attrs: attrs.get("department") == "Engineering")
   high_earners = g.filter_nodes({"salary": lambda x: x > 90000})

   # Access node/edge collections with lazy loading
   print(f"Nodes: {len(g.nodes)}, Edges: {len(g.edges)}")
   print(f"Alice is {g.nodes[alice]['age']} years old")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   examples/index
   api/index
   performance
   architecture
   testing
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Examples:

   examples/basic_usage
   examples/state_management
   examples/batch_operations
   examples/performance_optimization

.. toctree::
   :maxdepth: 1
   :caption: Python API Reference:

   api/gli
   api/graph
   api/data_structures
   api/views
   api/utils
   api/batch
   api/state

.. toctree::
   :maxdepth: 1
   :caption: Rust Backend:

   rust/overview
   rust/performance
   rust/ffi

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
