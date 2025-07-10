Groggy - Graph Language Interface Documentation
===============================================

.. image:: https://img.shields.io/badge/license-MIT-blue.svg
   :target: https://opensource.org/licenses/MIT
   :alt: License: MIT

.. image:: https://img.shields.io/badge/python-3.8+-blue.svg
   :target: https://www.python.org/downloads/
   :alt: Python 3.8+

.. image:: https://img.shields.io/badge/rust-1.70+-orange.svg
   :target: https://www.rust-lang.org/
   :alt: Rust 1.70+

**A high-performance graph library with unified Rust backend for efficient graph operations and optimized filtering at scale.**

Groggy is a powerful graph manipulation library designed for both rapid prototyping and production-scale applications. It features a unified Rust-based columnar storage system with bitmap indexing for maximum performance and a Python interface for ease of use.

Features
--------

* **High-Performance Unified Backend**: Columnar storage with bitmap indexing for O(1) exact match filtering
* **Competitive Performance**: 1.2-5.6x faster than NetworkX for common filtering operations
* **Optimized Filtering**: Fast bitmap-based exact matching and range queries with smart query detection
* **Scalable Architecture**: Unified type system (NodeData, EdgeData, GraphType) handles large graphs efficiently  
* **Batch Operations**: Efficient bulk operations for nodes and edges with significant performance improvements
* **Memory Efficient**: Optimized data structures with sparse storage and bitmap indices
* **State Management**: Save, restore, and track graph states over time
* **Intuitive API**: Clean, Pythonic interface with automatic optimization path selection
* **Comprehensive Testing**: Full test suite with performance benchmarks and stress testing

Quick Start
-----------

.. code-block:: python

   import groggy as gr

   # Create a graph (uses optimized Rust backend)
   g = gr.Graph()

   # Add nodes with attributes
   g.add_node("alice", age=30, role="engineer")
   g.add_node("bob", age=25, role="designer") 
   g.add_node("charlie", age=35, role="manager")

   # Add edges with attributes
   g.add_edge("alice", "bob", relationship="collaborates")
   g.add_edge("charlie", "alice", relationship="manages")

   # Efficient batch operations for large graphs
   nodes_data = [
       {'id': 'employee_001', 'name': 'Charlie', 'role': 'manager', 'salary': 75000},
       {'id': 'employee_002', 'name': 'Diana', 'role': 'engineer', 'salary': 68000}
   ]
   g.add_nodes(nodes_data)  # Add thousands of nodes efficiently

   edges_data = [
       {'source': 'alice', 'target': 'employee_001', 'relationship': 'reports_to'},
       {'source': 'bob', 'target': 'employee_002', 'relationship': 'collaborates'}
   ]
   g.add_edges(edges_data)  # Add thousands of edges efficiently

   # Fast filtering with automatic optimization
   engineers = g.filter_nodes(role='engineer')        # O(1) bitmap lookup
   high_earners = g.filter_nodes('salary > 70000')    # Optimized range query
   managers = g.filter_nodes({'role': 'manager'})     # Dictionary filter

   # Query the graph
   print(f"Nodes: {len(g.nodes)}, Edges: {len(g.edges)}")
   neighbors = g.get_neighbors("alice")
   
   # State management
   g.save_state("initial")
   g.update_node("alice", {"promoted": True, "salary": 80000})
   g.save_state("after_promotion")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
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

   api/groggy
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
