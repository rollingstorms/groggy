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

**A high-performance graph library with dual Python/Rust backends for efficient graph operations at scale.**

GLI (Graph Language Interface) is a powerful graph manipulation library designed for both rapid prototyping and production-scale applications.

Features
--------

* **Dual Backend Architecture**: Switch seamlessly between Python and Rust backends
* **High Performance**: Rust backend handles 2M+ nodes with excellent memory efficiency  
* **Batch Operations**: 30-40x faster filtering and bulk operations for large graphs
* **Rich Attributes**: Complex nested data structures on nodes and edges
* **Memory Efficient**: Content-addressed storage with deduplication
* **Developer Friendly**: Intuitive API with comprehensive error handling

Quick Start
-----------

.. code-block:: python

   from gli import Graph, set_backend

   # Create a graph (auto-selects best available backend)
   g = Graph()

   # Add nodes with attributes
   alice = g.add_node(label="Alice", age=30, city="New York")
   bob = g.add_node(label="Bob", age=25, city="Boston")

   # Add edges with attributes
   friendship = g.add_edge(alice, bob, 
                          relationship="friends", 
                          since=2020, 
                          strength=0.9)

   # Query the graph
   neighbors = g.get_neighbors(alice)
   print(f"Alice has {len(neighbors)} connections")

   # Access attributes
   alice_node = g.get_node(alice)
   print(f"Alice is {alice_node['age']} years old")

   # High-performance batch operations
   engineers = g.batch_filter_nodes(occupation="Engineer")
   attributes = g.batch_get_node_attributes(engineers)
   print(f"Found {len(engineers)} engineers")

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   installation
   quickstart
   api/index
   examples/index
   performance
   architecture
   testing
   contributing
   changelog

.. toctree::
   :maxdepth: 1
   :caption: Python API Reference:

   api/gli
   api/graph
   api/store
   api/data_structures
   api/views
   api/utils
   api/state
   api/delta

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
