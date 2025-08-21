Groggy Documentation
====================

.. image:: ../img/groggy.svg
   :width: 400px
   :align: center

**High-performance graph analytics with unified storage views**

Groggy is a next-generation graph processing library that combines high-performance Rust core with an intuitive Python API. It provides seamless integration between graph topology and advanced tabular analytics through unified storage views (Arrays, Matrices, Tables) that support both relational operations and graph-aware analysis.

🚀 **Latest Release: v0.3.0** - Storage View Unification Complete

Quick Start
-----------

.. code-block:: python

   import groggy as gr

   # Create a graph
   g = gr.Graph()

   # Add nodes and edges
   g.add_node("alice", age=30, role="engineer")
   g.add_node("bob", age=25, role="designer")
   g.add_edge("alice", "bob", weight=0.8)

   # Convert to table for analysis
   nodes_table = g.nodes.table()
   print(nodes_table.describe())

   # Advanced analytics
   communities = g.communities.louvain()
   centrality = g.centrality.betweenness()

Key Features
------------

🏗️ **Unified Storage Architecture**
   - **GraphArray**: High-performance columnar arrays with statistical operations
   - **GraphMatrix**: Homogeneous matrix operations with linear algebra support  
   - **GraphTable**: Pandas-like tabular operations with graph integration
   - **Lazy Evaluation**: Memory-efficient views with on-demand computation

📊 **Advanced Analytics**
   - **Multi-Table Operations**: JOIN (inner, left, right, outer), UNION, INTERSECT
   - **GROUP BY & Aggregation**: Complete statistical functions (sum, count, mean, min, max, std, var, first, last, unique)
   - **Graph-Aware Operations**: Neighborhood analysis, k-hop traversal, connectivity filtering
   - **Statistical Computing**: Comprehensive descriptive statistics with caching

⚡ **High Performance**
   - **Rust Core**: Memory-efficient columnar storage with attribute pools
   - **Batch Operations**: Vectorized operations for large-scale processing
   - **Smart Caching**: Intelligent cache invalidation for statistical computations
   - **Zero-Copy Views**: Efficient data access without unnecessary copying

🐍 **Python Integration**
   - **Pandas Compatible**: Familiar API with .head(), .tail(), .describe(), .group_by()
   - **Rich Display**: Beautiful HTML tables and formatted output in Jupyter
   - **Advanced Indexing**: Support for slicing, boolean masks, fancy indexing
   - **Graph Builders**: Unified gr.array(), gr.table(), gr.matrix() constructors

Table of Contents
-----------------

.. toctree::
   :maxdepth: 2
   :caption: Getting Started

   installation
   quickstart
   tutorials/index

.. toctree::
   :maxdepth: 2
   :caption: User Guide

   user-guide/graph-basics
   user-guide/storage-views
   user-guide/analytics
   user-guide/performance
   user-guide/integration

.. toctree::
   :maxdepth: 2
   :caption: API Reference

   api/graph
   api/array
   api/matrix
   api/table
   api/analytics
   api/utilities

.. toctree::
   :maxdepth: 2
   :caption: Architecture

   architecture/rust-core
   architecture/ffi-interface
   architecture/python-api
   architecture/memory-management

.. toctree::
   :maxdepth: 2
   :caption: Advanced Topics

   advanced/custom-algorithms
   advanced/performance-optimization
   advanced/extending-groggy
   advanced/visualization  

.. toctree::
   :maxdepth: 2
   :caption: Development

   development/contributing
   development/building
   development/testing
   development/release-notes

.. toctree::
   :maxdepth: 1
   :caption: Examples

   examples/basic-operations
   examples/data-analysis
   examples/graph-algorithms
   examples/integration-patterns

Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`