API Reference
=============

This section contains the complete API reference for Groggy.

Core Classes
------------

.. toctree::
   :maxdepth: 2

   groggy
   graph

Data Structures & Views
-----------------------

.. toctree::
   :maxdepth: 2

   data_structures
   views

Utilities
---------

.. toctree::
   :maxdepth: 2

   utils

Backend Architecture
--------------------

Groggy uses a unified high-performance Rust backend with Python bindings:

* **Rust Backend**: High-performance columnar storage implementation in Rust
* **Python Interface**: Pythonic API with optimized filtering pipeline
* **Columnar Storage**: Efficient attribute storage with bitmap indexing
* **Smart Query Optimization**: Automatic selection of optimal filtering strategy
* **Batch Operations**: Optimized bulk operations for large-scale graphs
* **Thread Safety**: Multi-threaded operations with proper synchronization

Backend Functions
~~~~~~~~~~~~~~~~~

.. currentmodule:: groggy

.. autosummary::
   :toctree: generated/

   get_available_backends
   set_backend
   get_current_backend
   set_backend
   get_current_backend

Core Constants
~~~~~~~~~~~~~~

.. autodata:: RUST_BACKEND_AVAILABLE
   :annotation: = True/False
   
   Boolean indicating whether the Rust backend is available in this installation.
