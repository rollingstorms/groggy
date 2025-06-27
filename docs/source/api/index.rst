API Reference
=============

This section contains the complete API reference for GLI.

Core Classes
------------

.. toctree::
   :maxdepth: 2

   gli
   graph

Data Structures & Views
-----------------------

.. toctree::
   :maxdepth: 2

   data_structures
   views

Graph Operations
----------------

.. toctree::
   :maxdepth: 2

   batch
   state

Utilities
---------

.. toctree::
   :maxdepth: 2

   utils

Backend Architecture
--------------------

GLI uses a high-performance Rust backend with Python bindings for optimal performance:

* **Rust Backend**: High-performance core implementation in Rust
* **Python Interface**: Pythonic API with lazy loading and smart caching
* **Automatic Conversion**: Seamless handling of mixed int/str node IDs
* **State Management**: Git-like branching and state management
* **Batch Operations**: Optimized bulk operations for large-scale graphs

Backend Functions
~~~~~~~~~~~~~~~~~

.. currentmodule:: gli

.. autosummary::
   :toctree: generated/

   get_available_backends
   set_backend
   get_current_backend

Core Constants
~~~~~~~~~~~~~~

.. autodata:: RUST_BACKEND_AVAILABLE
   :annotation: = True/False
   
   Boolean indicating whether the Rust backend is available in this installation.
