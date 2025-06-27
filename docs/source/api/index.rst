API Reference
=============

This section contains the complete API reference for GLI.

Core Classes
------------

.. toctree::
   :maxdepth: 2

   gli
   graph
   store

Data Structures
---------------

.. toctree::
   :maxdepth: 2

   data_structures
   views

Utilities
---------

.. toctree::
   :maxdepth: 2

   utils
   state
   delta

Backend Management
------------------

GLI supports multiple backends for different performance characteristics:

* **Python Backend**: Pure Python implementation for maximum compatibility
* **Rust Backend**: High-performance Rust implementation with Python bindings

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
