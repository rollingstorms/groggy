groggy module
=============

.. automodule:: groggy
   :members:
   :undoc-members:
   :show-inheritance:

Module Contents
---------------

The main Groggy module provides the primary interface for graph operations with a unified Rust-based backend.

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Graph

Functions
~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   create_random_graph

Constants
~~~~~~~~~

.. autodata:: __version__
   :annotation: = "0.2.0"

Core Graph Interface
-------------------

The unified Graph class provides high-performance operations through Rust backend:

.. autoclass:: Graph
   :members:
   :undoc-members:

.. autofunction:: set_backend

.. autofunction:: get_current_backend

.. autodata:: RUST_BACKEND_AVAILABLE
