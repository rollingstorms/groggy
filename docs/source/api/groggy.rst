groggy module
=============

.. automodule:: groggy
   :members:
   :undoc-members:
   :show-inheritance:

Module Contents
---------------

The main Groggy module provides the primary interface for graph operations.

Classes
~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   Graph
   GraphStore

Functions
~~~~~~~~~

.. autosummary::
   :toctree: generated/
   :nosignatures:

   create_random_graph
   get_available_backends
   set_backend
   get_current_backend

Constants
~~~~~~~~~

.. autodata:: __version__
   :annotation: = "0.2.0"

.. autodata:: RUST_BACKEND_AVAILABLE
   :annotation: = True/False

Backend Management
------------------

Groggy supports multiple computational backends:

.. autofunction:: get_available_backends

.. autofunction:: set_backend

.. autofunction:: get_current_backend

.. autodata:: RUST_BACKEND_AVAILABLE
