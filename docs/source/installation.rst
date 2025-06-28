Installation
============

Groggy can be installed from source with both Python and Rust backend support.

Requirements
------------

**Python Requirements:**
- Python 3.8 or higher
- pip

**Rust Requirements (for compilation):**
- Rust 1.70 or higher
- Cargo

From Source
-----------

Clone the repository and install:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/your-org/groggy.git
   cd groggy

   # Install Python dependencies
   pip install -e .

   # For development with all dependencies
   pip install -e ".[dev]"

Building with Rust Backend
---------------------------

The Rust backend provides significant performance improvements. To build with Rust support:

.. code-block:: bash

   # Ensure Rust is installed
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env

   # Install maturin for Python-Rust bindings
   pip install maturin

   # Build and install with Rust backend
   maturin develop --release

Development Installation
------------------------

For development work:

.. code-block:: bash

   # Install in development mode with all dependencies
   pip install -e ".[dev]"

   # Install pre-commit hooks (if available)
   pre-commit install

Docker Installation
-------------------

You can also run GLI in a Docker container:

.. code-block:: bash

   # Build the Docker image
   docker build -t gli .

   # Run interactive Python session
   docker run -it gli python

Verifying Installation
----------------------

Test your installation:

.. code-block:: python

   import groggy as gr
   print(f"Groggy version: {gr.__version__}")
   print(f"Available backends: {gr.get_available_backends()}")
   
   # Test basic functionality
   g = gr.Graph()
   node_id = g.add_node(name="test")
   print(f"Created node: {node_id}")

Troubleshooting
---------------

**ImportError: No module named '_core'**
   The Rust backend is not compiled. Use ``maturin develop`` to build it, or the library will fall back to Python backend.

**Rust compilation errors**
   Ensure you have Rust 1.70+ installed and try updating:
   
   .. code-block:: bash
   
      rustup update
      cargo clean
      maturin develop --release

**Permission errors on macOS**
   You may need to install Rust via homebrew:
   
   .. code-block:: bash
   
      brew install rust
