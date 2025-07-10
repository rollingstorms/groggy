Installation
============

Groggy requires building from source due to its Rust-based columnar backend. The installation process is straightforward and includes both Python and Rust components.

Requirements
------------

**System Requirements:**
- Python 3.8 or higher
- Rust 1.70 or higher (automatically handled by maturin)
- Git for cloning the repository

**Development Requirements:**
- Maturin for building Python-Rust extensions
- pytest for running tests

From Source
-----------

Clone the repository and build with maturin:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/rollingstorms/groggy.git
   cd groggy

   # Install maturin for Python-Rust bindings
   pip install maturin

   # Build and install with optimized Rust backend
   maturin develop --release

   # Verify installation
   python -c "import groggy; print('Groggy installed successfully!')"

Development Installation
------------------------

For development work with additional testing and documentation tools:

.. code-block:: bash

   # Install development dependencies
   pip install pytest pytest-benchmark sphinx

   # Build in development mode (faster builds, less optimization)
   maturin develop

   # Run tests to verify installation
   python -m pytest tests/ -v

Testing Installation
-------------------

Verify your installation works correctly:

.. code-block:: python

   import groggy as gr
   print(f"Groggy version: {gr.__version__}")
   
   # Test basic functionality
   g = gr.Graph()
   node_id = g.add_node(name="test", value=42)
   print(f"Created node: {node_id}")
   
   # Test filtering performance
   nodes = [{'id': f'node_{i}', 'value': i} for i in range(1000)]
   g.add_nodes(nodes)
   filtered = g.filter_nodes(value=500)
   print(f"Filtering works: {len(filtered)} nodes found")

Troubleshooting
---------------

**ImportError: No module named groggy._core**
   The Rust backend failed to compile. Check that maturin installed correctly:
   
   .. code-block:: bash
   
      pip install maturin
      maturin develop --release

**Rust compilation errors**
   Ensure you have a compatible Rust version:
   
   .. code-block:: bash
   
      # Install/update Rust
      curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
      source ~/.cargo/env
      rustup update
      
      # Clean and rebuild
      cargo clean
      maturin develop --release

**Performance issues**
   Make sure you built with the ``--release`` flag for optimal performance:
   
   .. code-block:: bash
   
      maturin develop --release
   
      brew install rust
