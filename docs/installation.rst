Installation
============

This guide covers how to install Groggy on your system.

Requirements
------------

- Python 3.8 or higher
- Rust 1.70 or higher (for building from source)
- Operating Systems: Linux, macOS, Windows

From Source (Recommended)
--------------------------

Currently, Groggy is available only from source. We plan to provide pre-built packages in the future.

**Step 1: Clone the repository**

.. code-block:: bash

   git clone https://github.com/rollingstorms/groggy.git
   cd groggy

**Step 2: Install build dependencies**

.. code-block:: bash

   pip install maturin

**Step 3: Build and install**

.. code-block:: bash

   # Development build (faster compilation)
   maturin develop

   # Release build (optimized performance)
   maturin develop --release

**Step 4: Verify installation**

.. code-block:: python

   import groggy as gr
   
   g = gr.Graph()
   g.add_node("test", value=42)
   print(f"Node count: {g.node_count()}")
   # Should output: Node count: 1

Development Installation
------------------------

For development work on Groggy itself:

.. code-block:: bash

   # Clone the repository
   git clone https://github.com/rollingstorms/groggy.git
   cd groggy

   # Install development dependencies
   pip install -r requirements-dev.txt

   # Install in development mode
   maturin develop

   # Run tests to verify everything works
   python -m pytest tests/ -v

Optional Dependencies
---------------------

For enhanced functionality, you may want to install:

.. code-block:: bash

   # For pandas integration
   pip install pandas

   # For NumPy integration  
   pip install numpy

   # For NetworkX compatibility
   pip install networkx

   # For visualization (future feature)
   pip install matplotlib plotly

   # For Jupyter notebook support
   pip install jupyter ipython

Docker Installation
-------------------

You can also run Groggy in a Docker container:

.. code-block:: bash

   # Build the Docker image
   docker build -t groggy .

   # Run with mounted volume
   docker run -it -v $(pwd):/workspace groggy

Troubleshooting
---------------

**Rust not found**

If you get an error about Rust not being found:

.. code-block:: bash

   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env

**Build fails on Windows**

On Windows, you may need to install Visual Studio Build Tools:

1. Download and install `Visual Studio Build Tools <https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022>`_
2. Select "C++ build tools" during installation
3. Restart your command prompt and try again

**ImportError after installation**

If you get an import error:

.. code-block:: bash

   # Make sure you're in the right Python environment
   which python
   python -c "import sys; print(sys.path)"

   # Reinstall if necessary
   maturin develop --release --force

**Performance issues**

For best performance, always use the release build:

.. code-block:: bash

   maturin develop --release

Updating
--------

To update to the latest version:

.. code-block:: bash

   cd groggy
   git pull origin main
   maturin develop --release

Uninstallation
--------------

To uninstall Groggy:

.. code-block:: bash

   pip uninstall groggy