Building from Source
===================

This guide covers building Groggy from source code for development, customization, or platforms without pre-built packages.

Prerequisites
-------------

System Requirements
~~~~~~~~~~~~~~~~~~

**Minimum Requirements**:
- RAM: 4GB (8GB+ recommended for large graphs)
- CPU: x86_64 or ARM64 architecture
- Storage: 2GB free space for build artifacts

**Operating Systems**:
- Linux (Ubuntu 18.04+, CentOS 7+, or equivalent)
- macOS 10.15+ (Catalina or later)
- Windows 10+ with WSL2 or native Windows

Required Tools
~~~~~~~~~~~~~

**Rust Toolchain**:

.. code-block:: bash

   # Install Rust via rustup (recommended)
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env

   # Verify installation
   rustc --version
   cargo --version

   # Install required components
   rustup component add rustfmt clippy

**Python Environment**:

.. code-block:: bash

   # Python 3.8 or later required
   python3 --version

   # Install pip and virtual environment tools
   python3 -m pip install --upgrade pip setuptools wheel virtualenv

**Build Tools**:

Linux:

.. code-block:: bash

   # Ubuntu/Debian
   sudo apt update
   sudo apt install build-essential pkg-config libssl-dev

   # CentOS/RHEL/Fedora
   sudo yum groupinstall "Development Tools"
   sudo yum install pkg-config openssl-devel

   # Alpine
   sudo apk add build-base pkgconfig openssl-dev

macOS:

.. code-block:: bash

   # Install Xcode command line tools
   xcode-select --install

   # Or install via Homebrew
   brew install gcc pkg-config openssl

Windows:

.. code-block:: bash

   # Install Visual Studio Build Tools 2019 or later
   # Or Visual Studio Community with C++ development tools

   # Install Git for Windows
   # https://git-scm.com/download/win

Getting the Source Code
----------------------

Clone Repository
~~~~~~~~~~~~~~~

.. code-block:: bash

   # Clone the main repository
   git clone https://github.com/groggy-dev/groggy.git
   cd groggy

   # Check available branches
   git branch -a

   # Switch to development branch (optional)
   git checkout develop

Source Code Structure
~~~~~~~~~~~~~~~~~~~~

.. code-block:: text

   groggy/
   ├── src/                     # Rust source code
   │   ├── core/               # Core graph algorithms
   │   ├── ffi/                # Python FFI bindings
   │   └── lib.rs             # Main library entry point
   ├── python-groggy/          # Python package
   │   ├── src/               # Python source
   │   └── pyproject.toml     # Python package configuration
   ├── Cargo.toml              # Rust package configuration
   ├── Cargo.lock              # Dependency lock file
   ├── docs/                   # Documentation source
   ├── tests/                  # Test suites
   ├── benchmarks/             # Performance benchmarks
   └── examples/               # Usage examples

Setting Up Development Environment
---------------------------------

Python Virtual Environment
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Create virtual environment
   python3 -m venv venv

   # Activate virtual environment
   # Linux/macOS:
   source venv/bin/activate
   # Windows:
   venv\Scripts\activate

   # Upgrade pip
   pip install --upgrade pip

Install Dependencies
~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install Python build dependencies
   pip install maturin[patchelf] pytest

   # Install development dependencies (optional)
   pip install -r requirements-dev.txt

Building the Project
-------------------

Standard Build
~~~~~~~~~~~~~

.. code-block:: bash

   # Development build (faster, unoptimized)
   maturin develop

   # Release build (slower, optimized)
   maturin develop --release

   # Build with specific Python version
   maturin develop --python-interpreter python3.9

Maturin Build Options
~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Build wheel package
   maturin build

   # Build with debug symbols
   maturin develop --cargo-extra-args="--features debug"

   # Build with specific target
   maturin build --target x86_64-unknown-linux-gnu

   # Build with custom Rust flags
   RUSTFLAGS="-C target-cpu=native" maturin develop --release

Configuration Options
--------------------

Build Configurations
~~~~~~~~~~~~~~~~~~~

**Cargo Features**:

.. code-block:: toml

   # In Cargo.toml
   [features]
   default = ["parallel", "compression"]
   parallel = ["rayon", "crossbeam"]
   compression = ["zstd", "lz4"]
   simd = ["packed_simd"]
   python-extension = ["pyo3/extension-module"]
   debug = ["log", "env_logger"]

.. code-block:: bash

   # Build with specific features
   maturin develop --cargo-extra-args="--features simd,debug"

   # Build without default features
   maturin develop --cargo-extra-args="--no-default-features --features parallel"

Environment Variables
~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Rust compiler optimizations
   export RUSTFLAGS="-C target-cpu=native -C opt-level=3"

   # Link-time optimization
   export CARGO_PROFILE_RELEASE_LTO=true

   # Parallel compilation
   export CARGO_BUILD_JOBS=8

   # Custom target directory
   export CARGO_TARGET_DIR=/tmp/groggy-build

Platform-Specific Instructions
------------------------------

Linux
~~~~~

**Ubuntu/Debian**:

.. code-block:: bash

   # Install system dependencies
   sudo apt update
   sudo apt install python3-dev python3-venv build-essential \
                    pkg-config libssl-dev curl git

   # For SIMD support
   sudo apt install gcc-multilib

   # Build the project
   git clone https://github.com/groggy-dev/groggy.git
   cd groggy
   python3 -m venv venv
   source venv/bin/activate
   pip install maturin
   maturin develop --release

**CentOS/RHEL**:

.. code-block:: bash

   # Install system dependencies
   sudo yum groupinstall "Development Tools"
   sudo yum install python3-devel openssl-devel pkg-config curl git

   # Install Rust
   curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
   source ~/.cargo/env

   # Build the project
   git clone https://github.com/groggy-dev/groggy.git
   cd groggy
   python3 -m venv venv
   source venv/bin/activate
   pip install maturin
   maturin develop --release

macOS
~~~~~

**With Homebrew**:

.. code-block:: bash

   # Install dependencies
   brew install python rust pkg-config openssl git

   # Set environment variables for OpenSSL
   export PKG_CONFIG_PATH="/opt/homebrew/lib/pkgconfig:$PKG_CONFIG_PATH"
   export OPENSSL_DIR="/opt/homebrew/opt/openssl"

   # Build the project
   git clone https://github.com/groggy-dev/groggy.git
   cd groggy
   python3 -m venv venv
   source venv/bin/activate
   pip install maturin
   maturin develop --release

**Apple Silicon (M1/M2)**:

.. code-block:: bash

   # Additional considerations for Apple Silicon
   export ARCHFLAGS="-arch arm64"
   export RUSTFLAGS="-C target-cpu=apple-m1"

   # Build with native optimizations
   maturin develop --release

Windows
~~~~~~~

**With WSL2 (Recommended)**:

.. code-block:: bash

   # Install Ubuntu on WSL2
   wsl --install -d Ubuntu

   # Follow Linux instructions inside WSL2
   # Performance will be near-native

**Native Windows**:

.. code-block:: powershell

   # Install Visual Studio Build Tools
   # Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

   # Install Rust
   # Download from: https://rustup.rs/

   # Install Git and Python
   # Download from official websites

   # Open PowerShell as Administrator
   git clone https://github.com/groggy-dev/groggy.git
   cd groggy
   python -m venv venv
   venv\Scripts\activate
   pip install maturin
   maturin develop --release

Cross-Compilation
-----------------

Building for Different Targets
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Install cross-compilation target
   rustup target add aarch64-unknown-linux-gnu

   # Install cross-compiler
   sudo apt install gcc-aarch64-linux-gnu

   # Set environment variables
   export CC_aarch64_unknown_linux_gnu=aarch64-linux-gnu-gcc
   export CARGO_TARGET_AARCH64_UNKNOWN_LINUX_GNU_LINKER=aarch64-linux-gnu-gcc

   # Build for target
   maturin build --target aarch64-unknown-linux-gnu

Docker Cross-Compilation
~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: dockerfile

   # Dockerfile.cross-compile
   FROM quay.io/pypa/manylinux2014_x86_64

   RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
   ENV PATH="/root/.cargo/bin:${PATH}"

   RUN /opt/python/cp39-cp39/bin/pip install maturin

   WORKDIR /src
   COPY . .

   RUN /opt/python/cp39-cp39/bin/maturin build --release --strip

.. code-block:: bash

   # Build with Docker
   docker build -f Dockerfile.cross-compile -t groggy-builder .
   docker run --rm -v $(pwd)/target:/src/target groggy-builder

Optimization and Performance
---------------------------

Profile-Guided Optimization (PGO)
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Step 1: Build with instrumentation
   export RUSTFLAGS="-Cprofile-generate=/tmp/pgo-data"
   maturin build --release

   # Step 2: Run representative workload
   python -c "
   import groggy as gr
   g = gr.random_graph(10000, 0.001)
   g.centrality.pagerank()
   g.communities.louvain()
   "

   # Step 3: Merge profile data
   rustup run stable -- llvm-profdata merge -o /tmp/pgo-data/merged.profdata /tmp/pgo-data/*.profraw

   # Step 4: Build with optimization
   export RUSTFLAGS="-Cprofile-use=/tmp/pgo-data/merged.profdata"
   maturin build --release

Link-Time Optimization
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: toml

   # In Cargo.toml
   [profile.release]
   lto = true
   codegen-units = 1
   panic = "abort"

Custom Memory Allocator
~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: rust

   // In src/lib.rs
   #[cfg(not(target_env = "msvc"))]
   use jemallocator::Jemalloc;

   #[cfg(not(target_env = "msvc"))]
   #[global_allocator]
   static GLOBAL: Jemalloc = Jemalloc;

.. code-block:: toml

   # In Cargo.toml
   [dependencies]
   jemallocator = { version = "0.5", optional = true }

   [features]
   jemalloc = ["jemallocator"]

Testing the Build
----------------

Verification Tests
~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Import test
   python -c "import groggy; print(groggy.__version__)"

   # Basic functionality test
   python -c "
   import groggy as gr
   g = gr.Graph()
   g.add_node('test')
   print('Build successful!')
   "

   # Run test suite
   python -m pytest tests/ -v

Performance Verification
~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: bash

   # Run benchmarks
   python -m pytest benchmarks/ --benchmark-only

   # Profile memory usage
   python -c "
   import groggy as gr
   import tracemalloc
   
   tracemalloc.start()
   g = gr.random_graph(1000, 0.01)
   result = g.centrality.pagerank()
   current, peak = tracemalloc.get_traced_memory()
   print(f'Peak memory: {peak / 1024 / 1024:.1f} MB')
   "

Troubleshooting
--------------

Common Build Issues
~~~~~~~~~~~~~~~~~~

**Rust Compiler Errors**:

.. code-block:: bash

   # Update Rust toolchain
   rustup update

   # Clean build artifacts
   cargo clean

   # Check for missing dependencies
   cargo check

**Python Extension Issues**:

.. code-block:: bash

   # Reinstall maturin
   pip install --upgrade maturin

   # Clear Python cache
   python -c "import site; print(site.getsitepackages())"
   # Remove cached .so files

**OpenSSL Issues (macOS)**:

.. code-block:: bash

   # Set OpenSSL paths
   export OPENSSL_DIR=/opt/homebrew/opt/openssl
   export PKG_CONFIG_PATH=/opt/homebrew/lib/pkgconfig

**Memory Issues During Build**:

.. code-block:: bash

   # Reduce parallel compilation
   export CARGO_BUILD_JOBS=2

   # Use incremental compilation
   export CARGO_INCREMENTAL=1

Platform-Specific Issues
~~~~~~~~~~~~~~~~~~~~~~~

**Linux: Missing System Libraries**:

.. code-block:: bash

   # Install missing development packages
   sudo apt install build-essential python3-dev libffi-dev

**macOS: Xcode Issues**:

.. code-block:: bash

   # Update Xcode command line tools
   sudo xcode-select --install
   sudo xcode-select --reset

**Windows: MSVC Issues**:

.. code-block:: powershell

   # Ensure correct Visual Studio version
   # Use Visual Studio Installer to modify/repair installation

Getting Help
-----------

If you encounter issues building from source:

1. **Check Documentation**: Review this guide and error messages carefully
2. **Search Issues**: Look for similar problems in GitHub issues
3. **Create Issue**: Report build problems with:
   - Operating system and version
   - Rust and Python versions
   - Complete error messages
   - Build command used

**Useful Debug Information**:

.. code-block:: bash

   # System information
   uname -a
   python --version
   rustc --version
   cargo --version

   # Build with verbose output
   maturin develop --release --verbose

   # Environment variables
   env | grep -E "(RUST|CARGO|PKG_CONFIG)"

Building from source gives you full control over optimizations and features, enabling you to customize Groggy for your specific use case and hardware configuration.