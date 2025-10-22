# Installation

## Quick Install

The easiest way to install Groggy is via pip:

```bash
pip install groggy
```

This installs the latest stable release from PyPI with pre-built wheels for major platforms.

---

## Supported Platforms

Groggy provides pre-built wheels for:

- **Linux**: x86_64, aarch64 (Python 3.8+)
- **macOS**: x86_64 (Intel), arm64 (Apple Silicon) (Python 3.8+)
- **Windows**: x86_64 (Python 3.8+)

If a wheel isn't available for your platform, pip will attempt to build from source (requires Rust toolchain).

---

## Install from Source

### Prerequisites

To build Groggy from source, you need:

1. **Python 3.8+**
2. **Rust toolchain** (stable)
3. **maturin** (Python-Rust build tool)

### Step 1: Install Rust

If you don't have Rust installed:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Follow the prompts to complete installation. Then update your PATH:

```bash
source $HOME/.cargo/env
```

Verify installation:

```bash
rustc --version
cargo --version
```

### Step 2: Clone the Repository

```bash
git clone https://github.com/rollingstorms/groggy.git
cd groggy
```

### Step 3: Install maturin

```bash
pip install maturin
```

### Step 4: Build and Install

For development (editable install):

```bash
cd python-groggy
maturin develop
```

For production (optimized build):

```bash
cd python-groggy
maturin develop --release
```

The `--release` flag enables Rust optimizations (much faster, but slower to compile).

---

## Verify Installation

Test that Groggy is installed correctly:

```python
import groggy as gr

# Create a simple graph
g = gr.Graph()
g.add_node(name="Test")

print(f"Groggy {gr.__version__} installed successfully!")
```

Expected output:
```
Groggy 0.5.1 installed successfully!
```

---

## Optional Dependencies

Groggy has minimal required dependencies, but some features require additional packages:

### Visualization

For graph visualization:

```bash
pip install matplotlib networkx
```

### Data Import/Export

For working with various file formats:

```bash
pip install pandas pyarrow
```

### Neural Networks

For graph neural network functionality:

```bash
pip install torch
```

### All Optional Dependencies

Install everything at once:

```bash
pip install groggy[all]
```

---

## Development Installation

If you're contributing to Groggy, install development dependencies:

```bash
git clone https://github.com/rollingstorms/groggy.git
cd groggy

# Install Python dev dependencies
pip install -r requirements-dev.txt

# Build in development mode
cd python-groggy
maturin develop

# Run tests
cd ..
python -m pytest tests/
```

### Running Tests

```bash
# Python tests
python -m pytest tests/

# Rust tests
cargo test

# All tests
cargo test && python -m pytest tests/
```

### Code Formatting

```bash
# Format Rust code
cargo fmt

# Check Rust lints
cargo clippy

# Format Python code
black python-groggy/python/
```

---

## Troubleshooting

### "No matching distribution found"

If pip can't find a wheel for your platform, you'll need to build from source. Make sure you have the Rust toolchain installed (see above).

### "maturin: command not found"

Install maturin:

```bash
pip install maturin
```

### "rustc: command not found"

Install the Rust toolchain:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
source $HOME/.cargo/env
```

### Build Errors on macOS

If you encounter linker errors on macOS, ensure Xcode Command Line Tools are installed:

```bash
xcode-select --install
```

### "ModuleNotFoundError: No module named '_groggy'"

This usually means the Rust extension didn't compile. Try rebuilding:

```bash
cd python-groggy
maturin develop --release
```

### Performance Issues

If Groggy feels slow, make sure you're using a release build:

```bash
cd python-groggy
maturin develop --release  # Note the --release flag
```

Debug builds are 10-100x slower than release builds.

---

## Updating Groggy

### From PyPI

```bash
pip install --upgrade groggy
```

### From Source

```bash
cd groggy
git pull origin main
cd python-groggy
maturin develop --release
```

---

## Uninstalling

```bash
pip uninstall groggy
```

---

## Next Steps

Now that Groggy is installed, try the [Quickstart Guide](quickstart.md) to build your first graph!

For more advanced usage, check out the [User Guide](guide/graph-core.md).
