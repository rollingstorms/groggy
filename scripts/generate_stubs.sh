#!/bin/bash
# Generate Python type stub files (.pyi) for Groggy
#
# This script builds the extension and generates type stubs for IDE autocomplete.
# Run this after making changes to the Rust FFI layer.

set -e

cd "$(dirname "$0")/.."

echo "🔨 Generating Python type stubs for Groggy..."
echo

# Step 1: Build the extension (if needed)
if [ "$1" != "--skip-build" ]; then
    echo "📦 Building Rust extension..."
    maturin develop --release
    echo
fi

# Step 2: Generate stubs
echo "📝 Generating type stubs..."
python scripts/generate_stubs.py

echo
echo "✅ Type stubs generated successfully!"
echo
echo "📚 Test autocomplete in Jupyter:"
echo "   import groggy"
echo "   g = groggy.Graph()"
echo "   g.<TAB>  # Should show all methods!"
echo
