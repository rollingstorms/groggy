# GLI Tests and Examples

This directory contains test scripts, benchmarks, and tutorials for the GLI (Graph Language Interface) library.

## Files Overview

### Tutorial and Documentation
- **`gli_tutorial.ipynb`** - Comprehensive Jupyter notebook tutorial demonstrating all GLI features
- **`gli_tests.ipynb`** - Development notebook with various test scenarios

### Performance Tests
- **`simple_performance_test.py`** - Basic performance comparison between Python and Rust backends
- **`rust_stress_test.py`** - Stress test specifically for the Rust backend with large graphs
- **`performance_benchmark.py`** - Detailed benchmarking suite for both backends

### Advanced Testing
- **`complexity_stress_test.py`** - Tests complex attributes and nested data structures
- **`advanced_complexity_test.py`** - Advanced scenarios with realistic data and complex queries
- **`ultimate_stress_test.py`** - Comprehensive test combining scale, complexity, and concurrent operations
- **`extreme_stress_test.py`** - Maximum stress test for performance limits

### Development Tools
- **`debug_delta.py`** - Debugging utilities for graph state changes

## Running the Tests

### Prerequisites
Make sure you have GLI installed and built:
```bash
# From the root directory
pip install -e .
```

If using the Rust backend:
```bash
# Build the Rust components
cargo build --release
```

### Running Performance Tests
```bash
# Basic performance comparison
python tests/simple_performance_test.py

# Rust-specific stress test
python tests/rust_stress_test.py

# Comprehensive benchmarking
python tests/performance_benchmark.py
```

### Running the Tutorial
Open `gli_tutorial.ipynb` in Jupyter Lab or VS Code:
```bash
jupyter lab tests/gli_tutorial.ipynb
```

### Expected Performance
- **Python Backend**: Good for graphs <1K nodes, development and prototyping
- **Rust Backend**: Excellent for large graphs (tested up to 2M+ nodes), production workloads

## Test Results Summary

The tests validate:
- ✅ Both Python and Rust backends work correctly
- ✅ Backend switching functions properly
- ✅ Complex attributes and nested data structures
- ✅ High-performance operations on large graphs
- ✅ Memory efficiency and scalability
- ✅ Concurrent operations and thread safety
- ✅ Real-world usage scenarios

## Contributing

When adding new tests:
1. Follow the existing naming convention
2. Include performance benchmarks where relevant
3. Add docstrings explaining the test purpose
4. Update this README with new test descriptions
