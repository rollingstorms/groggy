# Groggy Tests

This directory contains tests and validation scripts for Groggy.

## Test Scripts

### Documentation Validation
- **`test_documentation_validation.py`**: Comprehensive validation of all documented features (95%+ working)
- **`simple_validation_test.py`**: Quick validation test for basic functionality
- **`validation_test_suite.py`**: Full validation suite with detailed reporting

### Running Tests

```bash
# Run comprehensive documentation validation
python tests/test_documentation_validation.py

# Run simple validation
python tests/simple_validation_test.py

# Run full test suite
python tests/validation_test_suite.py
```

### Test Results

Current validation shows **95%+ documented features working correctly**:
- ✅ **18+ core features** validated and working
- ✅ **Graph operations**: Node/edge creation, analytics, filtering
- ✅ **Data structures**: GraphArray, GraphTable, GraphMatrix operations
- ✅ **Statistical operations**: mean(), min(), max(), describe()
- ✅ **Export compatibility**: to_networkx(), to_pandas(), to_numpy()

### Test Categories

1. **Core Graph Operations**: Basic graph creation, node/edge operations
2. **Data Structure Tests**: Array, table, matrix creation and operations  
3. **Statistical Operations**: Mathematical operations on graph data
4. **Analytics Tests**: Connected components, traversal algorithms
5. **Export Tests**: Integration with NetworkX, Pandas, NumPy
6. **Display Tests**: Rich formatting and Unicode display

### Rust Unit Tests

Rust unit tests are embedded in the core library modules:

```bash
# Run Rust unit tests
cargo test

# Run specific module tests
cargo test core::array
cargo test core::table
```