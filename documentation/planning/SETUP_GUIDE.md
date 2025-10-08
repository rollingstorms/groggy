# Groggy Development Setup Guide

## Local Development Setup

### Building and Installing Locally

Groggy uses **maturin** to build the Rust extension and install it locally for development. Always work with the local build rather than a pip-installed version.

#### Step 1: Uninstall any pip-installed groggy

```bash
pip uninstall groggy
```

Confirm it's removed:
```bash
pip list | grep groggy  # Should return nothing
```

#### Step 2: Build and install locally with maturin

From the repository root:

```bash
# Build in release mode (optimized, recommended)
maturin develop --release

# Or for faster builds during development (less optimized)
maturin develop
```

This installs groggy as an **editable** package, pointing to your local `python-groggy/python/groggy/` directory.

#### Step 3: Verify installation

```bash
python -c "import groggy; print('✅ Groggy location:', groggy.__file__)"
```

Should output:
```
✅ Groggy location: /Users/michaelroth/Documents/Code/groggy/python-groggy/python/groggy/__init__.py
```

### When to Rebuild

You need to run `maturin develop --release` after:
- ✅ Changing Rust code in `src/`
- ✅ Changing FFI bindings in `python-groggy/src/ffi/`
- ❌ Changing Python code in `python-groggy/python/groggy/` (editable install, no rebuild needed)

### Running Tests

#### Module Tests (Recommended)
```bash
# All module tests
pytest tests/modules/ -v

# Specific test file
pytest tests/modules/test_graph_core.py -v

# Array-related tests
pytest tests/modules/test_*array*.py -v

# Current stats: 361 passed, 6 failed, 28 skipped
```

#### Comprehensive Library Test
```bash
python comprehensive_library_testing.py
```

This discovers and tests every method in the API. Results are saved to CSV files.

#### Using Comprehensive Test Results
```python
from comprehensive_library_testing import load_comprehensive_test_graph, iter_method_results

# Load as a graph (nodes = objects, edges = methods)
g = load_comprehensive_test_graph()

# Or as a table for columnar analysis
gt = load_comprehensive_test_graph(to_graph=False)

# Iterate through method results
for result in iter_method_results(only_failed=True):
    print(f"{result['object_name']}.{result['method_name']}: {result['message']}")
```

### Test Coverage Analysis

```bash
# Analyze which methods are tested in module tests
python analyze_test_coverage.py
```

Note: The AST-based analysis has limitations with inheritance-based testing patterns. See `ACTUAL_TEST_COVERAGE.md` for the accurate picture.

---

## Common Issues

### ImportError: No module named 'groggy'

**Solution:** Run `maturin develop --release`

### ImportError: api_meta_graph_extractor

**Solution:** The script adds the correct paths automatically. Make sure you're running from the repository root:
```bash
cd /Users/michaelroth/Documents/Code/groggy
python comprehensive_library_testing.py
```

### Old groggy version being imported

**Solution:**
1. Check where groggy is imported from:
   ```bash
   python -c "import groggy; print(groggy.__file__)"
   ```
2. If it's not from `python-groggy/python/groggy/`, uninstall the pip version:
   ```bash
   pip uninstall groggy
   maturin develop --release
   ```

### Tests failing after Rust changes

**Solution:** Rebuild the extension:
```bash
maturin develop --release
```

---

## Development Workflow

### Standard Workflow

1. **Make changes** to Rust code in `src/` or FFI in `python-groggy/src/ffi/`
2. **Rebuild**: `maturin develop --release`
3. **Run tests**: `pytest tests/modules/test_relevant_module.py -v`
4. **Validate**: `python comprehensive_library_testing.py` (optional, for full coverage check)
5. **Commit** changes

### Python-only changes

1. **Edit** Python files in `python-groggy/python/groggy/`
2. **No rebuild needed** (editable install)
3. **Run tests**: `pytest tests/modules/ -v`
4. **Commit** changes

### Before committing

```bash
# Format Rust code
cargo fmt --all

# Lint Rust code
cargo clippy --all-targets -- -D warnings

# Format Python code
black .
isort .

# Run pre-commit hooks
pre-commit run --all-files

# Run tests
pytest tests/modules/ -v
```

---

## Project Structure

```
groggy/
├── src/                          # Rust core implementation
│   ├── graph/                    # Graph data structures
│   ├── algorithms/               # Graph algorithms
│   └── ...
├── python-groggy/
│   ├── src/ffi/                  # FFI translation layer
│   └── python/groggy/            # Python API (editable)
├── tests/
│   ├── modules/                  # Structured module tests (MAIN)
│   │   ├── test_graph_core.py
│   │   ├── test_array_base.py
│   │   └── ...
│   └── ...
├── comprehensive_library_testing.py   # API discovery & testing
├── analyze_test_coverage.py          # Coverage analysis
└── documentation/
    └── meta_api_discovery/           # API extraction tools
```

---

## Testing Philosophy

### Module Tests (`tests/modules/`)
- **Purpose:** Structured, comprehensive testing of each module
- **Pattern:** Inheritance-based with shared test base classes
- **Coverage:** 361 passing tests across 18 modules
- **When to use:** Regular development, CI/CD, targeted debugging

### Comprehensive Test (`comprehensive_library_testing.py`)
- **Purpose:** Discover entire API surface and test every method
- **Pattern:** Brute-force testing with automatic parameter generation
- **Coverage:** 921 methods across 27 objects
- **When to use:** Release validation, API coverage analysis, finding regressions

Both systems complement each other:
- Comprehensive test **discovers** what exists
- Module tests provide **surgical, well-structured** coverage
- Together they ensure **complete** library validation

---

## Quick Reference

```bash
# Setup
maturin develop --release

# Test everything
pytest tests/modules/ -v

# Test specific component
pytest tests/modules/test_graph_core.py -v

# Run comprehensive test
python comprehensive_library_testing.py

# Check coverage analysis
python analyze_test_coverage.py

# Format code
cargo fmt --all && black . && isort .

# Lint code
cargo clippy --all-targets -- -D warnings
```

---

## Current Test Status

**Module Tests:** 361 passed, 6 failed, 28 skipped (92% pass rate)

**Comprehensive Test:** 613/921 methods passing (66.6% success rate)

**Well-tested components:**
- ✅ Graph core operations
- ✅ Accessor views (NodesAccessor, EdgesAccessor)
- ✅ Array types (BaseArray, NodesArray, EdgesArray, NumArray)
- ✅ Table types (BaseTable, GraphTable, NodesTable, EdgesTable)
- ✅ Matrix operations (GraphMatrix)
- ✅ Subgraph operations (SubgraphArray, ComponentsArray)

See `ACTUAL_TEST_COVERAGE.md` for detailed coverage analysis.
