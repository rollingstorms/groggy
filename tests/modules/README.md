# Groggy Modular Testing Documentation

## Overview

This directory contains the **Milestone-based Modular Testing Strategy** for Groggy. Each module represents a complete testing milestone with documented patterns, comprehensive coverage, and reusable infrastructure.

## 🎯 Milestone Approach

Instead of phases, we implement **8 distinct milestones**, each delivering:

- ✅ **Complete test suite** with 85%+ pass rate target
- 📚 **Comprehensive documentation** with reusable patterns
- 🔧 **Testing infrastructure** for future modules
- 📊 **Performance benchmarks** where applicable
- 🔗 **Integration examples** showing module interactions

## 📁 Module Structure

```
tests/modules/
├── README.md                    # This file - comprehensive testing guide
├── test_graph_core.py          # M1: Graph Core Foundation
├── test_array_base.py          # M2: Array Foundation (pending)
├── test_accessors.py           # M3: Accessor Views (pending)
├── test_table_*.py             # M4: Table Operations (pending)
├── test_subgraph*.py           # M5: Subgraph Operations (HIGH PRIORITY)
├── test_graph_matrix.py        # M6: Matrix & Advanced Arrays (pending)
├── test_utility_types.py       # M7: Utility Types (pending)
└── integration/                # M8: Integration Workflows (pending)
```

## 🏗️ Testing Infrastructure

### Smart Fixtures (`tests/fixtures/`)

The testing infrastructure provides intelligent fixture generation:

```python
from tests.fixtures import FixtureFactory, GraphFixtures, load_test_graph

# Smart parameter generation
factory = FixtureFactory(graph)
test_cases = factory.generate_test_cases(obj, "method_name")

# Pre-built graph structures
graph = GraphFixtures.attributed_graph()  # Rich attributes
graph = load_test_graph("karate")          # Zachary's Karate Club
graph = load_test_graph("social")          # Small social network
```

### Parametric Testing Patterns

```python
@pytest.mark.parametrize("graph_type", ["path", "cycle", "star", "complete"])
def test_across_graph_structures(graph_type):
    graph = load_test_graph(graph_type)
    # Test works across all graph types
```

### Error Condition Testing

```python
def test_invalid_operations(self, empty_graph):
    # Test graceful handling of edge cases
    assert not empty_graph.contains_node(999999)

    with pytest.raises(Exception):
        empty_graph.add_edge(999999, 888888)
```

## 📈 Milestone Progress

### ✅ M1: Graph Core Foundation (COMPLETED)

**Status**: Infrastructure complete, ready for testing
**Coverage**: 64 methods across core Graph functionality
**Current Issues**: 15 failures identified (see test results)

**Key Features Implemented**:
- Smart fixture generation for all parameter types
- Comprehensive test patterns for CRUD operations
- Performance benchmarks for core operations
- Error condition validation
- Parametric testing across graph structures

**Test Categories**:
- ✅ Graph creation and basic properties
- ✅ Node CRUD operations (add_node, add_nodes, contains_node)
- ✅ Edge CRUD operations (add_edge, add_edges, contains_edge)
- ✅ Attribute operations (get/set node/edge attributes)
- ✅ Graph queries and filters
- ✅ State management (commit, branches, checkout)
- ✅ Basic algorithms (BFS, DFS, connected components)
- ✅ Error conditions and edge cases
- ✅ Performance validation

**Reusable Patterns Established**:
- Parameter generation based on type hints and names
- Graph state management for consistent test environments
- Performance testing with timing assertions
- Error condition testing with graceful failure handling
- Parametric testing across multiple graph structures

### 🟡 M2: Array Foundation (NEXT)

**Target**: Array operations, indexing, slicing, aggregation
**Objects**: `NumArray` (16 methods), `NodesArray` (13), `EdgesArray` (15), `BaseArray` (61)
**Key Pattern**: Shared test base class for common array operations

### 🟡 M3-M8: Remaining Milestones

See `documentation/planning/modular_testing_strategy.md` for full milestone definitions.

## 🧪 Testing Patterns Guide

### 1. Smart Fixture Usage

**Problem**: Tests need diverse, valid parameter combinations
**Solution**: `FixtureFactory` automatically generates parameters based on method signatures

```python
def test_method_with_smart_fixtures(self, graph_with_factory):
    graph, factory = graph_with_factory

    # Generate test cases automatically
    test_cases = factory.generate_test_cases(graph, "add_edge")

    for case in test_cases:
        if case.should_succeed:
            result = graph.add_edge(*case.args, **case.kwargs)
            assert result is not None
        else:
            with pytest.raises(case.expected_exception):
                graph.add_edge(*case.args, **case.kwargs)
```

### 2. Parametric Graph Testing

**Problem**: Methods should work across different graph structures
**Solution**: Parametric fixtures testing multiple graph types

```python
@pytest.mark.parametrize("graph_name", ["empty", "path", "cycle", "star", "complete"])
def test_method_across_structures(graph_name):
    graph = load_test_graph(graph_name)

    # Test the method works regardless of graph structure
    result = graph.some_method()
    assert result is not None
```

### 3. Performance Validation

**Problem**: Core operations must meet performance standards
**Solution**: Embedded performance tests with timing assertions

```python
@pytest.mark.performance
def test_bulk_operation_performance(self, empty_graph):
    import time

    start_time = time.time()

    # Perform bulk operation
    for i in range(1000):
        empty_graph.add_node(index=i)

    elapsed = time.time() - start_time

    # Assert performance requirement
    assert elapsed < 1.0  # Should complete in under 1 second
    assert len(empty_graph.nodes) == 1000
```

### 4. Error Condition Testing

**Problem**: Methods should handle invalid inputs gracefully
**Solution**: Systematic testing of error conditions

```python
def test_invalid_inputs(self, empty_graph):
    # Test non-existent IDs
    assert not empty_graph.contains_node(999999)

    # Test operations that should fail
    with pytest.raises((ValueError, KeyError, Exception)):
        empty_graph.add_edge(999999, 888888)

    # Test operations on empty state
    result = empty_graph.all_node_attribute_names()
    assert isinstance(result, list)
    assert len(result) == 0
```

### 5. State Consistency Validation

**Problem**: Operations should maintain graph consistency
**Solution**: Helper functions validate graph state

```python
from tests.conftest import assert_graph_valid

def test_operation_maintains_consistency(self, simple_graph):
    # Perform operation
    node_id = simple_graph.add_node(label="Test")

    # Validate graph remains consistent
    assert_graph_valid(simple_graph)
    assert simple_graph.contains_node(node_id)
```

## 🚀 Running Tests

### Quick Commands

```bash
# Run all Graph Core tests
pytest tests/modules/test_graph_core.py -v

# Run specific test categories
pytest -m graph_core -v
pytest -m "graph_core and not slow" -v

# Run with coverage
pytest tests/modules/test_graph_core.py --cov=groggy --cov-report=html

# Run performance tests (they're marked as slow)
pytest -m "graph_core and performance" -v
```

### Test Organization

```bash
# Module-specific testing
pytest tests/modules/test_graph_core.py::TestNodeOperations -v

# Cross-module testing (when more modules exist)
pytest tests/modules/ -v

# Integration testing (when implemented)
pytest tests/integration/ -v
```

## 📊 Current Status & Results

### Graph Core Test Results

After implementing Module 1.1, run the comprehensive test to see improvements:

```bash
python comprehensive_library_testing.py | grep -A 10 "Graph.*methods"
```

**Expected Improvements**:
- Missing Parameter failures: 67.5% → <20% (smart fixtures)
- Method coverage: 49/64 → 55+/64 methods passing
- Error handling: Graceful failure documentation

### Known Issues Being Addressed

1. **Missing Parameter Handling** ✅ SOLVED
   Smart fixtures automatically provide valid parameters

2. **add_edges/add_nodes Bulk Operations** 🔍 IDENTIFIED
   Current failures documented with `pytest.skip()` for investigation

3. **Branch Operations** 🔍 IDENTIFIED
   Branch creation/checkout issues documented for fixing

4. **Attribute Access** 🔍 IDENTIFIED
   Node/edge attribute modification patterns documented

## 🔄 Development Workflow

### Per Milestone

1. **📋 Analyze Current Failures**
   ```bash
   python comprehensive_library_testing.py
   grep -i "Graph\|Array\|Table" comprehensive_test_results_*.json
   ```

2. **🏗️ Implement Test Infrastructure**
   - Create fixtures specific to the module
   - Establish testing patterns
   - Document reusable components

3. **✅ Write Comprehensive Tests**
   - Cover all methods in the module
   - Include edge cases and error conditions
   - Add performance tests for critical operations

4. **🔧 Fix Identified Issues**
   - Address test framework issues (missing parameters)
   - Fix actual bugs in Rust/FFI/Python layers
   - Document known limitations with clear explanations

5. **✅ Validate Improvements**
   ```bash
   pytest tests/modules/test_MODULE.py -v
   python comprehensive_library_testing.py  # Check overall improvement
   ```

### Success Metrics

- **M1 Target**: Graph Core 95%+ pass rate (currently ~76%)
- **M2 Target**: Array Foundation 90%+ pass rate
- **M3 Target**: Accessor Views 95%+ pass rate
- **M4 Target**: Table Operations 90%+ pass rate
- **M5 Target**: Subgraph Operations 85%+ pass rate (currently ~50% - HIGH PRIORITY)
- **Overall Target**: 90%+ library-wide pass rate

## 🤝 Contributing to Testing

### Adding New Test Modules

1. **Follow the established patterns** from `test_graph_core.py`
2. **Use smart fixtures** from `tests/fixtures/`
3. **Document new patterns** in this README
4. **Include performance tests** for critical operations
5. **Add comprehensive error testing**

### Test Class Organization

```python
@pytest.mark.module_name
class TestBasicOperations:
    """Test basic CRUD and core functionality"""

@pytest.mark.module_name
class TestAdvancedOperations:
    """Test complex operations and algorithms"""

@pytest.mark.module_name
class TestErrorConditions:
    """Test error handling and edge cases"""

@pytest.mark.module_name
@pytest.mark.performance
class TestPerformance:
    """Performance and stress testing"""
```

### Fixture Usage Guidelines

- **Use existing fixtures** when possible (`empty_graph`, `simple_graph`, etc.)
- **Create module-specific fixtures** for complex scenarios
- **Document fixture behavior** and expected properties
- **Make fixtures reusable** across test classes

## 📝 Documentation Standards

Every test module must include:

1. **Module docstring** explaining purpose and patterns
2. **Class docstrings** for test categories
3. **Method docstrings** for complex test scenarios
4. **Inline comments** explaining non-obvious test logic
5. **Performance assertions** with reasoning
6. **Skip conditions** with clear explanations

## 🎯 Next Steps

1. **✅ Complete M1 validation** - Run tests and measure improvements
2. **🚀 Begin M2 implementation** - Array Foundation testing
3. **📊 Monitor progress** - Track pass rates and performance
4. **🔄 Iterate and improve** - Refine patterns based on results

---

*This testing documentation is a living guide that evolves with each milestone. Patterns established here serve as the foundation for comprehensive, maintainable testing across the entire Groggy library.*