# Groggy Modular Testing Strategy

## Executive Summary

We have **~900 methods** across **23 object types**, with current coverage at **71.3% (405/568 tested methods passing)**. The main challenge is managing the complexity of testing diverse method signatures and parameter combinations across object hierarchies.

## Current State Analysis

### Test Results by Object (Sorted by Failure Count)

| Object | Pass Rate | Methods | Issues |
|--------|-----------|---------|--------|
| **Subgraph** | 50.0% | 62 | 31 failures - highest priority |
| **GraphMatrix** | 72.8% | 92 | 25 failures - complex operations |
| **BaseArray** | 70.5% | 61 | 18 failures - foundation type |
| **Graph** | 76.6% | 64 | 15 failures - core object |
| **EdgesTable** | 64.9% | 37 | 13 failures - table operations |
| **NodesTable** | 62.5% | 32 | 12 failures - table operations |
| **GraphTable** | 52.2% | 23 | 11 failures - high-level API |
| NodesArray | 69.2% | 13 | 4 failures |
| EdgesArray | 80.0% | 15 | 3 failures |
| SubgraphArray | 76.9% | 13 | 3 failures |
| NodesAccessor | 84.6% | 13 | 2 failures |
| NumArray | 87.5% | 16 | 2 failures |
| EdgesAccessor | 92.9% | 14 | 1 failure |
| ComponentsArray | 87.5% | 8 | 1 failure |

### Failure Categories

1. **Missing Parameters (110 failures - 67.5%)**: Test framework needs proper parameter provisioning
2. **Other Errors (37 failures - 22.7%)**: Logic issues, precondition failures, state management
3. **Type Conversion (8 failures - 4.9%)**: FFI type marshaling issues
4. **Not Implemented (8 failures - 4.9%)**: Known gaps in implementation

## Modular Testing Strategy - Milestone Approach

Each module represents a **milestone** with complete testing infrastructure, comprehensive documentation, and validated functionality. We implement module-by-module, leaving amazing documentation and test patterns for future developers.

### Milestone 1: Graph Core Foundation

Test the core Graph object that all other objects depend on.

#### Module 1.1: Graph Core
- **Object**: `Graph` (64 methods, 15 failures)
- **Focus**: Node/edge CRUD, attributes, queries, state management
- **Approach**:
  - Create fixture library with diverse graph structures
  - Test in isolation: empty graph, single node, connected components, cycles
  - Parametric tests for attribute types (int, float, str, bool, list, dict)
  - Edge cases: self-loops, multi-edges, missing nodes/edges
- **Test File**: `tests/modules/test_graph_core.py`
- **Documentation**: Complete test pattern guide in `tests/modules/README.md`
- **Success Criteria**: 95%+ pass rate, all CRUD operations stable, reusable test patterns documented

### Milestone 2: Array Foundation
- **Objects**: `NumArray` (16 methods), `NodesArray` (13), `EdgesArray` (15), `BaseArray` (61)
- **Focus**: Array operations, indexing, slicing, aggregation, transformation
- **Approach**:
  - Shared test base class for common array operations
  - Type-specific tests for NumArray (numeric ops), NodesArray/EdgesArray (graph ops)
  - Test array builders separately from instance methods
  - Fixture: Arrays of varying sizes (0, 1, 10, 100, 1000 elements)
- **Test Files**:
  - `tests/modules/test_array_base.py`
  - `tests/modules/test_num_array.py`
  - `tests/modules/test_graph_arrays.py`
- **Documentation**: Array testing patterns and shared base classes documented
- **Success Criteria**: 90%+ pass rate, operations compose correctly, base test classes established

### Milestone 3: Accessor Views

Test objects that provide filtered views and bulk operations.

#### Module 3.1: Accessors
- **Objects**: `NodesAccessor` (13 methods), `EdgesAccessor` (14 methods)
- **Focus**: Filtered access, bulk operations, attribute getters/setters
- **Approach**:
  - Test against pre-populated graph fixtures
  - Parametric filters (node IDs, attribute conditions, combinations)
  - Bulk vs. single-item operations
  - State synchronization with parent Graph
- **Test File**: `tests/modules/test_accessors.py`
- **Documentation**: Accessor testing patterns, filter composition examples
- **Success Criteria**: 95%+ pass rate, filter composition works, accessor patterns documented

### Milestone 4: Table Operations

Test objects that provide columnar operations and I/O.

#### Module 4.1: Tables
- **Objects**: `GraphTable` (23 methods), `NodesTable` (32), `EdgesTable` (37)
- **Focus**: Columnar operations, filtering, joins, aggregations, I/O
- **Approach**:
  - Shared table test framework (DRY principle)
  - Test columnar vs. row-oriented access
  - CSV/JSON import/export round-trips
  - Aggregation correctness (sum, mean, count, custom)
  - Join operations between tables
- **Test Files**:
  - `tests/modules/test_table_base.py`
  - `tests/modules/test_graph_table.py`
  - `tests/modules/test_nodes_edges_tables.py`
- **Documentation**: Table testing framework, I/O testing patterns, shared base classes
- **Success Criteria**: 90%+ pass rate, I/O preserves data, table patterns well-documented

### Milestone 5: Subgraph Operations (HIGH PRIORITY)

Test complex grouping and subgraph operations - highest failure count.

#### Module 5.1: Subgraphs
- **Objects**: `Subgraph` (62 methods - PRIORITY), `SubgraphArray` (13 methods)
- **Focus**: Grouping, graph operations on subgraphs, aggregation, iteration
- **Approach**:
  - Test creation: manual, group_by, filtering, connected components
  - Operations: collapse, aggregate, export, visualization
  - Array operations: iteration, indexing, bulk operations
  - Edge cases: empty subgraphs, overlapping subgraphs, disconnected
- **Test Files**:
  - `tests/modules/test_subgraph.py`
  - `tests/modules/test_subgraph_array.py`
- **Documentation**: Subgraph testing patterns, grouping operations, complex use cases
- **Success Criteria**: 85%+ pass rate (50% → 85% improvement target), subgraph patterns documented

### Milestone 6: Matrix & Advanced Arrays

Test matrix operations and specialized array types.

#### Module 6.1: Matrix & Special Arrays
- **Objects**: `GraphMatrix` (92 methods), `ComponentsArray` (8), `TableArray` (8)
- **Focus**: Matrix operations, linear algebra, special array types
- **Approach**:
  - Test matrix builders (adjacency, laplacian, transition, custom)
  - Sparse vs. dense representations
  - Matrix operations: multiply, transpose, eigenvalues
  - Components: detection, iteration, merge
- **Test Files**:
  - `tests/modules/test_graph_matrix.py`
  - `tests/modules/test_special_arrays.py`
- **Documentation**: Matrix testing patterns, linear algebra operations, performance benchmarks
- **Success Criteria**: 80%+ pass rate (complex operations, some known gaps), matrix patterns documented

### Milestone 7: Utility Types

Test supporting ID and configuration types.

#### Module 7.1: ID & Config Types
- **Objects**: `NodeId`, `EdgeId`, `StateId`, `BranchName`, `AttrName`, `DisplayConfig`
- **Focus**: Type safety, string operations, serialization, validation
- **Approach**:
  - Test type conversion and validation
  - String manipulation (for string-based IDs)
  - Config serialization/deserialization
  - Integration with main objects
- **Test File**: `tests/modules/test_utility_types.py`
- **Documentation**: Type safety patterns, validation approaches
- **Success Criteria**: 95%+ pass rate (simple types), type patterns documented

### Milestone 8: Integration Workflows

Test cross-module integration and real-world workflows.

#### Module 8.1: Cross-Module Integration
- **Focus**: Composability, data flow between modules, real-world workflows
- **Approach**:
  - Persona-driven scenarios (from personas/ docs)
  - Data pipeline tests: Graph → Table → Array → Export
  - Visualization workflows
  - Performance benchmarks for common operations
- **Test File**: `tests/integration/test_cross_module_workflows.py`
- **Documentation**: Complete integration testing guide, persona workflow examples
- **Success Criteria**: All personas' core workflows work end-to-end, integration patterns documented

## Testing Infrastructure Improvements

### 1. Smart Fixture Generation

Create a fixture library that automatically generates valid test data for each method signature.

```python
# tests/fixtures/smart_fixtures.py

class FixtureFactory:
    """Generate valid test data based on method signatures"""
    
    def __init__(self, graph: Graph):
        self.graph = graph
        self.node_ids = []
        self.edge_ids = []
        
    def get_fixture_for_param(self, param_name: str, param_type: type):
        """Return valid test value for a parameter"""
        if param_name in ['node_id', 'source', 'target']:
            return self.node_ids[0] if self.node_ids else None
        elif param_name == 'edge_id':
            return self.edge_ids[0] if self.edge_ids else None
        elif param_name in ['attrs', 'attrs_dict']:
            return {'test_attr': 'value'}
        # ... more mappings
        
    def generate_test_cases(self, method):
        """Generate multiple valid test cases for a method"""
        sig = inspect.signature(method)
        # Use type hints and parameter names to generate cases
        # Return list of (args, kwargs) tuples
```

### 2. Parametric Test Framework

Use pytest parametrize to test methods with multiple input combinations efficiently.

```python
# tests/conftest.py

import pytest
from fixtures.smart_fixtures import FixtureFactory

@pytest.fixture
def graph_fixture():
    """Standard test graph used across modules"""
    g = gr.Graph()
    # Add standard test data
    return g, FixtureFactory(g)

@pytest.fixture
def method_test_cases(request):
    """Generate test cases for a method"""
    obj, method_name = request.param
    factory = FixtureFactory(...)
    return factory.generate_test_cases(getattr(obj, method_name))
```

### 3. Failure Analysis Dashboard

Enhance comprehensive_library_testing.py output:

```python
def generate_test_report(results):
    """Generate HTML dashboard with:
    - Pass/fail breakdown by module
    - Failure clustering by error type
    - Method coverage heatmap
    - Trend analysis (if historical data exists)
    - Quick links to failing test code
    """
```

### 4. Incremental Testing

```python
# Run only tests for changed modules
python -m pytest tests/modules/test_graph_core.py -v

# Run full suite with coverage
python -m pytest tests/modules/ --cov=groggy --cov-report=html

# Run integration tests after modules pass
python -m pytest tests/integration/ -v
```

## Test Organization

```
tests/
├── modules/                    # Phase 1-3: Modular tests
│   ├── __init__.py
│   ├── test_graph_core.py      # Module 1.1
│   ├── test_array_base.py      # Module 1.2
│   ├── test_num_array.py
│   ├── test_graph_arrays.py
│   ├── test_accessors.py       # Module 2.1
│   ├── test_table_base.py      # Module 2.2
│   ├── test_graph_table.py
│   ├── test_nodes_edges_tables.py
│   ├── test_subgraph.py        # Module 3.1 (PRIORITY)
│   ├── test_subgraph_array.py
│   ├── test_graph_matrix.py    # Module 3.2
│   ├── test_special_arrays.py
│   └── test_utility_types.py   # Module 4.1
├── integration/                # Phase 4: Cross-module
│   ├── test_cross_module_workflows.py
│   ├── test_persona_engineer.py
│   ├── test_persona_researcher.py
│   └── test_persona_analyst.py
├── fixtures/                   # Shared test infrastructure
│   ├── __init__.py
│   ├── smart_fixtures.py
│   ├── graph_samples.py
│   └── data/
│       ├── small_graph.csv
│       ├── medium_graph.csv
│       └── large_graph.csv
└── conftest.py                 # Pytest configuration
```

## Development Workflow

### Per Module:

1. **Analyze Current Failures**
   ```bash
   python comprehensive_library_testing.py
   # Review CSV for specific module failures
   ```

2. **Create Fixtures**
   - Add module-specific fixtures to `fixtures/`
   - Ensure fixtures cover edge cases

3. **Write Tests**
   - Start with passing methods (ensure they stay passing)
   - Add tests for failing methods
   - Use parametrize for multiple input combinations

4. **Fix Issues**
   - Fix test framework issues (missing parameters)
   - Fix actual bugs in Rust/FFI/Python
   - Document known limitations

5. **Validate**
   ```bash
   pytest tests/modules/test_MODULE.py -v
   maturin develop --release  # If Rust changes
   pytest tests/modules/test_MODULE.py -v  # Retest
   ```

6. **Update Comprehensive Test**
   ```bash
   python comprehensive_library_testing.py
   # Verify improvements
   ```

### Success Metrics Per Phase:

- **Phase 1**: Foundation pass rate 90%+ (currently ~80%)
- **Phase 2**: Accessor/Table pass rate 85%+ (currently ~65%)
- **Phase 3**: Advanced structures pass rate 80%+ (currently ~60%)
- **Phase 4**: Integration tests all passing, 90%+ overall

## Known Issues to Address

### High Priority (Blocking Multiple Tests)

1. **Missing Parameter Handling**: Test framework needs smart defaults for required parameters
2. **Subgraph Operations**: 31 failures, core functionality, affects grouping/aggregation
3. **Table Operations**: 36 combined failures across table types, columnar operations
4. **GraphMatrix Operations**: 25 failures, advanced linear algebra operations

### Medium Priority

5. **Type Conversion in FFI**: 8 failures, dict/list conversions between Python/Rust
6. **Accessor Bulk Operations**: set_attrs() needs proper dict handling
7. **Array Builders**: BaseArray builder has 18 failures, used by multiple types

### Low Priority (Can Document as Limitations)

8. **Not Implemented Methods**: 8 methods intentionally not implemented yet
9. **Edge Case Error Messages**: Improve error messages for better debugging

## Milestone Timeline & Documentation Requirements

Each milestone must include:
- **Complete test suite** with 85%+ pass rate target
- **Comprehensive documentation** in `tests/modules/README.md` and module-specific docs
- **Reusable patterns** for future modules
- **Performance benchmarks** where applicable
- **Integration examples** showing module interactions

### Milestone Schedule
- **M1**: Graph Core Foundation - Core object stability
- **M2**: Array Foundation - Array patterns and base classes
- **M3**: Accessor Views - Filter composition patterns
- **M4**: Table Operations - Columnar operation patterns + I/O
- **M5**: Subgraph Operations - Complex grouping (HIGH PRIORITY - 31 failures)
- **M6**: Matrix & Advanced Arrays - Linear algebra patterns
- **M7**: Utility Types - Type safety patterns
- **M8**: Integration Workflows - End-to-end persona scenarios

**Target**: 8 milestones to comprehensive, well-tested library at 90%+ pass rate with exceptional documentation.

## Decision Points to Discuss

Before implementing, we need to decide:

1. **Test Organization**: Should we use the proposed `tests/modules/` structure or integrate with existing test files?
2. **Fixture Approach**: Smart auto-generation vs. hand-crafted fixtures per module?
3. **Priority Order**: Start with highest-failure modules (Subgraph) or foundation-first (Graph)?
4. **Coverage Goal**: Aim for 90%+ or pragmatic 80% with known limitations documented?
5. **Timeline**: Is 8 weeks realistic, or should we focus on critical path first?

## Next Steps (For Discussion)

1. **Review this plan together** - adjust approach based on your preferences
2. **Pick the first module** - which should we tackle first?
3. **Define test patterns** - establish conventions before writing code
4. **Set success criteria** - what does "done" look like for each module?

## Notes

- Focus on **quality over speed** - better to have 90% of tests solid than 100% brittle
- **Preserve existing passing tests** - don't break what works
- **Document known limitations** - some methods may have intentional constraints
- **Benchmark performance** - catch regressions early with `cargo bench`
- **Keep tests maintainable** - DRY principle, clear naming, good documentation
