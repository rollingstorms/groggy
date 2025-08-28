# Query Parsing Migration to Pure Rust Core

## 🎯 **Strategic Goal**
Eliminate circular dependency (`Rust → Python → Rust`) and move all query parsing logic to Rust core for universal multi-language support.

## 🚨 **Current Problem**
- **Circular Dependency**: FFI layer calls `py.import("groggy.query_parser")` from Rust
- **50+ Compilation Errors**: Cascading from this architectural anti-pattern
- **Performance Issues**: Python imports within Rust FFI causing overhead
- **Multi-port Blocker**: Cannot create clean language ports (JS, Go) with current architecture

## 🏗️ **Target Architecture**

### Current (Broken)
```
Python API → Rust FFI → py.import("groggy.query_parser") → Python Parser → Rust Types
     ↑                                                                           ↓
     └─────────────────── CIRCULAR DEPENDENCY ────────────────────────────────────┘
```

### Target (Clean)
```
Python API → Rust Core Parser → Rust Filter Executor → Results
JS API     → Rust Core Parser → Rust Filter Executor → Results  
Go API     → Rust Core Parser → Rust Filter Executor → Results
```

## 📋 **Migration Plan**

### **Phase 1: Core Implementation (Week 1 - Days 1-3)**

#### **Step 1.1: Create Core Parser Structure**
**Files to create:**
- `/src/core/query_parser.rs` - Main parser implementation
- Tests in `/src/core/query_parser.rs` - Comprehensive test suite

**Implementation:**
```rust
// src/core/query_parser.rs
pub struct QueryParser {
    tokens: Vec<Token>,
    position: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub enum Token {
    Identifier(String),
    Operator(CompOp),
    LogicalOp(LogicalOp),
    Value(AttrValue),
    LeftParen,
    RightParen,
}

impl QueryParser {
    pub fn parse_node_query(&mut self, query: &str) -> Result<NodeFilter, QueryError> {
        self.tokenize(query)?;
        self.parse_expression()
    }
    
    pub fn parse_edge_query(&mut self, query: &str) -> Result<EdgeFilter, QueryError> {
        self.tokenize(query)?;
        self.parse_expression()
    }
}
```

**Query Language Support:**
- Simple comparisons: `"salary > 120000"`
- String comparisons: `"department == 'Engineering'"`
- Logical operators: `"salary > 120000 AND department == 'Engineering'"`
- Parentheses: `"(salary > 120000 AND department == 'Engineering') OR age < 25"`
- Negation: `"NOT (department == 'HR')"`

#### **Step 1.2: Update Core Module**
```rust
// src/core/mod.rs
pub mod query_parser;  // ADD THIS LINE
```

### **Phase 2: FFI Bindings (Week 1 - Days 4-5)**

#### **Step 2.1: Create FFI Parser Wrapper**
**File to create:**
- `/src/ffi/core/query_parser.rs`

```rust
#[pyclass(name = "QueryParser")]
pub struct PyQueryParser {
    inner: groggy::core::QueryParser,
}

#[pymethods]
impl PyQueryParser {
    #[new]
    pub fn new() -> Self {
        Self {
            inner: groggy::core::QueryParser::new(),
        }
    }

    pub fn parse_node_query(&mut self, query: &str) -> PyResult<PyNodeFilter> {
        let filter = self.inner.parse_node_query(query)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Parse error: {}", e)))?;
        Ok(PyNodeFilter { inner: filter })
    }

    pub fn parse_edge_query(&mut self, query: &str) -> PyResult<PyEdgeFilter> {
        let filter = self.inner.parse_edge_query(query)
            .map_err(|e| PyErr::new::<PyValueError, _>(format!("Parse error: {}", e)))?;
        Ok(PyEdgeFilter { inner: filter })
    }
}
```

#### **Step 2.2: Update FFI Modules**
```rust
// src/ffi/core/mod.rs  
pub mod query_parser;  // ADD THIS

// src/ffi/mod.rs
pub use core::query_parser::PyQueryParser;  // ADD THIS

// src/lib.rs - add to Python module
m.add_class::<PyQueryParser>()?;  // ADD THIS
```

### **Phase 3: Eliminate Circular Dependencies (Week 1 - Days 6-7)**

#### **Step 3.1: Replace All Python Imports**
**Critical files with `py.import("groggy.query_parser")` to fix:**

1. `/src/ffi/api/graph.rs:853` (find_nodes method)
2. `/src/ffi/api/graph.rs:922` (find_edges method) 
3. `/src/ffi/api/graph_query.rs:103`
4. `/src/ffi/core/subgraph_old_complex.rs:501`
5. `/src/ffi/core/subgraph_old_complex.rs:1137`

**Replace this pattern:**
```rust
// ❌ REMOVE THIS EVERYWHERE
let query_parser = py.import("groggy.query_parser")?;
let parse_func = query_parser.getattr("parse_node_query");
let parsed_filter: PyNodeFilter = parse_func.call1((query_str,))?.extract()?;
```

**With this pattern:**
```rust
// ✅ USE THIS INSTEAD
let mut parser = groggy::core::QueryParser::new();
let filter = parser.parse_node_query(&query_str)
    .map_err(|e| PyErr::new::<PyValueError, _>(format!("Query parse error: {}", e)))?;
```

#### **Step 3.2: Update Graph FFI Methods**
```rust
// src/ffi/api/graph.rs - Updated find_nodes method
impl PyGraph {
    pub fn find_nodes(&mut self, py: Python, filter: &PyAny) -> PyResult<Vec<NodeId>> {
        let node_filter = if let Ok(filter_obj) = filter.extract::<PyNodeFilter>() {
            filter_obj.inner.clone()
        } else if let Ok(query_str) = filter.extract::<String>() {
            // ✅ FIXED: Parse in Rust, no Python import needed
            let mut parser = groggy::core::QueryParser::new();
            parser.parse_node_query(&query_str)
                .map_err(|e| PyErr::new::<PyValueError, _>(format!("Query parse error: {}", e)))?
        } else {
            return Err(PyErr::new::<PyTypeError, _>(
                "filter must be a NodeFilter object or a string query"
            ));
        };

        // Delegate to core Rust implementation (no changes needed here)
        let filtered_nodes = self.inner.borrow_mut()
            .find_nodes(node_filter)
            .map_err(graph_error_to_py_err)?;
            
        Ok(filtered_nodes)
    }
}
```

### **Phase 4: Python Layer Simplification (Week 2 - Days 8-10)**

#### **Step 4.1: Simplify query_parser.py**
**File to modify:**
- `python/groggy/query_parser.py`

**Replace ~400 lines of Python parsing logic with:**
```python
"""Query Parser - Delegates to Rust Core"""
from ._groggy import QueryParser

# Single Rust parser instance
_rust_parser = QueryParser()

def parse_node_query(query: str):
    """Parse node query string - delegates to Rust core."""
    return _rust_parser.parse_node_query(query)

def parse_edge_query(query: str):
    """Parse edge query string - delegates to Rust core."""  
    return _rust_parser.parse_edge_query(query)

# Keep any convenience functions that users rely on
def create_node_filter(attribute, operator, value):
    """Convenience function for creating filters."""
    if operator == "==":
        query = f"{attribute} == '{value}'"
    elif operator in [">", "<", ">=", "<="]:
        query = f"{attribute} {operator} {value}"
    else:
        raise ValueError(f"Unsupported operator: {operator}")
    
    return parse_node_query(query)
```

### **Phase 5: Testing & Validation (Week 2 - Days 11-12)**

#### **Step 5.1: Comprehensive Test Suite**
```rust
#[cfg(test)]
mod tests {
    use super::*;
    
    #[test]
    fn test_simple_comparison() {
        let mut parser = QueryParser::new();
        let filter = parser.parse_node_query("salary > 120000").unwrap();
        assert!(matches!(filter, NodeFilter::Comparison { .. }));
    }
    
    #[test]
    fn test_complex_logical() {
        let mut parser = QueryParser::new();
        let filter = parser.parse_node_query(
            "(salary > 120000 AND department == 'Engineering') OR age < 25"
        ).unwrap();
        assert!(matches!(filter, NodeFilter::Or(_)));
    }
    
    #[test]
    fn test_error_cases() {
        let mut parser = QueryParser::new();
        assert!(parser.parse_node_query("salary >").is_err());
        assert!(parser.parse_node_query("salary > > 100").is_err());
        assert!(parser.parse_node_query("(salary > 100").is_err()); // Missing closing paren
    }
}
```

#### **Step 5.2: Integration Tests**
```python
# Test that Python -> Rust delegation works identically
def test_query_parsing_parity():
    """Ensure new Rust parser produces same results as old Python parser."""
    test_queries = [
        "salary > 120000",
        "department == 'Engineering'",
        "age < 30 AND experience > 5",
        "(salary > 100000 OR bonus > 10000) AND department != 'HR'"
    ]
    
    for query in test_queries:
        rust_result = parse_node_query(query)
        # Validate filter structure and execution results
        assert rust_result is not None
```

### **Phase 6: Cleanup (Week 2 - Days 13-14)**

#### **Step 6.1: Remove Dead Code**
- Delete complex Python parsing logic from `query_parser.py`
- Remove unused imports and helper functions
- Clean up any remaining circular dependency artifacts

#### **Step 6.2: Update Documentation**
- Update architectural diagrams showing clean Rust core
- Document new query parsing flow
- Add performance benchmarks comparing old vs new approach

## 🎯 **Success Metrics**

### **Immediate (After Phase 3)**
- ✅ **Zero circular dependencies** - `cargo check` passes without py.import errors
- ✅ **All 50+ compilation errors resolved** - clean build
- ✅ **FFI boundary cleanly defined** - no Python imports from Rust

### **Long-term (After Phase 6)**  
- ✅ **100% feature parity** - all existing queries work identically
- ✅ **Performance improvement** - faster query parsing in Rust core
- ✅ **Maintainable architecture** - single source of truth
- ✅ **Multi-language ready** - clean foundation for JS/Go/other ports

## ⚠️ **Risk Mitigation**

1. **Feature Parity Risk**: Implement comprehensive cross-validation between old and new parsers
2. **Performance Risk**: Benchmark query parsing performance before/after migration  
3. **API Breaking Risk**: Maintain exact same Python API surface during transition
4. **Regression Risk**: Run full integration test suite after each phase

## 🚀 **Execution Order**

1. **Start with Phase 1** - Core parser implementation
2. **Validate with tests** - Ensure parser works correctly  
3. **Move to Phase 2** - FFI bindings
4. **Execute Phase 3** - **This breaks the circular dependency and fixes compilation**
5. **Complete remaining phases** - Python simplification and cleanup

**Critical Path**: Phases 1-3 will resolve the compilation errors. Phases 4-6 are architectural improvements.

---

**Next Step**: Begin Phase 1 implementation of the core Rust query parser.