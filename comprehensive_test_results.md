# Groggy Comprehensive Test Suite Report

Generated: 2025-08-29 18:19:52

## Executive Summary

This comprehensive test suite goes beyond basic validation to test edge cases, 
performance characteristics, persona-based scenarios, and complex composability patterns.

### Test Results
- **Total Tests**: 135
- **Passed**: 126 ✅  
- **Failed**: 9 ❌
- **Success Rate**: 93.3%

### Performance Metrics
- **Average Execution Time**: 0.0001s
- **Slowest Test**: 0.0015s
- **Tests > 100ms**: 0

## Results by Section

### ✅ Comprehensive Imports
- **Success Rate**: 11/11 (100.0%)
- **Total Time**: 0.0000s

### ✅ Edge Cases - Graph Creation
- **Success Rate**: 6/6 (100.0%)
- **Total Time**: 0.0011s

### ⚠️ Boundary Conditions - Nodes
- **Success Rate**: 35/36 (97.2%)
- **Total Time**: 0.0011s
- **Failed Tests**:
  - `Node with nested dict`: Unsupported attribute value type: dict. Supported types: int, float, str, bool, bytes, [int], [float], [str]

### ⚠️ Bulk Operations - Core Architecture
- **Success Rate**: 3/5 (60.0%)
- **Total Time**: 0.0000s
- **Failed Tests**:
  - `set_node_attrs - bulk attribute setting`: 'str' object cannot be interpreted as an integer
  - `Bulk edge operations`: 'str' object cannot be interpreted as an integer

### ⚠️ Attribute System - Edge Cases
- **Success Rate**: 21/22 (95.5%)
- **Total Time**: 0.0000s
- **Failed Tests**:
  - `Set/get nested structure`: Unsupported attribute value type: dict. Supported types: int, float, str, bool, bytes, [int], [float], [str]

### ⚠️ Query System - Comprehensive
- **Success Rate**: 35/40 (87.5%)
- **Total Time**: 0.0005s
- **Failed Tests**:
  - `Query Apply: Simple greater than`: "Attribute 'age' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."
  - `Query Apply: Simple less than`: "Attribute 'salary' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."
  - `Query Apply: Greater than or equal`: "Attribute 'age' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."
  - `Query Apply: Less than or equal`: "Attribute 'salary' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."
  - `Query Apply: Non-existent attribute`: "Attribute 'nonexistent' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."

### ✅ Performance Characteristics
- **Success Rate**: 15/15 (100.0%)
- **Total Time**: 0.0037s

## Failed Tests Analysis

### 1. Node with nested dict (Boundary Conditions - Nodes)

**Error**: `Unsupported attribute value type: dict. Supported types: int, float, str, bool, bytes, [int], [float], [str]`

---

### 2. set_node_attrs - bulk attribute setting (Bulk Operations - Core Architecture)

**Error**: `'str' object cannot be interpreted as an integer`

---

### 3. Bulk edge operations (Bulk Operations - Core Architecture)

**Error**: `'str' object cannot be interpreted as an integer`

---

### 4. Set/get nested structure (Attribute System - Edge Cases)

**Error**: `Unsupported attribute value type: dict. Supported types: int, float, str, bool, bytes, [int], [float], [str]`

---

### 5. Query Apply: Simple greater than (Query System - Comprehensive)

**Error**: `"Attribute 'age' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."`

---

### 6. Query Apply: Simple less than (Query System - Comprehensive)

**Error**: `"Attribute 'salary' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."`

---

### 7. Query Apply: Greater than or equal (Query System - Comprehensive)

**Error**: `"Attribute 'age' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."`

---

### 8. Query Apply: Less than or equal (Query System - Comprehensive)

**Error**: `"Attribute 'salary' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."`

---

### 9. Query Apply: Non-existent attribute (Query System - Comprehensive)

**Error**: `"Attribute 'nonexistent' does not exist on any nodes in the graph. Use graph.nodes.table().columns to see available attributes."`

---

## Performance Analysis

## Recommendations

### Critical Issues (9 failed tests)
1. **Review failed test cases** - may indicate bugs or missing features
2. **Check edge case handling** - several boundary condition tests failed
3. **Verify error handling** - ensure graceful degradation

## Test Categories Covered

1. **Basic Functionality**: Core graph operations, CRUD operations
2. **Edge Cases**: Boundary conditions, invalid inputs, empty states
3. **Performance**: Large graphs, bulk operations, memory usage  
4. **Query System**: Complex queries, malformed inputs, edge cases
5. **Composability**: Method chaining, feature integration
6. **Persona Testing**: Real-world scenarios from different user perspectives
7. **Cross-Feature Integration**: How different modules work together

---
*Generated by Groggy Comprehensive Test Suite*
*"Every method, every edge case, every weird vibe" - YN*
