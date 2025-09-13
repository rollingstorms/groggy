# NumArray API Compatibility Baseline Documentation

## Purpose
This document establishes the compatibility baseline for the NumArray API as part of Phase 2.3: NumArray Performance Optimization. It serves as a reference to ensure backward compatibility is maintained during optimization work.

## NumArray Core API (`src/storage/array/num_array.rs`)

### Creation and Basic Operations

```rust
// Construction
NumArray::new(data: Vec<f64>) -> NumArray
NumArray::from_slice(slice: &[f64]) -> NumArray

// Basic properties
.len() -> usize
.is_empty() -> bool
.get(index: usize) -> Option<&f64>
.iter() -> std::slice::Iter<f64>
```

### Statistical Operations

```rust
// Central tendencies
.sum() -> f64
.mean() -> Option<f64>
.median() -> Option<f64>

// Spread and variance
.std_dev() -> Option<f64>
.variance() -> Option<f64>

// Extremes
.min() -> Option<f64>
.max() -> Option<f64>
.min_max() -> Option<(f64, f64)>
```

### Array Operations

```rust
// Iteration and access
.iter() -> std::slice::Iter<f64>
.clone() -> NumArray

// Element access
.get(index: usize) -> Option<&f64>
```

## Performance Characteristics (Baseline)

### Benchmark Results from Phase 2.3 Testing

Based on comprehensive benchmarking performed during Phase 2.3 implementation:

#### Small Arrays (1,000 elements)
- **Creation**: 364ns (2.7B elem/sec)
- **Sum**: 13.69µs (73M elem/sec) 
- **Mean**: 10.64µs (94M elem/sec)
- **Std Dev**: 17.28µs (58M elem/sec)
- **Min/Max**: ~12-14µs (70-82M elem/sec)
- **Median**: 305.57µs (3.3M elem/sec)
- **Clone**: 41ns (24B elem/sec)
- **Iteration**: 11.75µs (85M elem/sec)

#### Medium Arrays (10,000 elements)
- **Creation**: 1.60µs (6.2B elem/sec)
- **Sum**: 80.76µs (124M elem/sec)
- **Mean**: 86.29µs (116M elem/sec) 
- **Std Dev**: 170.47µs (59M elem/sec)
- **Min/Max**: ~146-182µs (55-68M elem/sec)
- **Median**: 3.46ms (2.9M elem/sec)
- **Clone**: 35ns (286B elem/sec)
- **Iteration**: 38.68µs (259M elem/sec)

#### Memory Usage Baseline
- **Memory per element**: 8 bytes (f64)
- **Memory overhead**: Minimal (<1% for data structures)
- **Peak allocation**: Linear with array size

## API Compatibility Requirements

### 1. Method Signatures
All existing method signatures MUST remain unchanged:
- Parameter types and order
- Return types 
- Mutability requirements
- Generic constraints

### 2. Behavioral Contracts
- Statistical operations return `Option<f64>` for empty arrays
- `sum()` returns 0.0 for empty arrays (special case)
- Iteration order is guaranteed to be insertion order
- Memory safety guarantees maintained

### 3. Performance Guarantees
Optimizations MUST NOT regress performance beyond these baseline values:
- **Creation**: Must remain O(n) time complexity
- **Statistical ops**: Must remain O(n) time complexity
- **Element access**: Must remain O(1) time complexity
- **Memory usage**: Must remain O(n) space complexity

### 4. Error Handling
- No panics on valid inputs
- Graceful handling of edge cases (empty arrays, NaN values)
- Consistent `Option` returns for operations that may fail

## Testing Contract

### Unit Test Coverage
All existing unit tests MUST continue to pass:
- `tests/num_array_tests.rs` (if exists)
- Inline doc tests
- Integration tests

### Benchmark Regression Testing
Performance regressions beyond 10% of baseline values require:
1. Technical justification
2. Compensating improvements in other areas
3. Documentation update

## Memory Usage Baseline

### Allocation Patterns
- **Array creation**: Single allocation for data vector
- **Statistical operations**: Minimal temporary allocations
- **Cloning**: Single allocation of same size as original

### Memory Efficiency Scores
- **Creation**: 1.0 (theoretical maximum efficiency)
- **Statistical ops**: 0.99+ (minimal overhead)
- **Element access**: 1.0 (no additional allocations)

## Integration Points

### Module Integration
```rust
// Export requirements that must be maintained
pub use num_array::{NumArray, StatsSummary};
```

### FFI Compatibility
When Python bindings are restored:
- All public methods must remain accessible
- Type conversions must be preserved
- Error handling patterns must be consistent

## Optimization Guidelines

### Acceptable Changes
- Internal implementation optimizations
- SIMD instruction usage
- Memory layout improvements
- Algorithmic improvements maintaining O(n) complexity

### Prohibited Changes
- Breaking API changes
- Performance regressions >10%
- Memory usage increases >50%
- Changed error handling behavior

## Validation Process

### Pre-Optimization Checklist
1. ✅ Comprehensive benchmark suite implemented
2. ✅ Memory profiling infrastructure in place
3. ✅ API baseline documented
4. ⏳ Continuous benchmarking pipeline (Phase 2.3 Task 4)

### Post-Optimization Validation
1. Run full benchmark suite
2. Compare against baseline metrics
3. Verify all existing tests pass
4. Confirm memory usage patterns
5. Validate API compatibility

## Version Information

- **Baseline Version**: groggy v0.3.1
- **Benchmark Suite**: `src/storage/array/numarray_benchmark.rs`
- **Memory Profiler**: `src/storage/array/memory_profiler.rs`
- **Documentation Date**: 2025-09-13
- **Phase**: 2.3 - NumArray Performance Optimization

## Notes for Future Optimization Work

### Identified Optimization Opportunities
1. **Median calculation**: Currently slowest operation (3.46ms for 10K elements)
2. **SIMD potential**: Statistical operations could benefit from vectorization
3. **Memory access patterns**: Consider cache-friendly algorithms

### Baseline Comparisons
- Some operations show 1.62x slower than naive implementations
- Opportunity for algorithmic improvements while maintaining API

This baseline establishes the foundation for safe, compatible optimization work in subsequent phases of the NumArray performance enhancement project.