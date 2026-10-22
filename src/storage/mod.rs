//! Storage and view types
//!
//! This module contains data storage structures and views:
//! - Matrix storage
//! - Table storage
//! - Array storage
//! - Adjacency storage
//! - Memory pools
//! - Node and edge storage

pub mod adjacency;
pub mod advanced_matrix;
pub mod array; // BaseArray system
pub mod edge;
pub mod journal;
pub mod matrix;
pub mod node;
pub mod numeric;
pub mod pool;
pub mod table; // BaseTable system
pub(crate) mod value_codec;

// Re-export components
pub use adjacency::*;
pub use edge::*;
pub use matrix::*;
pub use node::*;
pub use pool::*;
pub use table::*;

// Re-export canonical numerical engine items; legacy imports remain aliases.
pub use matrix::{
    quick_benchmark as matrix_quick_benchmark, AdvancedMemoryPool, BenchmarkConfig,
    BenchmarkResult, BlasBackend, BlasType, DType, MatrixBenchmarkSuite, MatrixLayout, MemoryError,
    MemoryResult, NumPyBackend, NumericType, SharedBuffer, UnifiedMatrix,
};

// Re-export array items (using different name for conflicting benchmark)
pub use array::{
    quick_benchmark as array_quick_benchmark, AdvancedIndexing, ArrayArray, BaseArray,
    BatchQueryEvaluator, BenchmarkConfig as ArrayBenchmarkConfig, Benchmarker, BoolArray,
    LazyArrayIterator, NumArray, QueryEvaluator, SliceIndex, StatsSummary,
};
