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
pub mod matrix;
pub mod node;
pub mod pool;
pub mod table; // BaseTable system // Advanced Matrix System - Foundation Infrastructure

// Re-export components
pub use adjacency::*;
pub use edge::*;
pub use matrix::*;
pub use node::*;
pub use pool::*;
pub use table::*;

// Re-export advanced_matrix items (canonical benchmark items)
pub use advanced_matrix::{
    AdvancedMemoryPool, BenchmarkConfig, BenchmarkResult, BlasBackend, BlasType, MatrixLayout,
    MatrixBenchmarkSuite, MemoryError, MemoryResult, NumPyBackend, SharedBuffer,
    DType, NumericType, UnifiedMatrix,
    quick_benchmark as matrix_quick_benchmark,
};

// Re-export array items (using different name for conflicting benchmark)
pub use array::{
    ArrayArray, BaseArray, BoolArray, NumArray, StatsSummary,
    AdvancedIndexing, SliceIndex, LazyArrayIterator, QueryEvaluator, BatchQueryEvaluator,
    quick_benchmark as array_quick_benchmark,
    Benchmarker, BenchmarkConfig as ArrayBenchmarkConfig,
};
