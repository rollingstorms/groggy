//! BaseArray Universal Chaining System
//!
//! This module implements a unified array system that enables fluent chaining
//! operations (`.iter()`) on any collection of graph-related objects using
//! trait-based method injection.
//!
//! ## Architecture:
//! - `BaseArray<T>`: Fundamental array operations (len, get, iter)
//! - `NumArray<T>`: Statistical operations layer on top of BaseArray
//! - `ArrayArray<T>`: Array of arrays with aggregation support
//! - Specialized arrays delegate to appropriate base classes

pub mod array_array;
pub mod base;
pub mod base_array;
pub mod benchmark;
pub mod bool_array;
pub mod indexing;
pub mod iterator;
pub mod lazy_iterator;
pub mod memory_profiler;
pub mod num_array;
pub mod numarray_benchmark;
pub mod query;
pub mod simd_optimizations;
pub mod specialized;
pub mod traits;

pub use array_array::ArrayArray;
pub use base_array::BaseArray;
pub use benchmark::{quick_benchmark, BenchmarkConfig, Benchmarker};
pub use bool_array::BoolArray;
pub use indexing::{AdvancedIndexing, SliceIndex};
pub use iterator::*;
pub use lazy_iterator::LazyArrayIterator;
pub use memory_profiler::{
    detailed_memory_analysis, quick_memory_analysis, MemorySnapshot, NumArrayMemoryProfiler,
};
pub use num_array::{NumArray, StatsSummary};
pub use numarray_benchmark::{
    quick_numarray_benchmark, NumArrayBenchmarkConfig, NumArrayBenchmarker,
};
pub use query::{BatchQueryEvaluator, QueryEvaluator};
pub use specialized::*;
pub use traits::*;
