//! BaseArray Universal Chaining System
//! 
//! This module implements a unified array system that enables fluent chaining
//! operations (`.iter()`) on any collection of graph-related objects using
//! trait-based method injection.
//!
//! ## Architecture:
//! - `BaseArray<T>`: Fundamental array operations (len, get, iter)
//! - `NumArray<T>`: Statistical operations layer on top of BaseArray
//! - Specialized arrays delegate to appropriate base classes

pub mod traits;
pub mod iterator;
pub mod lazy_iterator;
pub mod benchmark;
pub mod query;
pub mod base;
pub mod specialized;
pub mod base_array;
pub mod num_array;

pub use traits::*;
pub use iterator::*;
pub use lazy_iterator::LazyArrayIterator;
pub use benchmark::{Benchmarker, BenchmarkConfig, quick_benchmark};
pub use query::{QueryEvaluator, BatchQueryEvaluator};
pub use base::*;
pub use specialized::*;
pub use base_array::BaseArray;
pub use num_array::{NumArray, StatsSummary};