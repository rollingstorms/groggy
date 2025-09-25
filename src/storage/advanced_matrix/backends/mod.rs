//! Backend implementations for the advanced matrix system
//!
//! This module contains various compute backend implementations that provide
//! optimal performance for different types of operations and hardware.

pub mod blas;
pub mod numpy;

pub use blas::{BlasBackend, BlasType};
pub use numpy::NumPyBackend;

/// Re-export for convenience
pub use crate::storage::advanced_matrix::backend::{
    BackendError, BackendHint, BackendPerformance, BackendResult, BackendSelector, ComputeBackend,
    ComputeBackendExt, OperationType,
};
