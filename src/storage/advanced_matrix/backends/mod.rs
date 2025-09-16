//! Backend implementations for the advanced matrix system
//! 
//! This module contains various compute backend implementations that provide
//! optimal performance for different types of operations and hardware.

pub mod numpy;
pub mod blas;

pub use numpy::NumPyBackend;
pub use blas::{BlasBackend, BlasType};

/// Re-export for convenience
pub use crate::storage::advanced_matrix::backend::{
    ComputeBackend, ComputeBackendExt, BackendHint, BackendSelector, 
    OperationType, BackendPerformance, BackendError, BackendResult
};