//! FFI Error Handling
//!
//! This module provides error conversion between Rust and Python.

use groggy::GraphError;
use pyo3::prelude::*;

/// Convert GraphError to Python exception
pub fn graph_error_to_py_err(error: GraphError) -> PyErr {
    // Use existing utility function
    crate::ffi::utils::graph_error_to_py_err(error)
}

// We'll add more error handling utilities as needed
