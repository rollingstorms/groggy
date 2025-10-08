//! FFI Error Handling
//!
//! This module provides error conversion between Rust and Python.

use groggy::errors::GraphError;
use pyo3::prelude::*;

/// Python wrapper for GraphError
#[derive(Debug)]
pub struct PyGraphError(pub GraphError);

impl From<GraphError> for PyGraphError {
    fn from(err: GraphError) -> Self {
        PyGraphError(err)
    }
}

impl From<PyGraphError> for PyErr {
    fn from(err: PyGraphError) -> Self {
        match err.0 {
            GraphError::NodeNotFound { .. }
            | GraphError::EdgeNotFound { .. }
            | GraphError::StateNotFound { .. }
            | GraphError::BranchNotFound { .. }
            | GraphError::AttributeNotFound { .. } => {
                pyo3::exceptions::PyKeyError::new_err(format!("{:?}", err.0))
            }
            GraphError::AttributeTypeMismatch { .. }
            | GraphError::InvalidAttributeName { .. }
            | GraphError::InvalidQuery { .. }
            | GraphError::QueryParseError { .. }
            | GraphError::InvalidConfiguration { .. } => {
                pyo3::exceptions::PyValueError::new_err(format!("{:?}", err.0))
            }
            GraphError::InternalError { .. }
            | GraphError::UnexpectedState { .. }
            | GraphError::CorruptedHistory { .. } => {
                pyo3::exceptions::PyRuntimeError::new_err(format!("{:?}", err.0))
            }
            _ => pyo3::exceptions::PyRuntimeError::new_err(format!("Graph error: {:?}", err.0)),
        }
    }
}
