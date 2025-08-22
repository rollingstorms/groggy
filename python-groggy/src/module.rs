//! Python Module Registration
//!
//! This module handles registration of all Python classes and functions
//! for the Groggy library.

use pyo3::prelude::*;

/// Register all classes and functions with the Python module
pub fn register_classes(_py: Python, m: &PyModule) -> PyResult<()> {
    // For now, return Ok - we'll add class registrations as we implement them
    // Example:
    // m.add_class::<PyGraph>()?;
    // m.add_class::<PySubgraph>()?;
    // m.add_class::<PyGraphArray>()?;

    Ok(())
}
