//! Neural Network FFI Module
//!
//! Python bindings for neural network operations and activation functions.
//! Provides functional interface: import groggy.neural as nn

pub mod activations;

use pyo3::prelude::*;

/// Neural network module for functional operations
#[pymodule]
pub fn neural(_py: Python, m: &PyModule) -> PyResult<()> {
    // Activation functions
    m.add_function(wrap_pyfunction!(activations::relu, m)?)?;
    m.add_function(wrap_pyfunction!(activations::sigmoid, m)?)?;
    m.add_function(wrap_pyfunction!(activations::tanh, m)?)?;
    m.add_function(wrap_pyfunction!(activations::softmax, m)?)?;
    m.add_function(wrap_pyfunction!(activations::gelu, m)?)?;

    Ok(())
}

/// Create neural submodule for registration
/// Future feature: neural network submodule
#[allow(dead_code)]
pub fn create_neural_submodule(py: Python<'_>) -> PyResult<&PyModule> {
    PyModule::new(py, "neural")
}
