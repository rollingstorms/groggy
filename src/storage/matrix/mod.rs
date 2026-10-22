//! Matrix module with neural network capabilities and advanced optimization
//!
//! This module provides the enhanced GraphMatrix with:
//! - Generic NumericType support (f64, f32, i64, i32, bool, etc.)
//! - Neural network operations (matmul, conv2d, activations, autodiff)
//! - Intelligent backend selection (BLAS, NumPy, native)
//! - Memory fusion optimization
//! - Full backward compatibility

mod complex_eigen;
pub mod conversions;
mod decomposition;
mod decomposition_derivatives;
mod derivative_linalg;
pub(crate) mod dot_product;
mod factorization;
mod least_squares;
mod linear_system;
pub mod matrix_core;
mod reductions;
pub mod slicing;
mod sparse_factorization;
mod spectral_derivatives;
mod triangular_derivatives;

pub use conversions::*;
pub use matrix_core::*;
pub use slicing::*;
