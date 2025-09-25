//! Matrix module with neural network capabilities and advanced optimization
//!
//! This module provides the enhanced GraphMatrix with:
//! - Generic NumericType support (f64, f32, i64, i32, bool, etc.)
//! - Neural network operations (matmul, conv2d, activations, autodiff)
//! - Intelligent backend selection (BLAS, NumPy, native)
//! - Memory fusion optimization
//! - Full backward compatibility

pub mod conversions;
pub mod matrix_core;
pub mod slicing;

pub use conversions::*;
pub use matrix_core::*;
pub use slicing::*;
