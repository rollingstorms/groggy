//! Neural Network Operations Module
//!
//! This module implements optimized neural network operations including activation functions,
//! convolution operations, automatic differentiation, and memory fusion optimizations.
//!
//! ## Development Status & Warning Policy
//!
//! **Active Development Area** - Warnings are intentionally kept visible here.
//!
//! Unlike the delegation system (complete but unadopted), neural network code warnings
//! indicate actual work needed:
//! - Unused parameters → Stub implementations that need completion
//! - Unused variables → Incomplete methods that need work
//!
//! Only architectural struct fields are marked with `#[allow(dead_code)]`:
//! - `im2col_buffer`, `gradients`, `data` fields for future features
//! - `ensure_same_graph` utility method
//!
//! **Keep warnings visible** - they prevent silent breakage during development.

pub mod activations;
pub mod autodiff;
pub mod convolution;
pub mod fusion;

pub use activations::{gelu, relu, sigmoid, softmax, tanh, ActivationFunction, ActivationOps};
pub use autodiff::{backward_pass, AutoDiffTensor, ComputationGraph, GradientTape};
pub use convolution::{im2col_transform, Conv2D, ConvolutionConfig, PaddingMode};
pub use fusion::{optimize_computation_graph, FusedOperation, FusionEngine, FusionPattern};
