//! Layer-specific activation, convolution and fusion APIs.
//! Generic matrix autodiff is implemented in matrix::autodiff; the historical
//! neural::autodiff path remains a compatibility alias.

pub mod activations;
// Compatibility path; automatic differentiation belongs to the matrix engine.
pub use super::autodiff;
pub mod convolution;
pub mod fusion;

pub use activations::{gelu, relu, sigmoid, softmax, tanh, ActivationFunction, ActivationOps};
pub use autodiff::{backward_pass, AutoDiffTensor, ComputationGraph, GradientTape};
pub use convolution::{im2col_transform, Conv2D, ConvolutionConfig, PaddingMode};
pub use fusion::{optimize_computation_graph, FusedOperation, FusionEngine, FusionPattern};
