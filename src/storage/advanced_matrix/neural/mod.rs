//! Neural Network Operations Module
//!
//! This module implements optimized neural network operations including activation functions,
//! convolution operations, automatic differentiation, and memory fusion optimizations.

pub mod activations;
pub mod autodiff;
pub mod convolution;
pub mod fusion;

pub use activations::{gelu, relu, sigmoid, softmax, tanh, ActivationFunction, ActivationOps};
pub use autodiff::{backward_pass, AutoDiffTensor, ComputationGraph, GradientTape};
pub use convolution::{im2col_transform, Conv2D, ConvolutionConfig, PaddingMode};
pub use fusion::{optimize_computation_graph, FusedOperation, FusionEngine, FusionPattern};
