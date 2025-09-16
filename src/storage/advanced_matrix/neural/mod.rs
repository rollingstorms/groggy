//! Neural Network Operations Module
//! 
//! This module implements optimized neural network operations including activation functions,
//! convolution operations, automatic differentiation, and memory fusion optimizations.

pub mod activations;
pub mod convolution;
pub mod autodiff;
pub mod fusion;

pub use activations::{ActivationFunction, ActivationOps, relu, gelu, sigmoid, tanh, softmax};
pub use convolution::{Conv2D, ConvolutionConfig, PaddingMode, im2col_transform};
pub use autodiff::{AutoDiffTensor, ComputationGraph, GradientTape, backward_pass};
pub use fusion::{FusionEngine, FusedOperation, FusionPattern, optimize_computation_graph};