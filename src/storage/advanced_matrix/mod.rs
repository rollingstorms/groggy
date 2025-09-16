//! Advanced Matrix System - Foundation Infrastructure
//! 
//! This module implements the comprehensive matrix and neural network infrastructure
//! designed to provide world-class linear algebra performance for Groggy.
//! 
//! This is the foundation infrastructure for Weeks 3-4 of the implementation plan.

pub mod numeric_type;
pub mod unified_matrix;
pub mod backend;
pub mod backends;
pub mod memory;
pub mod operations;
pub mod benchmarks;
pub mod neural;

pub use numeric_type::{NumericType, DType};
pub use unified_matrix::{UnifiedMatrix, Matrix64, Matrix32, MatrixI64, MatrixI32, MatrixError, MatrixResult, Shape, Strides};
pub use backend::{ComputeBackend, BackendHint, BackendSelector, OperationType, BackendError, BackendResult};
pub use backends::{NumPyBackend, BlasBackend, BlasType};
pub use memory::{SharedBuffer, AdvancedMemoryPool, MatrixLayout, MemoryError, MemoryResult};
pub use operations::{MatrixOperations, ActivationFunctions, MathFunctions, MatrixUtils, MatrixStats};
pub use benchmarks::{MatrixBenchmarkSuite, BenchmarkConfig, BenchmarkResult, quick_benchmark};
pub use neural::{
    ActivationFunction, ActivationOps, relu, gelu, sigmoid, tanh, softmax,
    Conv2D, ConvolutionConfig, PaddingMode, im2col_transform,
    AutoDiffTensor, ComputationGraph, GradientTape, backward_pass,
    FusionEngine, FusedOperation, FusionPattern, optimize_computation_graph
};