//! Advanced Matrix System - Foundation Infrastructure
//!
//! This module implements the comprehensive matrix and neural network infrastructure
//! designed to provide world-class linear algebra performance for Groggy.
//!
//! This is the foundation infrastructure for Weeks 3-4 of the implementation plan.

pub mod backend;
pub mod backends;
pub mod benchmarks;
pub mod memory;
pub mod neural;
pub mod numeric_type;
pub mod operations;
pub mod unified_matrix;

pub use backend::{
    BackendError, BackendHint, BackendResult, BackendSelector, ComputeBackend, OperationType,
};
pub use backends::{BlasBackend, BlasType, NumPyBackend};
pub use benchmarks::{quick_benchmark, BenchmarkConfig, BenchmarkResult, MatrixBenchmarkSuite};
pub use memory::{AdvancedMemoryPool, MatrixLayout, MemoryError, MemoryResult, SharedBuffer};
pub use neural::{
    backward_pass, gelu, im2col_transform, optimize_computation_graph, relu, sigmoid, softmax,
    tanh, ActivationFunction, ActivationOps, AutoDiffTensor, ComputationGraph, Conv2D,
    ConvolutionConfig, FusedOperation, FusionEngine, FusionPattern, GradientTape, PaddingMode,
};
pub use numeric_type::{DType, NumericType};
pub use operations::{
    ActivationFunctions, MathFunctions, MatrixOperations, MatrixStats, MatrixUtils,
};
pub use unified_matrix::{
    Matrix32, Matrix64, MatrixError, MatrixI32, MatrixI64, MatrixResult, Shape, Strides,
    UnifiedMatrix,
};
