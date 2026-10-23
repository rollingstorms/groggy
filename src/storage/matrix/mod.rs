//! Matrix engine: public GraphMatrix API, dense/CSR storage, checked arithmetic,
//! linear algebra, graph operators and automatic differentiation.
//!
//! GraphMatrix owns API metadata/tapes; UnifiedMatrix owns numerical storage.
//! NumericalMatrixData transfers owned numerical snapshots across threads.
//! Optional backend capability is reported explicitly; exporting a backend type
//! does not imply an external accelerator is available.

pub mod adjacency;
pub mod autodiff;
pub mod backend;
pub mod backends;
pub mod benchmarks;
pub(crate) mod checked_kernels;
mod csr;
pub mod memory;
pub mod neural;
pub mod numeric_type;
pub mod operations;
mod sparse_ops;
pub mod unified_matrix;

pub use autodiff::{backward_pass, AutoDiffTensor, ComputationGraph, GradientTape};
pub use backend::{
    BackendError, BackendHint, BackendResult, BackendSelector, ComputeBackend, OperationType,
};
pub use backends::{BlasBackend, BlasType, NumPyBackend};
pub use benchmarks::{quick_benchmark, BenchmarkConfig, BenchmarkResult, MatrixBenchmarkSuite};
pub use memory::{AdvancedMemoryPool, MatrixLayout, MemoryError, MemoryResult, SharedBuffer};
pub use neural::{
    gelu, im2col_transform, optimize_computation_graph, relu, sigmoid, softmax, tanh,
    ActivationFunction, ActivationOps, Conv2D, ConvolutionConfig, FusedOperation, FusionEngine,
    FusionPattern, PaddingMode,
};
pub use numeric_type::{DType, NumericType};
pub use operations::{
    ActivationFunctions, MathFunctions, MatrixOperations, MatrixStats, MatrixUtils,
};
pub use unified_matrix::{
    Matrix32, Matrix64, MatrixError, MatrixI32, MatrixI64, MatrixResult, Shape, Strides,
    UnifiedMatrix,
};

mod complex_eigen;
pub mod conversions;
mod decomposition;
mod decomposition_derivatives;
mod derivative_linalg;
pub(crate) mod dot_product;
mod factorization;
pub(crate) mod graph_projection;
mod least_squares;
mod linear_system;
pub mod matrix_core;
pub(crate) mod matrix_power;
mod reductions;
mod resource_policy;
pub use resource_policy::IMPLICIT_DENSE_ELEMENT_LIMIT;
pub mod slicing;
mod sparse_factorization;
mod spectral_derivatives;
mod triangular_derivatives;

pub use conversions::*;
pub use matrix_core::*;
pub use slicing::*;
