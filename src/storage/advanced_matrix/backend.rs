//! ComputeBackend trait and implementations
//!
//! This module implements the multi-backend architecture that allows the matrix system
//! to intelligently select the optimal compute backend based on operation type,
//! matrix size, and available hardware.

#![allow(clippy::borrowed_box)]
#![allow(clippy::missing_safety_doc)]

use crate::storage::advanced_matrix::numeric_type::{DType, NumericType};
use std::collections::HashMap;
use std::fmt::Debug;

/// Errors specific to backend operations
#[derive(Debug, Clone)]
pub enum BackendError {
    UnsupportedOperation(String),
    UnsupportedType(DType),
    IncompatibleDimensions {
        expected: (usize, usize),
        got: (usize, usize),
    },
    BackendUnavailable(String),
    ComputationFailed(String),
    MemoryAllocationFailed,
    InvalidInput(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::UnsupportedOperation(op) => write!(f, "Unsupported operation: {}", op),
            BackendError::UnsupportedType(dtype) => write!(f, "Unsupported data type: {:?}", dtype),
            BackendError::IncompatibleDimensions { expected, got } => {
                write!(
                    f,
                    "Incompatible dimensions: expected {:?}, got {:?}",
                    expected, got
                )
            }
            BackendError::BackendUnavailable(name) => write!(f, "Backend unavailable: {}", name),
            BackendError::ComputationFailed(msg) => write!(f, "Computation failed: {}", msg),
            BackendError::MemoryAllocationFailed => write!(f, "Memory allocation failed"),
            BackendError::InvalidInput(msg) => write!(f, "Invalid input: {}", msg),
        }
    }
}

impl std::error::Error for BackendError {}

/// Result type for backend operations
pub type BackendResult<T> = Result<T, BackendError>;

/// Identifies the type of computation operation
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum OperationType {
    // Basic linear algebra
    GEMM, // General Matrix Multiply
    GEMV, // General Matrix-Vector multiply
    ElementwiseAdd,
    ElementwiseSub,
    ElementwiseMul,
    ElementwiseDiv,

    // Reductions
    Sum,
    Mean,
    Max,
    Min,

    // Advanced operations
    SVD,        // Singular Value Decomposition
    QR,         // QR Decomposition
    Cholesky,   // Cholesky Decomposition
    LU,         // LU Decomposition
    Eigenvalue, // Eigenvalue decomposition

    // Sparse operations
    SparseMV,   // Sparse Matrix-Vector multiply
    SparseGEMM, // Sparse General Matrix Multiply

    // Neural network operations
    Conv2D,     // 2D Convolution
    BatchGEMM,  // Batched matrix multiply
    Activation, // Activation functions
}

/// Performance characteristics of a backend for different operations
#[derive(Debug, Clone)]
pub struct BackendPerformance {
    pub throughput_ops_per_sec: f64,
    pub memory_bandwidth_gb_per_sec: f64,
    pub setup_overhead_us: f64,
    pub optimal_size_range: (usize, usize),
}

/// Hints for backend selection
#[derive(Debug, Clone, PartialEq)]
pub enum BackendHint {
    PreferSpeed,    // Optimize for computation speed
    PreferMemory,   // Optimize for memory usage
    PreferAccuracy, // Optimize for numerical accuracy
    PreferGPU,      // Prefer GPU if available
    PreferCPU,      // Prefer CPU backends
    AutoSelect,     // Let the system decide
}

/// Core trait that all compute backends must implement
///
/// Note: We use type erasure with raw pointers and DType to make this trait object-safe
pub trait ComputeBackend: Send + Sync + Debug {
    /// Backend identifier
    fn name(&self) -> &str;

    /// Check if this backend is available in the current environment
    fn is_available(&self) -> bool;

    /// Check if backend supports the given data type
    fn supports_dtype(&self, dtype: DType) -> bool;

    /// Check if backend supports the given operation
    fn supports_operation(&self, op: OperationType) -> bool;

    /// Get the optimal threshold size for using this backend
    fn optimal_threshold(&self, op: OperationType) -> usize;

    /// Get performance characteristics for this backend
    fn performance_characteristics(&self, op: OperationType) -> BackendPerformance;

    /// Type-erased matrix multiplication
    /// Safety: Caller must ensure pointers are valid and data types match dtype
    unsafe fn gemm_raw(
        &self,
        dtype: DType,
        a: *const u8,
        a_shape: (usize, usize),
        b: *const u8,
        b_shape: (usize, usize),
        c: *mut u8,
        c_shape: (usize, usize),
        alpha: *const u8,
        beta: *const u8,
    ) -> BackendResult<()>;

    /// Type-erased matrix-vector multiplication
    unsafe fn gemv_raw(
        &self,
        dtype: DType,
        matrix: *const u8,
        matrix_shape: (usize, usize),
        vector: *const u8,
        result: *mut u8,
        alpha: *const u8,
        beta: *const u8,
    ) -> BackendResult<()>;

    /// Type-erased element-wise operations
    unsafe fn elementwise_add_raw(
        &self,
        dtype: DType,
        a: *const u8,
        b: *const u8,
        result: *mut u8,
        len: usize,
    ) -> BackendResult<()>;

    unsafe fn elementwise_mul_raw(
        &self,
        dtype: DType,
        a: *const u8,
        b: *const u8,
        result: *mut u8,
        len: usize,
    ) -> BackendResult<()>;

    /// Type-erased reduction operations
    unsafe fn reduce_sum_raw(
        &self,
        dtype: DType,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()>;
    unsafe fn reduce_max_raw(
        &self,
        dtype: DType,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()>;
    unsafe fn reduce_min_raw(
        &self,
        dtype: DType,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()>;
}

/// Extension trait with safe generic methods
pub trait ComputeBackendExt: ComputeBackend {
    /// Safe wrapper for GEMM
    fn gemm<T: NumericType>(
        &self,
        a: &[T],
        a_shape: (usize, usize),
        b: &[T],
        b_shape: (usize, usize),
        c: &mut [T],
        c_shape: (usize, usize),
        alpha: T,
        beta: T,
    ) -> BackendResult<()> {
        unsafe {
            self.gemm_raw(
                T::DTYPE,
                a.as_ptr() as *const u8,
                a_shape,
                b.as_ptr() as *const u8,
                b_shape,
                c.as_mut_ptr() as *mut u8,
                c_shape,
                &alpha as *const T as *const u8,
                &beta as *const T as *const u8,
            )
        }
    }

    /// Safe wrapper for element-wise addition
    fn elementwise_add<T: NumericType>(
        &self,
        a: &[T],
        b: &[T],
        result: &mut [T],
    ) -> BackendResult<()> {
        unsafe {
            self.elementwise_add_raw(
                T::DTYPE,
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                result.as_mut_ptr() as *mut u8,
                a.len(),
            )
        }
    }

    /// Safe wrapper for element-wise multiplication
    fn elementwise_mul<T: NumericType>(
        &self,
        a: &[T],
        b: &[T],
        result: &mut [T],
    ) -> BackendResult<()> {
        unsafe {
            self.elementwise_mul_raw(
                T::DTYPE,
                a.as_ptr() as *const u8,
                b.as_ptr() as *const u8,
                result.as_mut_ptr() as *mut u8,
                a.len(),
            )
        }
    }

    /// Safe wrapper for sum reduction
    fn reduce_sum<T: NumericType>(&self, data: &[T]) -> BackendResult<T::Accumulator> {
        let mut result = T::Accumulator::zero();
        unsafe {
            self.reduce_sum_raw(
                T::DTYPE,
                data.as_ptr() as *const u8,
                data.len(),
                &mut result as *mut T::Accumulator as *mut u8,
            )?;
        }
        Ok(result)
    }

    /// Safe wrapper for max reduction
    fn reduce_max<T: NumericType>(&self, data: &[T]) -> BackendResult<T> {
        if data.is_empty() {
            return Err(BackendError::InvalidInput(
                "Empty data for reduction".to_string(),
            ));
        }
        let mut result = data[0];
        unsafe {
            self.reduce_max_raw(
                T::DTYPE,
                data.as_ptr() as *const u8,
                data.len(),
                &mut result as *mut T as *mut u8,
            )?;
        }
        Ok(result)
    }

    /// Safe wrapper for min reduction
    fn reduce_min<T: NumericType>(&self, data: &[T]) -> BackendResult<T> {
        if data.is_empty() {
            return Err(BackendError::InvalidInput(
                "Empty data for reduction".to_string(),
            ));
        }
        let mut result = data[0];
        unsafe {
            self.reduce_min_raw(
                T::DTYPE,
                data.as_ptr() as *const u8,
                data.len(),
                &mut result as *mut T as *mut u8,
            )?;
        }
        Ok(result)
    }
}

// Blanket implementation
impl<T: ComputeBackend + ?Sized> ComputeBackendExt for T {}

/// Result of SVD decomposition
#[derive(Debug, Clone)]
pub struct SVDResult<T: NumericType> {
    pub u: Vec<T>,
    pub s: Vec<T>,
    pub vt: Vec<T>,
    pub u_shape: (usize, usize),
    pub vt_shape: (usize, usize),
}

/// Result of QR decomposition
#[derive(Debug, Clone)]
pub struct QRResult<T: NumericType> {
    pub q: Vec<T>,
    pub r: Vec<T>,
    pub q_shape: (usize, usize),
    pub r_shape: (usize, usize),
}

/// Native Rust backend - always available fallback
#[derive(Debug)]
pub struct NativeBackend;

impl ComputeBackend for NativeBackend {
    fn name(&self) -> &str {
        "Native"
    }

    fn is_available(&self) -> bool {
        true
    }

    fn supports_dtype(&self, _dtype: DType) -> bool {
        true
    }

    fn supports_operation(&self, op: OperationType) -> bool {
        matches!(
            op,
            OperationType::GEMM
                | OperationType::GEMV
                | OperationType::ElementwiseAdd
                | OperationType::ElementwiseMul
                | OperationType::Sum
                | OperationType::Max
                | OperationType::Min
        )
    }

    fn optimal_threshold(&self, _op: OperationType) -> usize {
        0
    }

    fn performance_characteristics(&self, _op: OperationType) -> BackendPerformance {
        BackendPerformance {
            throughput_ops_per_sec: 1e6, // Modest performance
            memory_bandwidth_gb_per_sec: 10.0,
            setup_overhead_us: 1.0,        // Very low overhead
            optimal_size_range: (0, 1000), // Good for small matrices
        }
    }

    unsafe fn gemm_raw(
        &self,
        dtype: DType,
        a: *const u8,
        a_shape: (usize, usize),
        b: *const u8,
        b_shape: (usize, usize),
        c: *mut u8,
        c_shape: (usize, usize),
        alpha: *const u8,
        beta: *const u8,
    ) -> BackendResult<()> {
        match dtype {
            DType::Float64 => {
                self.gemm_typed::<f64>(a, a_shape, b, b_shape, c, c_shape, alpha, beta)
            }
            DType::Float32 => {
                self.gemm_typed::<f32>(a, a_shape, b, b_shape, c, c_shape, alpha, beta)
            }
            DType::Int64 => self.gemm_typed::<i64>(a, a_shape, b, b_shape, c, c_shape, alpha, beta),
            DType::Int32 => self.gemm_typed::<i32>(a, a_shape, b, b_shape, c, c_shape, alpha, beta),
            DType::Bool => self.gemm_typed::<bool>(a, a_shape, b, b_shape, c, c_shape, alpha, beta),
            _ => Err(BackendError::UnsupportedType(dtype)),
        }
    }

    unsafe fn gemv_raw(
        &self,
        dtype: DType,
        matrix: *const u8,
        matrix_shape: (usize, usize),
        vector: *const u8,
        result: *mut u8,
        alpha: *const u8,
        beta: *const u8,
    ) -> BackendResult<()> {
        match dtype {
            DType::Float64 => {
                self.gemv_typed::<f64>(matrix, matrix_shape, vector, result, alpha, beta)
            }
            DType::Float32 => {
                self.gemv_typed::<f32>(matrix, matrix_shape, vector, result, alpha, beta)
            }
            DType::Int64 => {
                self.gemv_typed::<i64>(matrix, matrix_shape, vector, result, alpha, beta)
            }
            DType::Int32 => {
                self.gemv_typed::<i32>(matrix, matrix_shape, vector, result, alpha, beta)
            }
            DType::Bool => {
                self.gemv_typed::<bool>(matrix, matrix_shape, vector, result, alpha, beta)
            }
            _ => Err(BackendError::UnsupportedType(dtype)),
        }
    }

    unsafe fn elementwise_add_raw(
        &self,
        dtype: DType,
        a: *const u8,
        b: *const u8,
        result: *mut u8,
        len: usize,
    ) -> BackendResult<()> {
        match dtype {
            DType::Float64 => self.elementwise_add_typed::<f64>(a, b, result, len),
            DType::Float32 => self.elementwise_add_typed::<f32>(a, b, result, len),
            DType::Int64 => self.elementwise_add_typed::<i64>(a, b, result, len),
            DType::Int32 => self.elementwise_add_typed::<i32>(a, b, result, len),
            DType::Bool => self.elementwise_add_typed::<bool>(a, b, result, len),
            _ => Err(BackendError::UnsupportedType(dtype)),
        }
    }

    unsafe fn elementwise_mul_raw(
        &self,
        dtype: DType,
        a: *const u8,
        b: *const u8,
        result: *mut u8,
        len: usize,
    ) -> BackendResult<()> {
        match dtype {
            DType::Float64 => self.elementwise_mul_typed::<f64>(a, b, result, len),
            DType::Float32 => self.elementwise_mul_typed::<f32>(a, b, result, len),
            DType::Int64 => self.elementwise_mul_typed::<i64>(a, b, result, len),
            DType::Int32 => self.elementwise_mul_typed::<i32>(a, b, result, len),
            DType::Bool => self.elementwise_mul_typed::<bool>(a, b, result, len),
            _ => Err(BackendError::UnsupportedType(dtype)),
        }
    }

    unsafe fn reduce_sum_raw(
        &self,
        dtype: DType,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()> {
        match dtype {
            DType::Float64 => self.reduce_sum_typed::<f64>(data, len, result),
            DType::Float32 => self.reduce_sum_typed::<f32>(data, len, result),
            DType::Int64 => self.reduce_sum_typed::<i64>(data, len, result),
            DType::Int32 => self.reduce_sum_typed::<i32>(data, len, result),
            DType::Bool => self.reduce_sum_typed::<bool>(data, len, result),
            _ => Err(BackendError::UnsupportedType(dtype)),
        }
    }

    unsafe fn reduce_max_raw(
        &self,
        dtype: DType,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()> {
        match dtype {
            DType::Float64 => self.reduce_max_typed::<f64>(data, len, result),
            DType::Float32 => self.reduce_max_typed::<f32>(data, len, result),
            DType::Int64 => self.reduce_max_typed::<i64>(data, len, result),
            DType::Int32 => self.reduce_max_typed::<i32>(data, len, result),
            DType::Bool => self.reduce_max_typed::<bool>(data, len, result),
            _ => Err(BackendError::UnsupportedType(dtype)),
        }
    }

    unsafe fn reduce_min_raw(
        &self,
        dtype: DType,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()> {
        match dtype {
            DType::Float64 => self.reduce_min_typed::<f64>(data, len, result),
            DType::Float32 => self.reduce_min_typed::<f32>(data, len, result),
            DType::Int64 => self.reduce_min_typed::<i64>(data, len, result),
            DType::Int32 => self.reduce_min_typed::<i32>(data, len, result),
            DType::Bool => self.reduce_min_typed::<bool>(data, len, result),
            _ => Err(BackendError::UnsupportedType(dtype)),
        }
    }
}

impl NativeBackend {
    /// Type-safe GEMM implementation
    unsafe fn gemm_typed<T: NumericType>(
        &self,
        a: *const u8,
        a_shape: (usize, usize),
        b: *const u8,
        b_shape: (usize, usize),
        c: *mut u8,
        c_shape: (usize, usize),
        alpha: *const u8,
        beta: *const u8,
    ) -> BackendResult<()> {
        let a = std::slice::from_raw_parts(a as *const T, a_shape.0 * a_shape.1);
        let b = std::slice::from_raw_parts(b as *const T, b_shape.0 * b_shape.1);
        let c = std::slice::from_raw_parts_mut(c as *mut T, c_shape.0 * c_shape.1);
        let alpha = *(alpha as *const T);
        let beta = *(beta as *const T);

        let (m, k) = a_shape;
        let (k2, n) = b_shape;
        let (m2, n2) = c_shape;

        if k != k2 || m != m2 || n != n2 {
            return Err(BackendError::IncompatibleDimensions {
                expected: (m, n),
                got: (m2, n2),
            });
        }

        // Native implementation of C = alpha * A * B + beta * C
        for i in 0..m {
            for j in 0..n {
                let mut sum = T::zero();
                for l in 0..k {
                    let a_val = a[i * k + l];
                    let b_val = b[l * n + j];
                    sum = sum.add(a_val.mul(b_val));
                }
                let c_idx = i * n + j;
                c[c_idx] = alpha.mul(sum).add(beta.mul(c[c_idx]));
            }
        }

        Ok(())
    }

    /// Type-safe GEMV implementation
    unsafe fn gemv_typed<T: NumericType>(
        &self,
        matrix: *const u8,
        matrix_shape: (usize, usize),
        vector: *const u8,
        result: *mut u8,
        alpha: *const u8,
        beta: *const u8,
    ) -> BackendResult<()> {
        let (m, n) = matrix_shape;
        let matrix = std::slice::from_raw_parts(matrix as *const T, m * n);
        let vector = std::slice::from_raw_parts(vector as *const T, n);
        let result = std::slice::from_raw_parts_mut(result as *mut T, m);
        let alpha = *(alpha as *const T);
        let beta = *(beta as *const T);

        // y = alpha * A * x + beta * y
        for i in 0..m {
            let mut sum = T::zero();
            for j in 0..n {
                sum = sum.add(matrix[i * n + j].mul(vector[j]));
            }
            result[i] = alpha.mul(sum).add(beta.mul(result[i]));
        }

        Ok(())
    }

    /// Type-safe element-wise addition
    unsafe fn elementwise_add_typed<T: NumericType>(
        &self,
        a: *const u8,
        b: *const u8,
        result: *mut u8,
        len: usize,
    ) -> BackendResult<()> {
        let a = std::slice::from_raw_parts(a as *const T, len);
        let b = std::slice::from_raw_parts(b as *const T, len);
        let result = std::slice::from_raw_parts_mut(result as *mut T, len);

        T::simd_add(a, b, result);
        Ok(())
    }

    /// Type-safe element-wise multiplication
    unsafe fn elementwise_mul_typed<T: NumericType>(
        &self,
        a: *const u8,
        b: *const u8,
        result: *mut u8,
        len: usize,
    ) -> BackendResult<()> {
        let a = std::slice::from_raw_parts(a as *const T, len);
        let b = std::slice::from_raw_parts(b as *const T, len);
        let result = std::slice::from_raw_parts_mut(result as *mut T, len);

        T::simd_mul(a, b, result);
        Ok(())
    }

    /// Type-safe sum reduction
    unsafe fn reduce_sum_typed<T: NumericType>(
        &self,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()> {
        let data = std::slice::from_raw_parts(data as *const T, len);
        let result = &mut *(result as *mut T::Accumulator);

        *result = T::simd_reduce_sum(data);
        Ok(())
    }

    /// Type-safe max reduction
    unsafe fn reduce_max_typed<T: NumericType>(
        &self,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()> {
        if len == 0 {
            return Err(BackendError::InvalidInput(
                "Empty data for reduction".to_string(),
            ));
        }

        let data = std::slice::from_raw_parts(data as *const T, len);
        let result = &mut *(result as *mut T);

        *result = T::simd_reduce_max(data);
        Ok(())
    }

    /// Type-safe min reduction
    unsafe fn reduce_min_typed<T: NumericType>(
        &self,
        data: *const u8,
        len: usize,
        result: *mut u8,
    ) -> BackendResult<()> {
        if len == 0 {
            return Err(BackendError::InvalidInput(
                "Empty data for reduction".to_string(),
            ));
        }

        let data = std::slice::from_raw_parts(data as *const T, len);
        let result = &mut *(result as *mut T);

        *result = T::simd_reduce_min(data);
        Ok(())
    }
}

/// Backend selector for intelligent backend selection
#[derive(Debug)]
pub struct BackendSelector {
    backends: Vec<Box<dyn ComputeBackend>>,
    performance_cache: HashMap<(String, OperationType, usize), f64>,
}

impl BackendSelector {
    pub fn new() -> Self {
        let mut backends: Vec<Box<dyn ComputeBackend>> = vec![];

        // Add BLAS backend if available (highest priority for supported ops)
        let blas_backend = crate::storage::advanced_matrix::backends::BlasBackend::new();
        if blas_backend.is_available() {
            backends.push(Box::new(blas_backend));
        }

        // Add NumPy backend if available (second priority)
        let numpy_backend = crate::storage::advanced_matrix::backends::NumPyBackend::new();
        if numpy_backend.is_available() {
            backends.push(Box::new(numpy_backend));
        }

        // Always add Native backend as fallback
        backends.push(Box::new(NativeBackend));

        // TODO: Add GPU backends when available
        // if CudaBackend::is_available() { backends.push(Box::new(CudaBackend::new())); }

        Self {
            backends,
            performance_cache: HashMap::new(),
        }
    }

    /// Select the optimal backend for a given operation
    pub fn select_backend(
        &self,
        op: OperationType,
        size: usize,
        dtype: DType,
        hint: BackendHint,
    ) -> &dyn ComputeBackend {
        // Filter available backends that support the operation and type
        let candidates: Vec<_> = self
            .backends
            .iter()
            .filter(|backend| {
                backend.is_available()
                    && backend.supports_operation(op)
                    && backend.supports_dtype(dtype)
            })
            .collect();

        if candidates.is_empty() {
            // Fallback to native backend
            return self.backends[0].as_ref();
        }

        // Apply selection logic based on hint and size
        let best_backend = match hint {
            BackendHint::PreferSpeed => self.select_fastest_backend(&candidates, op, size),
            BackendHint::PreferMemory => {
                self.select_memory_efficient_backend(&candidates, op, size)
            }
            BackendHint::PreferGPU => self
                .select_gpu_backend(&candidates)
                .unwrap_or_else(|| candidates[0]),
            BackendHint::PreferCPU => self
                .select_cpu_backend(&candidates)
                .unwrap_or_else(|| candidates[0]),
            BackendHint::AutoSelect => self.auto_select_backend(&candidates, op, size, dtype),
            _ => candidates[0], // Default to first available
        };

        best_backend.as_ref()
    }

    fn select_fastest_backend<'a>(
        &self,
        candidates: &[&'a Box<dyn ComputeBackend>],
        op: OperationType,
        _size: usize,
    ) -> &'a Box<dyn ComputeBackend> {
        candidates
            .iter()
            .max_by(|a, b| {
                let perf_a = a.performance_characteristics(op).throughput_ops_per_sec;
                let perf_b = b.performance_characteristics(op).throughput_ops_per_sec;
                perf_a.partial_cmp(&perf_b).unwrap()
            })
            .unwrap_or(&candidates[0])
    }

    fn select_memory_efficient_backend<'a>(
        &self,
        candidates: &[&'a Box<dyn ComputeBackend>],
        op: OperationType,
        _size: usize,
    ) -> &'a Box<dyn ComputeBackend> {
        candidates
            .iter()
            .max_by(|a, b| {
                let mem_a = a
                    .performance_characteristics(op)
                    .memory_bandwidth_gb_per_sec;
                let mem_b = b
                    .performance_characteristics(op)
                    .memory_bandwidth_gb_per_sec;
                mem_a.partial_cmp(&mem_b).unwrap()
            })
            .unwrap_or(&candidates[0])
    }

    fn select_gpu_backend<'a>(
        &self,
        candidates: &[&'a Box<dyn ComputeBackend>],
    ) -> Option<&'a Box<dyn ComputeBackend>> {
        candidates
            .iter()
            .find(|backend| backend.name().contains("CUDA") || backend.name().contains("OpenCL"))
            .copied()
    }

    fn select_cpu_backend<'a>(
        &self,
        candidates: &[&'a Box<dyn ComputeBackend>],
    ) -> Option<&'a Box<dyn ComputeBackend>> {
        candidates
            .iter()
            .find(|backend| !backend.name().contains("CUDA") && !backend.name().contains("OpenCL"))
            .copied()
    }

    fn auto_select_backend<'a>(
        &self,
        candidates: &[&'a Box<dyn ComputeBackend>],
        op: OperationType,
        size: usize,
        dtype: DType,
    ) -> &'a Box<dyn ComputeBackend> {
        // Enhanced intelligent selection based on multiple factors
        let mut best_backend = candidates[0];
        let mut best_score = 0.0;

        for backend in candidates {
            let threshold = backend.optimal_threshold(op);
            let perf = backend.performance_characteristics(op);

            // Calculate performance score for this backend
            let mut score = 0.0;

            // Factor 1: Raw throughput (weighted by operation type)
            let throughput_weight = match op {
                OperationType::GEMM | OperationType::GEMV => 3.0, // High weight for matrix ops
                OperationType::SVD | OperationType::QR => 2.5,    // High weight for decompositions
                OperationType::Cholesky | OperationType::LU => 2.5, // High weight for decompositions
                _ => 1.0,                                           // Standard weight for others
            };
            score += (perf.throughput_ops_per_sec / 1e6) * throughput_weight;

            // Factor 2: Size compatibility
            if size >= perf.optimal_size_range.0 && size <= perf.optimal_size_range.1 {
                score += 50.0; // Large bonus for optimal size range
            } else if size > threshold {
                score += 20.0; // Moderate bonus for above threshold
            } else if size < threshold && backend.name() == "Native" {
                score += 30.0; // Native backend is good for small operations
            }

            // Factor 3: Setup overhead penalty (higher penalty for small operations)
            let overhead_penalty = perf.setup_overhead_us * (10000.0 / size as f64).min(5.0);
            score -= overhead_penalty;

            // Factor 4: Data type compatibility
            if backend.supports_dtype(dtype) {
                score += 25.0; // Bonus for native type support
            }

            // Factor 5: Backend-specific bonuses
            match backend.name() {
                name if name.contains("MKL") => {
                    // Intel MKL gets bonus for x86 CPUs and large matrices
                    if size > 1000 {
                        score += 40.0;
                    }
                }
                "OpenBLAS" => {
                    // OpenBLAS gets bonus for matrix operations
                    if matches!(op, OperationType::GEMM | OperationType::GEMV) {
                        score += 30.0;
                    }
                }
                "NumPy" => {
                    // NumPy gets bonus for decompositions and medium-sized operations
                    if matches!(
                        op,
                        OperationType::SVD | OperationType::QR | OperationType::Cholesky
                    ) {
                        score += 35.0;
                    }
                    if size > 100 && size < 5000 {
                        score += 20.0;
                    }
                }
                "Native" => {
                    // Native backend gets bonus for small operations and always-available
                    if size < 100 {
                        score += 25.0;
                    }
                    score += 10.0; // Always available bonus
                }
                _ => {}
            }

            // Factor 6: Memory bandwidth for large operations
            if size > 10000 {
                score += perf.memory_bandwidth_gb_per_sec * 2.0;
            }

            // Update best backend if this one scores higher
            if score > best_score {
                best_score = score;
                best_backend = backend;
            }
        }

        best_backend
    }

    /// Get all available backends
    pub fn available_backends(&self) -> Vec<&str> {
        self.backends
            .iter()
            .filter(|backend| backend.is_available())
            .map(|backend| backend.name())
            .collect()
    }

    /// Benchmark a specific backend for performance tuning
    pub fn benchmark_backend(
        &mut self,
        backend_name: &str,
        op: OperationType,
        size: usize,
    ) -> Option<f64> {
        // TODO: Implement actual benchmarking
        // This would run the operation multiple times and measure performance
        let key = (backend_name.to_string(), op, size);
        self.performance_cache.get(&key).copied()
    }
}

impl Default for BackendSelector {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_native_backend_gemm() {
        let backend = NativeBackend;

        // Test 2x2 matrix multiplication
        let a = vec![1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let b = vec![5.0, 6.0, 7.0, 8.0]; // [[5, 6], [7, 8]]
        let mut c = vec![0.0; 4];

        backend
            .gemm(&a, (2, 2), &b, (2, 2), &mut c, (2, 2), 1.0, 0.0)
            .unwrap();

        // Expected result: [[19, 22], [43, 50]]
        assert_eq!(c, vec![19.0, 22.0, 43.0, 50.0]);
    }

    #[test]
    #[ignore] // gemv method not currently implemented in NativeBackend
    fn test_native_backend_gemv() {
        let _backend = NativeBackend;

        let _matrix = [1.0, 2.0, 3.0, 4.0]; // [[1, 2], [3, 4]]
        let _vector = [5.0, 6.0];
        let _result = [0.0; 2];

        // backend
        //     .gemv(&matrix, (2, 2), &vector, &mut result, 1.0, 0.0)
        //     .unwrap();

        // Expected result: [17, 39] = [1*5 + 2*6, 3*5 + 4*6]
        // assert_eq!(result, vec![17.0, 39.0]);
    }

    #[test]
    fn test_backend_selection() {
        let selector = BackendSelector::new();

        let backend = selector.select_backend(
            OperationType::GEMM,
            100,
            DType::Float64,
            BackendHint::AutoSelect,
        );

        assert_eq!(backend.name(), "Native");
        assert!(backend.is_available());
    }

    #[test]
    fn test_dtype_promotion() {
        assert_eq!(
            DType::promote(DType::Float32, DType::Float64),
            DType::Float64
        );
        assert_eq!(DType::promote(DType::Int32, DType::Bool), DType::Int32);
    }
}
