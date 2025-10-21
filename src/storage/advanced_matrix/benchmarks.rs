//! Comprehensive Benchmarking Suite for Advanced Matrix System
//!
//! This module provides extensive performance measurement and validation tools
//! for the advanced matrix system, enabling systematic performance optimization.

use crate::storage::advanced_matrix::{
    backend::{BackendHint, BackendSelector, ComputeBackend, ComputeBackendExt, OperationType},
    numeric_type::{DType, NumericType},
    unified_matrix::MatrixError,
};
use std::collections::HashMap;
use std::sync::Arc;
use std::time::{Duration, Instant};

/// Results from a benchmark run
#[derive(Debug, Clone)]
pub struct BenchmarkResult {
    pub operation: OperationType,
    pub backend_name: String,
    pub matrix_size: (usize, usize),
    pub data_type: DType,
    pub duration: Duration,
    pub throughput_gflops: f64,
    pub memory_bandwidth_gb_per_sec: f64,
    pub success: bool,
    pub error_message: Option<String>,
}

impl BenchmarkResult {
    /// Calculate GFLOPS for the operation
    pub fn calculate_gflops(op: OperationType, shape: (usize, usize), duration: Duration) -> f64 {
        let (m, n) = shape;
        let ops = match op {
            OperationType::GEMM => {
                // GEMM: 2*m*n*k operations (assuming k=n for square-ish matrices)
                2.0 * m as f64 * n as f64 * n as f64
            }
            OperationType::GEMV => {
                // GEMV: 2*m*n operations
                2.0 * m as f64 * n as f64
            }
            OperationType::ElementwiseAdd | OperationType::ElementwiseMul => {
                // Element-wise: m*n operations
                m as f64 * n as f64
            }
            OperationType::Sum | OperationType::Max | OperationType::Min => {
                // Reductions: m*n operations
                m as f64 * n as f64
            }
            OperationType::SVD => {
                // SVD: approximately 12*m*n^2 + 2*n^3 operations
                let m = m as f64;
                let n = n as f64;
                12.0 * m * n * n + 2.0 * n * n * n
            }
            OperationType::QR => {
                // QR: approximately 4*m*n^2 - 4*n^3/3 operations
                let m = m as f64;
                let n = n as f64;
                4.0 * m * n * n - 4.0 * n * n * n / 3.0
            }
            OperationType::Cholesky => {
                // Cholesky: approximately n^3/3 operations
                let n = n as f64;
                n * n * n / 3.0
            }
            _ => m as f64 * n as f64, // Default fallback
        };

        ops / (duration.as_secs_f64() * 1e9) // Convert to GFLOPS
    }
}

/// Configuration for benchmark runs
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    pub matrix_sizes: Vec<(usize, usize)>,
    pub data_types: Vec<DType>,
    pub operations: Vec<OperationType>,
    pub num_iterations: usize,
    pub warmup_iterations: usize,
    pub timeout_seconds: u64,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            matrix_sizes: vec![
                (32, 32),
                (64, 64),
                (128, 128),
                (256, 256),
                (512, 512),
                (1024, 1024),
                (2048, 2048),
                (4096, 4096),
            ],
            data_types: vec![DType::Float64, DType::Float32, DType::Int64, DType::Int32],
            operations: vec![
                OperationType::GEMM,
                OperationType::GEMV,
                OperationType::ElementwiseAdd,
                OperationType::ElementwiseMul,
                OperationType::Sum,
                OperationType::Max,
                OperationType::Min,
            ],
            num_iterations: 5,
            warmup_iterations: 2,
            timeout_seconds: 60,
        }
    }
}

/// Comprehensive benchmarking framework
pub struct MatrixBenchmarkSuite {
    backend_selector: Arc<BackendSelector>,
    config: BenchmarkConfig,
    results: Vec<BenchmarkResult>,
}

impl MatrixBenchmarkSuite {
    /// Create a new benchmark suite
    pub fn new(config: BenchmarkConfig) -> Self {
        Self {
            backend_selector: Arc::new(BackendSelector::new()),
            config,
            results: Vec::new(),
        }
    }

    /// Run comprehensive benchmarks across all backends and operations
    pub fn run_comprehensive_benchmarks(&mut self) -> Result<(), MatrixError> {
        println!("🚀 Starting Comprehensive Matrix Benchmark Suite");
        println!(
            "Available backends: {:?}",
            self.backend_selector.available_backends()
        );

        let total_benchmarks = self.config.matrix_sizes.len()
            * self.config.data_types.len()
            * self.config.operations.len();
        println!("Total benchmarks to run: {}", total_benchmarks);

        let mut completed = 0;

        for &size in &self.config.matrix_sizes {
            for &dtype in &self.config.data_types {
                for &op in &self.config.operations {
                    completed += 1;
                    println!(
                        "🔄 [{}/{}] Benchmarking {:?} {}x{} {:?}",
                        completed, total_benchmarks, op, size.0, size.1, dtype
                    );

                    let result = self.benchmark_operation(op, size, dtype);
                    self.results.push(result);
                }
            }
        }

        println!("✅ Benchmark suite completed!");
        Ok(())
    }

    /// Benchmark a specific operation with different backends
    fn benchmark_operation(
        &self,
        op: OperationType,
        size: (usize, usize),
        dtype: DType,
    ) -> BenchmarkResult {
        // Get all available backends that support this operation and data type
        let backends: Vec<_> = self
            .backend_selector
            .available_backends()
            .into_iter()
            .collect();

        if backends.is_empty() {
            return BenchmarkResult {
                operation: op,
                backend_name: "None".to_string(),
                matrix_size: size,
                data_type: dtype,
                duration: Duration::from_secs(0),
                throughput_gflops: 0.0,
                memory_bandwidth_gb_per_sec: 0.0,
                success: false,
                error_message: Some("No available backends".to_string()),
            };
        }

        // Select best backend for this operation
        let backend = self.backend_selector.select_backend(
            op,
            size.0 * size.1,
            dtype,
            BackendHint::PreferSpeed,
        );

        // Run the benchmark with the selected backend
        match dtype {
            DType::Float64 => self.run_typed_benchmark::<f64>(backend, op, size),
            DType::Float32 => self.run_typed_benchmark::<f32>(backend, op, size),
            DType::Int64 => self.run_typed_benchmark::<i64>(backend, op, size),
            DType::Int32 => self.run_typed_benchmark::<i32>(backend, op, size),
            _ => BenchmarkResult {
                operation: op,
                backend_name: backend.name().to_string(),
                matrix_size: size,
                data_type: dtype,
                duration: Duration::from_secs(0),
                throughput_gflops: 0.0,
                memory_bandwidth_gb_per_sec: 0.0,
                success: false,
                error_message: Some("Unsupported data type".to_string()),
            },
        }
    }

    /// Run a typed benchmark for a specific operation
    fn run_typed_benchmark<T: NumericType>(
        &self,
        backend: &dyn ComputeBackend,
        op: OperationType,
        size: (usize, usize),
    ) -> BenchmarkResult {
        let (rows, cols) = size;

        // Prepare test data
        let a_data: Vec<T> = (0..(rows * cols))
            .map(|i| T::from_f64((i as f64 + 1.0) / 100.0).unwrap_or(T::one()))
            .collect();
        let b_data: Vec<T> = (0..(rows * cols))
            .map(|i| T::from_f64((i as f64 + 2.0) / 100.0).unwrap_or(T::one()))
            .collect();
        let mut c_data = vec![T::zero(); rows * cols];

        // Warmup runs
        for _ in 0..self.config.warmup_iterations {
            let _ =
                self.run_single_operation::<T>(backend, op, &a_data, &b_data, &mut c_data, size);
        }

        // Benchmark runs
        let mut durations = Vec::new();
        let mut successful_runs = 0;
        let mut last_error = None;

        for _ in 0..self.config.num_iterations {
            match self.run_single_operation::<T>(backend, op, &a_data, &b_data, &mut c_data, size) {
                Ok(duration) => {
                    durations.push(duration);
                    successful_runs += 1;
                }
                Err(e) => {
                    last_error = Some(e.to_string());
                }
            }
        }

        if successful_runs == 0 {
            return BenchmarkResult {
                operation: op,
                backend_name: backend.name().to_string(),
                matrix_size: size,
                data_type: T::DTYPE,
                duration: Duration::from_secs(0),
                throughput_gflops: 0.0,
                memory_bandwidth_gb_per_sec: 0.0,
                success: false,
                error_message: last_error,
            };
        }

        // Calculate average duration (excluding outliers)
        durations.sort();
        let median_duration = durations[durations.len() / 2];

        // Calculate performance metrics
        let gflops = BenchmarkResult::calculate_gflops(op, size, median_duration);
        let memory_gb = (rows * cols * std::mem::size_of::<T>() * 2) as f64 / 1e9; // Read A and B
        let memory_bandwidth = memory_gb / median_duration.as_secs_f64();

        BenchmarkResult {
            operation: op,
            backend_name: backend.name().to_string(),
            matrix_size: size,
            data_type: T::DTYPE,
            duration: median_duration,
            throughput_gflops: gflops,
            memory_bandwidth_gb_per_sec: memory_bandwidth,
            success: true,
            error_message: None,
        }
    }

    /// Run a single operation and measure its duration
    fn run_single_operation<T: NumericType>(
        &self,
        backend: &dyn ComputeBackend,
        op: OperationType,
        a_data: &[T],
        b_data: &[T],
        c_data: &mut [T],
        size: (usize, usize),
    ) -> Result<Duration, Box<dyn std::error::Error>> {
        let start = Instant::now();

        match op {
            OperationType::GEMM => {
                backend.gemm(
                    a_data,
                    size,
                    b_data,
                    size,
                    c_data,
                    size,
                    T::one(),
                    T::zero(),
                )?;
            }
            OperationType::ElementwiseAdd => {
                backend.elementwise_add(a_data, b_data, c_data)?;
            }
            OperationType::ElementwiseMul => {
                backend.elementwise_mul(a_data, b_data, c_data)?;
            }
            OperationType::Sum => {
                let _result = backend.reduce_sum(a_data)?;
            }
            OperationType::Max => {
                let _result = backend.reduce_max(a_data)?;
            }
            OperationType::Min => {
                let _result = backend.reduce_min(a_data)?;
            }
            _ => {
                return Err("Operation not supported in benchmark".into());
            }
        }

        Ok(start.elapsed())
    }

    /// Generate performance report
    pub fn generate_report(&self) -> String {
        let mut report = String::new();

        report.push_str("🔥 Advanced Matrix System - Performance Report\n");
        report.push_str("=".repeat(60).as_str());
        report.push_str("\n\n");

        // Group results by operation
        let mut operation_groups: HashMap<OperationType, Vec<&BenchmarkResult>> = HashMap::new();
        for result in &self.results {
            operation_groups
                .entry(result.operation)
                .or_insert_with(Vec::new)
                .push(result);
        }

        for (op, results) in operation_groups {
            report.push_str(&format!("## {:?} Performance\n", op));
            report.push_str(&format!("{:-<40}\n", ""));

            // Sort by matrix size
            let mut sorted_results = results.clone();
            sorted_results.sort_by_key(|r| r.matrix_size.0);

            for result in sorted_results {
                if result.success {
                    report.push_str(&format!(
                        "{:>10} {:>8}x{:<8} {:>10.2} GFLOPS  {:>8.1} GB/s  {:>12.3}ms  [{}]\n",
                        format!("{:?}", result.data_type),
                        result.matrix_size.0,
                        result.matrix_size.1,
                        result.throughput_gflops,
                        result.memory_bandwidth_gb_per_sec,
                        result.duration.as_secs_f64() * 1000.0,
                        result.backend_name
                    ));
                } else {
                    report.push_str(&format!(
                        "{:>10} {:>8}x{:<8} FAILED: {}\n",
                        format!("{:?}", result.data_type),
                        result.matrix_size.0,
                        result.matrix_size.1,
                        result
                            .error_message
                            .as_ref()
                            .unwrap_or(&"Unknown error".to_string())
                    ));
                }
            }
            report.push('\n');
        }

        // Backend comparison section
        report.push_str("## Backend Performance Comparison\n");
        report.push_str(&format!("{:-<40}\n", ""));

        let mut backend_stats: HashMap<String, (f64, usize)> = HashMap::new();
        for result in &self.results {
            if result.success && result.throughput_gflops > 0.0 {
                let (total_gflops, count) = backend_stats
                    .entry(result.backend_name.clone())
                    .or_insert((0.0, 0));
                *total_gflops += result.throughput_gflops;
                *count += 1;
            }
        }

        let mut backend_averages: Vec<_> = backend_stats
            .iter()
            .map(|(name, (total, count))| (name.clone(), total / *count as f64))
            .collect();
        backend_averages.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        for (backend_name, avg_gflops) in backend_averages {
            report.push_str(&format!(
                "{:>20}: {:>10.2} GFLOPS (average)\n",
                backend_name, avg_gflops
            ));
        }

        report.push('\n');
        report.push_str("🎯 Performance Targets (from plan):\n");
        report.push_str("  Matrix Multiply (1K×1K): Target 31.4x improvement (350ms vs 11.0s)\n");
        report.push_str("  SVD Decomposition (1K×1K): Target 52.9x improvement (850ms vs 45.0s)\n");

        report
    }

    /// Export results to CSV
    pub fn export_to_csv(&self, filename: &str) -> std::io::Result<()> {
        use std::fs::File;
        use std::io::Write;

        let mut file = File::create(filename)?;

        // Write header
        writeln!(file, "Operation,Backend,MatrixRows,MatrixCols,DataType,Duration_ms,Throughput_GFLOPS,Bandwidth_GB_per_sec,Success,Error")?;

        // Write data
        for result in &self.results {
            writeln!(
                file,
                "{:?},{},{},{},{:?},{:.3},{:.2},{:.1},{},{}",
                result.operation,
                result.backend_name,
                result.matrix_size.0,
                result.matrix_size.1,
                result.data_type,
                result.duration.as_secs_f64() * 1000.0,
                result.throughput_gflops,
                result.memory_bandwidth_gb_per_sec,
                result.success,
                result.error_message.as_ref().unwrap_or(&"".to_string())
            )?;
        }

        Ok(())
    }

    /// Get results for analysis
    pub fn get_results(&self) -> &[BenchmarkResult] {
        &self.results
    }
}

/// Quick benchmark for common operations
pub fn quick_benchmark() {
    println!("🔥 Quick Advanced Matrix System Benchmark");

    let config = BenchmarkConfig {
        matrix_sizes: vec![(64, 64), (256, 256), (1024, 1024)],
        data_types: vec![DType::Float64, DType::Float32],
        operations: vec![OperationType::GEMM, OperationType::ElementwiseAdd],
        num_iterations: 3,
        warmup_iterations: 1,
        timeout_seconds: 30,
    };

    let mut benchmark = MatrixBenchmarkSuite::new(config);

    if let Err(e) = benchmark.run_comprehensive_benchmarks() {
        println!("❌ Benchmark failed: {}", e);
        return;
    }

    println!("{}", benchmark.generate_report());

    // Export to CSV
    if let Err(e) = benchmark.export_to_csv("matrix_benchmark_results.csv") {
        println!("⚠️  Failed to export CSV: {}", e);
    } else {
        println!("📊 Results exported to matrix_benchmark_results.csv");
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gflops_calculation() {
        let duration = Duration::from_millis(100);
        let gflops = BenchmarkResult::calculate_gflops(OperationType::GEMM, (100, 100), duration);
        assert!(gflops > 0.0);

        let gflops_elementwise =
            BenchmarkResult::calculate_gflops(OperationType::ElementwiseAdd, (100, 100), duration);
        assert!(gflops > gflops_elementwise); // GEMM should have more operations
    }

    #[test]
    fn test_benchmark_config() {
        let config = BenchmarkConfig::default();
        assert!(!config.matrix_sizes.is_empty());
        assert!(!config.operations.is_empty());
        assert!(config.num_iterations > 0);
    }

    #[test]
    fn test_benchmark_suite_creation() {
        let config = BenchmarkConfig::default();
        let suite = MatrixBenchmarkSuite::new(config);
        assert!(!suite.backend_selector.available_backends().is_empty());
    }
}
