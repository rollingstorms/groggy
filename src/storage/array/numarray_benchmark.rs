//! NumArray Performance Benchmarking Suite
//!
//! This module provides comprehensive benchmarking for NumArray operations
//! to establish performance baselines and identify optimization opportunities.

use crate::errors::GraphResult;
use crate::storage::array::NumArray;
use std::collections::HashMap;
use std::time::{Duration, Instant};

/// Comprehensive benchmark configuration for NumArray operations
#[derive(Debug, Clone)]
pub struct NumArrayBenchmarkConfig {
    /// Array sizes to test (small, medium, large)
    pub sizes: Vec<usize>,
    /// Number of iterations per test
    pub iterations: usize,
    /// Whether to include memory usage measurements
    pub measure_memory: bool,
    /// Whether to compare against naive implementations
    pub include_baseline_comparison: bool,
}

impl Default for NumArrayBenchmarkConfig {
    fn default() -> Self {
        Self {
            sizes: vec![100, 1_000, 10_000, 100_000],
            iterations: 50,
            measure_memory: true,
            include_baseline_comparison: true,
        }
    }
}

/// Results for a single benchmark operation
#[derive(Debug, Clone)]
pub struct OperationResult {
    pub operation: String,
    pub size: usize,
    pub average_time: Duration,
    pub min_time: Duration,
    pub max_time: Duration,
    pub throughput_elements_per_second: f64,
    pub memory_usage_bytes: Option<u64>,
    pub iterations: usize,
}

/// Complete benchmark results for NumArray performance analysis
#[derive(Debug, Clone)]
pub struct NumArrayBenchmarkResults {
    pub config: NumArrayBenchmarkConfig,
    pub operation_results: Vec<OperationResult>,
    pub baseline_comparisons: HashMap<String, f64>, // Operation -> speedup ratio
    pub timestamp: std::time::SystemTime,
}

impl NumArrayBenchmarkResults {
    /// Get results for a specific operation and size
    pub fn get_operation_result(&self, operation: &str, size: usize) -> Option<&OperationResult> {
        self.operation_results
            .iter()
            .find(|r| r.operation == operation && r.size == size)
    }

    /// Calculate performance improvement over baseline for an operation
    pub fn baseline_improvement(&self, operation: &str) -> Option<f64> {
        self.baseline_comparisons.get(operation).copied()
    }

    /// Generate performance summary report
    pub fn generate_summary(&self) -> String {
        let mut report = String::new();

        report.push_str("# NumArray Performance Benchmark Summary\n\n");
        report.push_str(&format!("**Timestamp**: {:?}\n", self.timestamp));
        report.push_str(&format!(
            "**Iterations per test**: {}\n",
            self.config.iterations
        ));
        report.push_str(&format!(
            "**Array sizes tested**: {:?}\n\n",
            self.config.sizes
        ));

        // Group results by operation
        let mut operations: HashMap<String, Vec<&OperationResult>> = HashMap::new();
        for result in &self.operation_results {
            operations
                .entry(result.operation.clone())
                .or_default()
                .push(result);
        }

        for (operation, results) in operations {
            report.push_str(&format!("## {}\n\n", operation));

            report.push_str("| Size | Time | Throughput (elements/sec) | Memory (bytes) |\n");
            report.push_str("|------|------|---------------------------|---------------|\n");

            for result in results {
                report.push_str(&format!(
                    "| {} | {:?} | {:.0} | {} |\n",
                    result.size,
                    result.average_time,
                    result.throughput_elements_per_second,
                    result
                        .memory_usage_bytes
                        .map(|m| m.to_string())
                        .unwrap_or_else(|| "N/A".to_string())
                ));
            }

            if let Some(improvement) = self.baseline_improvement(&operation) {
                report.push_str(&format!(
                    "\n**Baseline improvement**: {:.2}x faster\n\n",
                    improvement
                ));
            } else {
                report.push('\n');
            }
        }

        report
    }
}

/// NumArray performance benchmarker
pub struct NumArrayBenchmarker {
    config: NumArrayBenchmarkConfig,
}

impl NumArrayBenchmarker {
    pub fn new() -> Self {
        Self {
            config: NumArrayBenchmarkConfig::default(),
        }
    }

    pub fn with_config(config: NumArrayBenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run comprehensive NumArray performance benchmark
    pub fn run_comprehensive_benchmark(&self) -> GraphResult<NumArrayBenchmarkResults> {
        println!("🚀 Starting NumArray Performance Benchmark Suite");
        println!(
            "Sizes: {:?}, Iterations: {}",
            self.config.sizes, self.config.iterations
        );

        let mut results = NumArrayBenchmarkResults {
            config: self.config.clone(),
            operation_results: Vec::new(),
            baseline_comparisons: HashMap::new(),
            timestamp: std::time::SystemTime::now(),
        };

        // Test different operations across different sizes
        for &size in &self.config.sizes {
            println!("\n📊 Testing array size: {}", size);

            // Basic statistical operations
            self.benchmark_statistical_operations(size, &mut results)?;

            // Element-wise operations
            self.benchmark_elementwise_operations(size, &mut results)?;

            // Memory allocation operations
            self.benchmark_allocation_operations(size, &mut results)?;

            // Iterator operations
            self.benchmark_iterator_operations(size, &mut results)?;
        }

        // Run baseline comparisons if requested
        if self.config.include_baseline_comparison {
            println!("\n🔍 Running baseline comparisons...");
            self.run_baseline_comparisons(&mut results)?;
        }

        println!("\n✅ Benchmark suite completed!");
        Ok(results)
    }

    /// Benchmark statistical operations (mean, sum, std_dev, etc.)
    fn benchmark_statistical_operations(
        &self,
        size: usize,
        results: &mut NumArrayBenchmarkResults,
    ) -> GraphResult<()> {
        let test_data = self.generate_f64_test_data(size);
        let array = NumArray::new(test_data);

        // Benchmark mean calculation
        let mean_result = self.benchmark_operation("mean", size, || {
            let _ = array.mean();
        });
        results.operation_results.push(mean_result);

        // Benchmark sum calculation
        let sum_result = self.benchmark_operation("sum", size, || {
            let _ = array.sum();
        });
        results.operation_results.push(sum_result);

        // Benchmark standard deviation
        let std_result = self.benchmark_operation("std_dev", size, || {
            let _ = array.std_dev();
        });
        results.operation_results.push(std_result);

        // Benchmark min/max
        let min_result = self.benchmark_operation("min", size, || {
            let _ = array.min();
        });
        results.operation_results.push(min_result);

        let max_result = self.benchmark_operation("max", size, || {
            let _ = array.max();
        });
        results.operation_results.push(max_result);

        // Benchmark median (expensive operation)
        let median_result = self.benchmark_operation("median", size, || {
            let _ = array.median();
        });
        results.operation_results.push(median_result);

        println!("  ✓ Statistical operations benchmarked");
        Ok(())
    }

    /// Benchmark element-wise operations
    fn benchmark_elementwise_operations(
        &self,
        size: usize,
        results: &mut NumArrayBenchmarkResults,
    ) -> GraphResult<()> {
        let test_data = self.generate_f64_test_data(size);
        let array = NumArray::new(test_data);

        // Benchmark iteration
        let iter_result = self.benchmark_operation("iteration", size, || {
            let mut _sum = 0.0;
            for &val in array.iter() {
                _sum += val;
            }
        });
        results.operation_results.push(iter_result);

        // Benchmark element access
        let access_result = self.benchmark_operation("element_access", size, || {
            let mut _sum = 0.0;
            for i in 0..array.len() {
                if let Some(&val) = array.get(i) {
                    _sum += val;
                }
            }
        });
        results.operation_results.push(access_result);

        println!("  ✓ Element-wise operations benchmarked");
        Ok(())
    }

    /// Benchmark memory allocation and creation operations
    fn benchmark_allocation_operations(
        &self,
        size: usize,
        results: &mut NumArrayBenchmarkResults,
    ) -> GraphResult<()> {
        let test_data = self.generate_f64_test_data(size);

        // Benchmark NumArray creation from Vec
        let create_result = self.benchmark_operation("creation_from_vec", size, || {
            let _ = NumArray::new(test_data.clone());
        });
        results.operation_results.push(create_result);

        // Benchmark cloning
        let array = NumArray::new(test_data.clone());
        let clone_result = self.benchmark_operation("clone", size, || {
            let _ = array.clone();
        });
        results.operation_results.push(clone_result);

        println!("  ✓ Allocation operations benchmarked");
        Ok(())
    }

    /// Benchmark iterator operations
    fn benchmark_iterator_operations(
        &self,
        size: usize,
        results: &mut NumArrayBenchmarkResults,
    ) -> GraphResult<()> {
        let test_data = self.generate_f64_test_data(size);
        let array = NumArray::new(test_data);

        // Benchmark collect to Vec
        let collect_result = self.benchmark_operation("collect_to_vec", size, || {
            let _: Vec<&f64> = array.iter().collect();
        });
        results.operation_results.push(collect_result);

        // Benchmark filtering
        let filter_result = self.benchmark_operation("filter", size, || {
            let _: Vec<&f64> = array.iter().filter(|&&x| x > 0.5).collect();
        });
        results.operation_results.push(filter_result);

        // Benchmark mapping
        let map_result = self.benchmark_operation("map", size, || {
            let _: Vec<f64> = array.iter().map(|&x| x * 2.0).collect();
        });
        results.operation_results.push(map_result);

        println!("  ✓ Iterator operations benchmarked");
        Ok(())
    }

    /// Run baseline comparisons against naive implementations
    fn run_baseline_comparisons(&self, results: &mut NumArrayBenchmarkResults) -> GraphResult<()> {
        let size = 10_000; // Use medium size for comparisons
        let test_data = self.generate_f64_test_data(size);
        let array = NumArray::new(test_data.clone());

        // Compare sum operation against naive Vec sum
        let numarray_time = self.time_operation(|| array.sum());
        let naive_time = self.time_operation(|| test_data.iter().copied().sum::<f64>());

        if naive_time.as_nanos() > 0 {
            let speedup = naive_time.as_nanos() as f64 / numarray_time.as_nanos() as f64;
            results
                .baseline_comparisons
                .insert("sum".to_string(), speedup);
        }

        // Compare mean operation
        let numarray_mean_time = self.time_operation(|| array.mean());
        let naive_mean_time = self.time_operation(|| {
            if test_data.is_empty() {
                None
            } else {
                Some(test_data.iter().copied().sum::<f64>() / test_data.len() as f64)
            }
        });

        if naive_mean_time.as_nanos() > 0 {
            let speedup = naive_mean_time.as_nanos() as f64 / numarray_mean_time.as_nanos() as f64;
            results
                .baseline_comparisons
                .insert("mean".to_string(), speedup);
        }

        Ok(())
    }

    /// Benchmark a single operation with timing and metrics
    fn benchmark_operation<F>(&self, operation: &str, size: usize, mut op: F) -> OperationResult
    where
        F: FnMut(),
    {
        let mut times = Vec::with_capacity(self.config.iterations);

        // Warmup
        for _ in 0..3 {
            op();
        }

        // Actual timing
        for _ in 0..self.config.iterations {
            let start = Instant::now();
            op();
            times.push(start.elapsed());
        }

        // Calculate statistics
        let average_time: Duration = times.iter().sum::<Duration>() / times.len() as u32;
        let min_time = *times.iter().min().unwrap();
        let max_time = *times.iter().max().unwrap();

        let throughput = if average_time.as_secs_f64() > 0.0 {
            size as f64 / average_time.as_secs_f64()
        } else {
            0.0
        };

        OperationResult {
            operation: operation.to_string(),
            size,
            average_time,
            min_time,
            max_time,
            throughput_elements_per_second: throughput,
            memory_usage_bytes: if self.config.measure_memory {
                Some(self.estimate_memory_usage(size))
            } else {
                None
            },
            iterations: self.config.iterations,
        }
    }

    /// Time a single operation
    fn time_operation<F, R>(&self, mut op: F) -> Duration
    where
        F: FnMut() -> R,
    {
        let start = Instant::now();
        let _ = op();
        start.elapsed()
    }

    /// Generate test data for f64 NumArray
    fn generate_f64_test_data(&self, size: usize) -> Vec<f64> {
        (0..size)
            .map(|i| {
                // Generate diverse data for realistic testing
                match i % 6 {
                    0 => i as f64 / 1000.0,          // Small decimals
                    1 => (i as f64).sin(),           // Sine wave
                    2 => (i as f64 * 1.618).fract(), // Golden ratio fractions
                    3 => (i as f64).sqrt(),          // Square roots
                    4 => {
                        if i % 2 == 0 {
                            i as f64
                        } else {
                            -(i as f64)
                        }
                    } // Positive/negative
                    5 => (i as f64) * 1000.0,        // Large numbers
                    _ => unreachable!(),
                }
            })
            .collect()
    }

    /// Estimate memory usage for a given array size (simplified)
    fn estimate_memory_usage(&self, size: usize) -> u64 {
        // Base NumArray memory + Vec<f64> memory + some overhead
        let vec_size = size * std::mem::size_of::<f64>();
        let struct_overhead = std::mem::size_of::<NumArray<f64>>();
        (vec_size + struct_overhead) as u64
    }

    /// Print comprehensive benchmark results
    pub fn print_results(&self, results: &NumArrayBenchmarkResults) {
        println!("\n{}", "=".repeat(80));
        println!("📊 NUMARRAY PERFORMANCE BENCHMARK RESULTS");
        println!("{}", "=".repeat(80));

        println!("Configuration:");
        println!("  • Array sizes: {:?}", results.config.sizes);
        println!("  • Iterations per test: {}", results.config.iterations);
        println!("  • Memory measurement: {}", results.config.measure_memory);

        // Group and display results by operation
        let mut operations: HashMap<String, Vec<&OperationResult>> = HashMap::new();
        for result in &results.operation_results {
            operations
                .entry(result.operation.clone())
                .or_default()
                .push(result);
        }

        for (operation, operation_results) in &operations {
            println!("\n{}", "-".repeat(60));
            println!("🔧 Operation: {}", operation.to_uppercase());
            println!("{}", "-".repeat(60));

            for result in operation_results {
                println!(
                    "  Size: {:>8} | Time: {:>10.2?} | Throughput: {:>12.0} elem/sec{}",
                    result.size,
                    result.average_time,
                    result.throughput_elements_per_second,
                    if let Some(memory) = result.memory_usage_bytes {
                        format!(" | Memory: {} bytes", memory)
                    } else {
                        String::new()
                    }
                );
            }

            // Show baseline comparison if available
            if let Some(speedup) = results.baseline_improvement(operation) {
                if speedup > 1.0 {
                    println!("  🚀 {:.2}x faster than naive implementation", speedup);
                } else {
                    println!(
                        "  ⚠️  {:.2}x slower than naive implementation",
                        1.0 / speedup
                    );
                }
            }
        }

        println!("\n{}", "=".repeat(80));
        println!("🎯 PERFORMANCE ANALYSIS");
        println!("{}", "=".repeat(80));

        // Find performance hotspots
        let slowest_ops: Vec<_> = results
            .operation_results
            .iter()
            .filter(|r| r.size == 10_000) // Focus on medium size for analysis
            .collect();

        if let Some(slowest) = slowest_ops.iter().max_by_key(|r| r.average_time) {
            println!(
                "⚠️  Slowest operation: {} ({:.2?} for 10K elements)",
                slowest.operation, slowest.average_time
            );
        }

        if let Some(fastest) = slowest_ops.iter().min_by_key(|r| r.average_time) {
            println!(
                "⚡ Fastest operation: {} ({:.2?} for 10K elements)",
                fastest.operation, fastest.average_time
            );
        }

        // Identify optimization opportunities
        let expensive_ops: Vec<_> = slowest_ops
            .iter()
            .filter(|r| r.average_time > Duration::from_millis(1))
            .collect();

        if !expensive_ops.is_empty() {
            println!("\n🎯 OPTIMIZATION OPPORTUNITIES:");
            for op in expensive_ops {
                println!(
                    "  • {} could benefit from optimization (currently {:.2?})",
                    op.operation, op.average_time
                );
            }
        }

        println!("\n💡 RECOMMENDATIONS:");

        // Check for scaling issues
        for operation in operations.keys() {
            let op_results: Vec<_> = results
                .operation_results
                .iter()
                .filter(|r| r.operation == *operation)
                .collect();

            if op_results.len() >= 2 {
                let small_result = op_results.iter().min_by_key(|r| r.size).unwrap();
                let large_result = op_results.iter().max_by_key(|r| r.size).unwrap();

                let size_ratio = large_result.size as f64 / small_result.size as f64;
                let time_ratio = large_result.average_time.as_nanos() as f64
                    / small_result.average_time.as_nanos() as f64;

                if time_ratio > size_ratio * 1.5 {
                    println!(
                        "  • {} shows worse than linear scaling - consider optimization",
                        operation
                    );
                } else if time_ratio < size_ratio * 0.8 {
                    println!("  • {} shows good scaling characteristics", operation);
                }
            }
        }

        println!("{}", "=".repeat(80));
    }
}

impl Default for NumArrayBenchmarker {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick benchmark runner for immediate performance feedback
pub fn quick_numarray_benchmark() -> GraphResult<()> {
    let config = NumArrayBenchmarkConfig {
        sizes: vec![1_000, 10_000],
        iterations: 20,
        measure_memory: true,
        include_baseline_comparison: true,
    };

    let benchmarker = NumArrayBenchmarker::with_config(config);
    let results = benchmarker.run_comprehensive_benchmark()?;
    benchmarker.print_results(&results);

    Ok(())
}

/// Focused performance test for a specific operation
pub fn benchmark_specific_operation(
    operation: &str,
    sizes: Vec<usize>,
) -> GraphResult<Vec<OperationResult>> {
    let config = NumArrayBenchmarkConfig {
        sizes,
        iterations: 100,
        measure_memory: false,
        include_baseline_comparison: false,
    };

    let benchmarker = NumArrayBenchmarker::with_config(config);
    let results = benchmarker.run_comprehensive_benchmark()?;

    Ok(results
        .operation_results
        .into_iter()
        .filter(|r| r.operation == operation)
        .collect())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_benchmark() {
        let result = quick_numarray_benchmark();
        assert!(result.is_ok());
    }

    #[test]
    fn test_specific_operation_benchmark() {
        let result = benchmark_specific_operation("sum", vec![100, 1_000]);
        assert!(result.is_ok());
        let results = result.unwrap();
        assert_eq!(results.len(), 2);
        assert!(results.iter().all(|r| r.operation == "sum"));
    }

    #[test]
    fn test_benchmark_configuration() {
        let config = NumArrayBenchmarkConfig {
            sizes: vec![50, 100],
            iterations: 5,
            measure_memory: false,
            include_baseline_comparison: false,
        };

        let benchmarker = NumArrayBenchmarker::with_config(config.clone());
        assert_eq!(benchmarker.config.sizes, config.sizes);
        assert_eq!(benchmarker.config.iterations, config.iterations);
    }
}
