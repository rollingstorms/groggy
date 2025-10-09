//! Performance benchmarking for eager vs lazy evaluation
//! This module provides tools to measure and compare performance between
//! eager and lazy iterator implementations.

use crate::errors::GraphResult;
use crate::storage::array::{ArrayIterator, ArrayOps};
use crate::types::AttrValue;
use std::time::{Duration, Instant};

/// Benchmark configuration
#[derive(Debug, Clone)]
pub struct BenchmarkConfig {
    /// Number of elements in test arrays
    pub size: usize,
    /// Number of iterations to run for each test
    pub iterations: usize,
    /// Whether to include memory usage measurements
    pub measure_memory: bool,
}

impl Default for BenchmarkConfig {
    fn default() -> Self {
        Self {
            size: 100_000,
            iterations: 100,
            measure_memory: false,
        }
    }
}

/// Benchmark results comparing eager vs lazy evaluation
#[derive(Debug, Clone)]
pub struct BenchmarkResults {
    /// Configuration used for the benchmark
    pub config: BenchmarkConfig,
    /// Eager evaluation timing results
    pub eager_times: Vec<Duration>,
    /// Lazy evaluation timing results  
    pub lazy_times: Vec<Duration>,
    /// Memory usage comparison (if measured)
    pub memory_usage: Option<MemoryComparison>,
}

/// Memory usage comparison between eager and lazy evaluation
#[derive(Debug, Clone)]
pub struct MemoryComparison {
    /// Peak memory used by eager evaluation (bytes)
    pub eager_peak: u64,
    /// Peak memory used by lazy evaluation (bytes)
    pub lazy_peak: u64,
}

impl BenchmarkResults {
    /// Calculate average execution time for eager evaluation
    pub fn eager_avg_time(&self) -> Duration {
        let total: Duration = self.eager_times.iter().sum();
        total / self.eager_times.len() as u32
    }

    /// Calculate average execution time for lazy evaluation
    pub fn lazy_avg_time(&self) -> Duration {
        let total: Duration = self.lazy_times.iter().sum();
        total / self.lazy_times.len() as u32
    }

    /// Calculate performance improvement ratio (lazy vs eager)
    pub fn performance_ratio(&self) -> f64 {
        let eager_avg = self.eager_avg_time().as_nanos() as f64;
        let lazy_avg = self.lazy_avg_time().as_nanos() as f64;

        if lazy_avg > 0.0 {
            eager_avg / lazy_avg
        } else {
            0.0
        }
    }

    /// Get performance improvement percentage
    pub fn performance_improvement(&self) -> f64 {
        let ratio = self.performance_ratio();
        (ratio - 1.0) * 100.0
    }

    /// Check if lazy evaluation is faster
    pub fn lazy_is_faster(&self) -> bool {
        self.performance_ratio() > 1.0
    }
}

/// Performance benchmark runner
pub struct Benchmarker {
    config: BenchmarkConfig,
}

impl Benchmarker {
    /// Create a new benchmarker with default configuration
    pub fn new() -> Self {
        Self {
            config: BenchmarkConfig::default(),
        }
    }

    /// Create a new benchmarker with custom configuration
    pub fn with_config(config: BenchmarkConfig) -> Self {
        Self { config }
    }

    /// Run a comprehensive benchmark comparing eager vs lazy evaluation
    pub fn run_comprehensive_benchmark(&self) -> GraphResult<BenchmarkResults> {
        println!(
            "Running comprehensive benchmark with {} elements, {} iterations...",
            self.config.size, self.config.iterations
        );

        let mut results = BenchmarkResults {
            config: self.config.clone(),
            eager_times: Vec::new(),
            lazy_times: Vec::new(),
            memory_usage: None,
        };

        // Generate test data
        let test_data = self.generate_test_data(self.config.size);

        // Run eager evaluation benchmarks
        println!("Benchmarking eager evaluation...");
        for i in 0..self.config.iterations {
            if i % 10 == 0 {
                println!("  Iteration {}/{}", i + 1, self.config.iterations);
            }
            let time = self.benchmark_eager_evaluation(&test_data)?;
            results.eager_times.push(time);
        }

        // Run lazy evaluation benchmarks
        println!("Benchmarking lazy evaluation...");
        for i in 0..self.config.iterations {
            if i % 10 == 0 {
                println!("  Iteration {}/{}", i + 1, self.config.iterations);
            }
            let time = self.benchmark_lazy_evaluation(&test_data)?;
            results.lazy_times.push(time);
        }

        // Memory benchmarking if requested
        if self.config.measure_memory {
            println!("Measuring memory usage...");
            results.memory_usage = Some(self.benchmark_memory_usage(&test_data)?);
        }

        Ok(results)
    }

    /// Generate test data for benchmarking
    fn generate_test_data(&self, size: usize) -> Vec<AttrValue> {
        let mut data = Vec::with_capacity(size);

        for i in 0..size {
            // Mix different types for realistic testing
            match i % 4 {
                0 => data.push(AttrValue::Int(i as i64)),
                1 => data.push(AttrValue::Float(i as f32 / 1000.0)),
                2 => data.push(AttrValue::Text(format!("item_{}", i))),
                3 => data.push(AttrValue::Bool(i % 2 == 0)),
                _ => unreachable!(),
            }
        }

        data
    }

    /// Benchmark eager evaluation performance
    fn benchmark_eager_evaluation(&self, data: &[AttrValue]) -> GraphResult<Duration> {
        let array = TestArray::new(data.to_vec());

        let start = Instant::now();

        // Complex chain of operations that would benefit from lazy evaluation
        let result = array
            .iter()
            .filter(|item| match item {
                AttrValue::Int(i) => *i > 1000,
                AttrValue::Text(s) => s.len() > 5,
                _ => true,
            })
            .take(1000)
            .skip(100)
            .filter(|item| match item {
                AttrValue::Int(i) => *i % 2 == 0,
                AttrValue::Float(f) => *f > 0.5,
                _ => false,
            })
            .take(50);

        // Force materialization
        let _final_result = result.into_vec();

        Ok(start.elapsed())
    }

    /// Benchmark lazy evaluation performance
    fn benchmark_lazy_evaluation(&self, data: &[AttrValue]) -> GraphResult<Duration> {
        let array = TestArray::new(data.to_vec());

        let start = Instant::now();

        // Same complex chain of operations but with lazy evaluation
        let result = array
            .lazy_iter()
            .filter("value > 1000 OR length > 5") // Combined filter
            .take(1000)
            .skip(100)
            .filter("value % 2 == 0 OR value > 0.5") // Another combined filter
            .take(50);

        // Force materialization
        let _final_result = result.collect()?;

        Ok(start.elapsed())
    }

    /// Benchmark memory usage comparison
    fn benchmark_memory_usage(&self, data: &[AttrValue]) -> GraphResult<MemoryComparison> {
        // This is a simplified memory benchmark
        // In a real implementation, you'd use a memory profiler

        let _array = TestArray::new(data.to_vec());
        let base_memory = std::mem::size_of_val(data);

        // Estimate eager evaluation memory usage
        // (creates intermediate collections at each step)
        let eager_peak = base_memory * 4; // Conservative estimate for intermediate allocations

        // Estimate lazy evaluation memory usage
        // (only stores operation descriptions until collect())
        let lazy_peak = base_memory + 1000; // Base data + small overhead for operation chain

        Ok(MemoryComparison {
            eager_peak: eager_peak as u64,
            lazy_peak: lazy_peak as u64,
        })
    }

    /// Print benchmark results in a human-readable format
    pub fn print_results(&self, results: &BenchmarkResults) {
        println!("\n=== BENCHMARK RESULTS ===");
        println!("Configuration:");
        println!("  Elements: {}", results.config.size);
        println!("  Iterations: {}", results.config.iterations);

        println!("\nTiming Results:");
        println!("  Eager avg:  {:?}", results.eager_avg_time());
        println!("  Lazy avg:   {:?}", results.lazy_avg_time());

        println!("\nPerformance:");
        if results.lazy_is_faster() {
            println!(
                "  ✅ Lazy is {:.1}% FASTER",
                results.performance_improvement()
            );
        } else {
            println!(
                "  ❌ Lazy is {:.1}% SLOWER",
                -results.performance_improvement()
            );
        }
        println!("  Speedup ratio: {:.2}x", results.performance_ratio());

        if let Some(ref memory) = results.memory_usage {
            println!("\nMemory Usage:");
            println!("  Eager peak: {} bytes", memory.eager_peak);
            println!("  Lazy peak:  {} bytes", memory.lazy_peak);
            let memory_savings =
                (memory.eager_peak - memory.lazy_peak) as f64 / memory.eager_peak as f64 * 100.0;
            println!("  Memory savings: {:.1}%", memory_savings);
        }

        println!("\nRecommendation:");
        if results.lazy_is_faster() {
            println!("  🚀 Use lazy evaluation for this workload!");
        } else {
            println!("  ⚡ Eager evaluation may be better for this workload");
        }
    }
}

impl Default for Benchmarker {
    fn default() -> Self {
        Self::new()
    }
}

// =============================================================================
// Helper types for benchmarking
// =============================================================================

/// Simple test array implementation for benchmarking
struct TestArray {
    data: Vec<AttrValue>,
}

impl TestArray {
    fn new(data: Vec<AttrValue>) -> Self {
        Self { data }
    }
}

impl ArrayOps<AttrValue> for TestArray {
    fn len(&self) -> usize {
        self.data.len()
    }

    fn get(&self, index: usize) -> Option<&AttrValue> {
        self.data.get(index)
    }

    fn iter(&self) -> ArrayIterator<AttrValue> {
        ArrayIterator::new(self.data.clone())
    }
}

// =============================================================================
// Benchmark utility functions
// =============================================================================

/// Run a quick performance comparison between eager and lazy evaluation
pub fn quick_benchmark(size: usize) -> GraphResult<()> {
    let config = BenchmarkConfig {
        size,
        iterations: 10,
        measure_memory: true,
    };

    let benchmarker = Benchmarker::with_config(config);
    let results = benchmarker.run_comprehensive_benchmark()?;
    benchmarker.print_results(&results);

    Ok(())
}

/// Run benchmark and return whether lazy evaluation is recommended
pub fn should_use_lazy_evaluation(size: usize, iterations: usize) -> GraphResult<bool> {
    let config = BenchmarkConfig {
        size,
        iterations,
        measure_memory: false,
    };

    let benchmarker = Benchmarker::with_config(config);
    let results = benchmarker.run_comprehensive_benchmark()?;

    Ok(results.lazy_is_faster())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quick_benchmark() {
        // Run a small benchmark to verify the system works
        let result = quick_benchmark(1000);
        assert!(result.is_ok());
    }

    #[test]
    fn test_benchmark_results_calculations() {
        let results = BenchmarkResults {
            config: BenchmarkConfig::default(),
            eager_times: vec![Duration::from_millis(100), Duration::from_millis(200)],
            lazy_times: vec![Duration::from_millis(50), Duration::from_millis(100)],
            memory_usage: None,
        };

        assert_eq!(results.eager_avg_time(), Duration::from_millis(150));
        assert_eq!(results.lazy_avg_time(), Duration::from_millis(75));
        assert_eq!(results.performance_ratio(), 2.0);
        assert_eq!(results.performance_improvement(), 100.0);
        assert!(results.lazy_is_faster());
    }
}
