//! NumArray Performance Benchmark Runner
//!
//! This binary provides a command-line interface to run NumArray performance
//! benchmarks and establish baseline performance metrics.
//!
//! Usage: cargo run --bin numarray_benchmark_runner [--quick]

use groggy::storage::array::{
    quick_numarray_benchmark, NumArrayBenchmarkConfig, NumArrayBenchmarker,
};
use std::env;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = env::args().collect();

    println!("🚀 NumArray Performance Benchmark Runner");
    println!("==========================================");

    if args.len() > 1 && args[1] == "--quick" {
        println!("Running quick benchmark...\n");
        quick_numarray_benchmark()?;
    } else {
        println!("Running comprehensive benchmark suite...\n");

        let config = NumArrayBenchmarkConfig {
            sizes: vec![100, 1_000, 10_000, 50_000],
            iterations: 100,
            measure_memory: true,
            include_baseline_comparison: true,
        };

        let benchmarker = NumArrayBenchmarker::with_config(config);
        let results = benchmarker.run_comprehensive_benchmark()?;
        benchmarker.print_results(&results);

        // Generate markdown report
        let report = results.generate_summary();

        // Save report to file
        use std::fs;
        let report_path = "numarray_benchmark_report.md";
        fs::write(report_path, report)?;
        println!("\n📄 Detailed report saved to: {}", report_path);

        // Print key recommendations
        println!("\n🎯 KEY RECOMMENDATIONS:");
        println!("1. Run this benchmark before and after performance optimizations");
        println!("2. Focus optimization efforts on operations taking >1ms for 10K elements");
        println!("3. Consider SIMD optimization for operations with linear scaling");
        println!("4. Use --quick flag for rapid iteration during development");
    }

    println!("\n✅ Benchmark completed successfully!");
    Ok(())
}
