//! Memory Usage Profiler for NumArray Operations
//!
//! This module provides tools to analyze memory allocation patterns
//! in NumArray operations to identify optimization opportunities.

use crate::errors::GraphResult;
use crate::storage::array::NumArray;
use std::alloc::{GlobalAlloc, Layout, System};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;

/// Memory allocation tracker that wraps the system allocator
pub struct TrackingAllocator {
    allocated: AtomicUsize,
    peak_allocated: AtomicUsize,
    allocation_count: AtomicUsize,
}

impl TrackingAllocator {
    pub const fn new() -> Self {
        Self {
            allocated: AtomicUsize::new(0),
            peak_allocated: AtomicUsize::new(0),
            allocation_count: AtomicUsize::new(0),
        }
    }

    pub fn current_allocated(&self) -> usize {
        self.allocated.load(Ordering::Relaxed)
    }

    pub fn peak_allocated(&self) -> usize {
        self.peak_allocated.load(Ordering::Relaxed)
    }

    pub fn allocation_count(&self) -> usize {
        self.allocation_count.load(Ordering::Relaxed)
    }

    pub fn reset(&self) {
        self.allocated.store(0, Ordering::Relaxed);
        self.peak_allocated.store(0, Ordering::Relaxed);
        self.allocation_count.store(0, Ordering::Relaxed);
    }
}

unsafe impl GlobalAlloc for TrackingAllocator {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        let ptr = System.alloc(layout);
        if !ptr.is_null() {
            let current =
                self.allocated.fetch_add(layout.size(), Ordering::Relaxed) + layout.size();
            self.allocation_count.fetch_add(1, Ordering::Relaxed);

            // Update peak if necessary
            let mut peak = self.peak_allocated.load(Ordering::Relaxed);
            while current > peak {
                match self.peak_allocated.compare_exchange_weak(
                    peak,
                    current,
                    Ordering::Relaxed,
                    Ordering::Relaxed,
                ) {
                    Ok(_) => break,
                    Err(new_peak) => peak = new_peak,
                }
            }
        }
        ptr
    }

    unsafe fn dealloc(&self, ptr: *mut u8, layout: Layout) {
        System.dealloc(ptr, layout);
        self.allocated.fetch_sub(layout.size(), Ordering::Relaxed);
    }
}

/// Memory usage snapshot for a specific operation
#[derive(Debug, Clone)]
pub struct MemorySnapshot {
    pub operation: String,
    pub array_size: usize,
    pub initial_memory: usize,
    pub peak_memory: usize,
    pub final_memory: usize,
    pub allocation_count: usize,
    pub memory_efficiency_score: f64,
}

impl MemorySnapshot {
    /// Calculate memory overhead as percentage of data size
    pub fn memory_overhead_percent(&self) -> f64 {
        let data_size = self.array_size * std::mem::size_of::<f64>();
        if data_size > 0 {
            ((self.peak_memory as f64 - data_size as f64) / data_size as f64) * 100.0
        } else {
            0.0
        }
    }

    /// Estimate memory churn (allocations per element)
    pub fn memory_churn(&self) -> f64 {
        if self.array_size > 0 {
            self.allocation_count as f64 / self.array_size as f64
        } else {
            0.0
        }
    }
}

/// Memory profiler for NumArray operations
pub struct NumArrayMemoryProfiler {
    /// Shared allocator tracker (Note: In a real implementation,
    /// we'd need to set this as the global allocator)
    tracker: Arc<TrackingAllocator>,
}

impl NumArrayMemoryProfiler {
    pub fn new() -> Self {
        Self {
            tracker: Arc::new(TrackingAllocator::new()),
        }
    }

    /// Profile memory usage of a NumArray operation
    pub fn profile_operation<F, R>(
        &self,
        operation_name: &str,
        array_size: usize,
        operation: F,
    ) -> (R, MemorySnapshot)
    where
        F: FnOnce() -> R,
    {
        // Reset counters
        self.tracker.reset();

        let initial_memory = self.tracker.current_allocated();
        let initial_count = self.tracker.allocation_count();

        // Run the operation
        let result = operation();

        let peak_memory = self.tracker.peak_allocated();
        let final_memory = self.tracker.current_allocated();
        let final_count = self.tracker.allocation_count();

        // Calculate efficiency score (lower is better - less memory overhead)
        let data_size = array_size * std::mem::size_of::<f64>();
        let efficiency_score = if peak_memory > 0 {
            data_size as f64 / peak_memory as f64
        } else {
            1.0
        };

        let snapshot = MemorySnapshot {
            operation: operation_name.to_string(),
            array_size,
            initial_memory,
            peak_memory,
            final_memory,
            allocation_count: final_count - initial_count,
            memory_efficiency_score: efficiency_score,
        };

        (result, snapshot)
    }

    /// Profile multiple NumArray operations and compare memory usage
    pub fn profile_numarray_operations(
        &self,
        array_size: usize,
    ) -> GraphResult<Vec<MemorySnapshot>> {
        let mut snapshots = Vec::new();

        println!(
            "🔍 Profiling memory usage for NumArray operations (size: {})",
            array_size
        );

        // Generate test data
        let test_data: Vec<f64> = (0..array_size).map(|i| i as f64 * 0.1).collect();

        // Profile array creation
        let (array, creation_snapshot) =
            self.profile_operation("array_creation", array_size, || {
                NumArray::new(test_data.clone())
            });
        snapshots.push(creation_snapshot);

        // Profile statistical operations
        let (_, sum_snapshot) =
            self.profile_operation("sum_calculation", array_size, || array.sum());
        snapshots.push(sum_snapshot);

        let (_, mean_snapshot) =
            self.profile_operation("mean_calculation", array_size, || array.mean());
        snapshots.push(mean_snapshot);

        let (_, std_snapshot) =
            self.profile_operation("std_dev_calculation", array_size, || array.std_dev());
        snapshots.push(std_snapshot);

        // Profile more expensive operations
        let (_, median_snapshot) =
            self.profile_operation("median_calculation", array_size, || array.median());
        snapshots.push(median_snapshot);

        // Profile cloning (memory duplication)
        let (_, clone_snapshot) =
            self.profile_operation("array_cloning", array_size, || array.clone());
        snapshots.push(clone_snapshot);

        // Profile iteration
        let (_, iteration_snapshot) = self.profile_operation("array_iteration", array_size, || {
            let mut sum = 0.0;
            for &val in array.iter() {
                sum += val;
            }
            sum
        });
        snapshots.push(iteration_snapshot);

        Ok(snapshots)
    }

    /// Generate memory usage report
    pub fn generate_memory_report(&self, snapshots: &[MemorySnapshot]) -> String {
        let mut report = String::new();

        report.push_str("# NumArray Memory Usage Analysis Report\n\n");

        if snapshots.is_empty() {
            report.push_str("No memory snapshots available.\n");
            return report;
        }

        let array_size = snapshots[0].array_size;
        let data_size = array_size * std::mem::size_of::<f64>();

        report.push_str(&format!(
            "**Array Size**: {} elements ({} bytes data)\n\n",
            array_size, data_size
        ));

        // Table of results
        report.push_str("## Memory Usage by Operation\n\n");
        report.push_str("| Operation | Peak Memory | Overhead% | Allocations | Efficiency |\n");
        report.push_str("|-----------|-------------|-----------|-------------|------------|\n");

        for snapshot in snapshots {
            report.push_str(&format!(
                "| {} | {} bytes | {:.1}% | {} | {:.3} |\n",
                snapshot.operation,
                snapshot.peak_memory,
                snapshot.memory_overhead_percent(),
                snapshot.allocation_count,
                snapshot.memory_efficiency_score
            ));
        }

        // Analysis section
        report.push_str("\n## Analysis\n\n");

        // Find most memory-intensive operation
        if let Some(max_memory_op) = snapshots.iter().max_by_key(|s| s.peak_memory) {
            report.push_str(&format!(
                "**Most memory-intensive operation**: {} ({} bytes peak)\n\n",
                max_memory_op.operation, max_memory_op.peak_memory
            ));
        }

        // Find operation with most allocations
        if let Some(max_allocs_op) = snapshots.iter().max_by_key(|s| s.allocation_count) {
            report.push_str(&format!(
                "**Most allocation-heavy operation**: {} ({} allocations)\n\n",
                max_allocs_op.operation, max_allocs_op.allocation_count
            ));
        }

        // Calculate average overhead
        let avg_overhead: f64 = snapshots
            .iter()
            .map(|s| s.memory_overhead_percent())
            .sum::<f64>()
            / snapshots.len() as f64;
        report.push_str(&format!(
            "**Average memory overhead**: {:.1}%\n\n",
            avg_overhead
        ));

        // Optimization recommendations
        report.push_str("## Optimization Recommendations\n\n");

        for snapshot in snapshots {
            if snapshot.memory_overhead_percent() > 50.0 {
                report.push_str(&format!(
                    "- **{}**: High memory overhead ({:.1}%) - consider optimizing memory allocations\n",
                    snapshot.operation,
                    snapshot.memory_overhead_percent()
                ));
            }

            if snapshot.allocation_count > array_size / 10 {
                report.push_str(&format!(
                    "- **{}**: High allocation count ({}) - consider reducing temporary allocations\n",
                    snapshot.operation,
                    snapshot.allocation_count
                ));
            }
        }

        report
    }

    /// Print memory analysis results to console
    pub fn print_memory_analysis(&self, snapshots: &[MemorySnapshot]) {
        if snapshots.is_empty() {
            println!("No memory snapshots to analyze.");
            return;
        }

        let array_size = snapshots[0].array_size;
        let data_size = array_size * std::mem::size_of::<f64>();

        println!("\n{}", "=".repeat(70));
        println!("🧠 NUMARRAY MEMORY USAGE ANALYSIS");
        println!("{}", "=".repeat(70));
        println!("Array size: {} elements ({} bytes)", array_size, data_size);

        println!("\n{}", "-".repeat(70));
        println!(
            "{:<20} {:>12} {:>10} {:>12} {:>10}",
            "Operation", "Peak Memory", "Overhead%", "Allocations", "Efficiency"
        );
        println!("{}", "-".repeat(70));

        for snapshot in snapshots {
            println!(
                "{:<20} {:>12} {:>9.1}% {:>12} {:>10.3}",
                snapshot.operation,
                format!("{} B", snapshot.peak_memory),
                snapshot.memory_overhead_percent(),
                snapshot.allocation_count,
                snapshot.memory_efficiency_score
            );
        }

        // Highlight issues
        println!("\n🎯 OPTIMIZATION OPPORTUNITIES:");

        let high_overhead_ops: Vec<_> = snapshots
            .iter()
            .filter(|s| s.memory_overhead_percent() > 50.0)
            .collect();

        let high_allocation_ops: Vec<_> = snapshots
            .iter()
            .filter(|s| s.allocation_count > array_size / 10)
            .collect();

        if !high_overhead_ops.is_empty() {
            println!("  ⚠️  High Memory Overhead (>50%):");
            for op in &high_overhead_ops {
                println!(
                    "     • {} ({:.1}%)",
                    op.operation,
                    op.memory_overhead_percent()
                );
            }
        }

        if !high_allocation_ops.is_empty() {
            println!("  ⚠️  High Allocation Count:");
            for op in &high_allocation_ops {
                println!(
                    "     • {} ({} allocations)",
                    op.operation, op.allocation_count
                );
            }
        }

        if high_overhead_ops.is_empty() && high_allocation_ops.is_empty() {
            println!("  ✅ Memory usage patterns look efficient!");
        }

        println!("{}", "=".repeat(70));
    }
}

impl Default for NumArrayMemoryProfiler {
    fn default() -> Self {
        Self::new()
    }
}

/// Quick memory analysis for immediate feedback
pub fn quick_memory_analysis(array_size: usize) -> GraphResult<()> {
    let profiler = NumArrayMemoryProfiler::new();
    let snapshots = profiler.profile_numarray_operations(array_size)?;
    profiler.print_memory_analysis(&snapshots);
    Ok(())
}

/// Detailed memory analysis with report generation
pub fn detailed_memory_analysis(sizes: Vec<usize>) -> GraphResult<String> {
    let profiler = NumArrayMemoryProfiler::new();
    let mut full_report = String::new();

    full_report.push_str("# Comprehensive NumArray Memory Analysis\n\n");
    full_report.push_str(&format!(
        "Analysis performed on {} different array sizes.\n\n",
        sizes.len()
    ));

    for size in sizes {
        let snapshots = profiler.profile_numarray_operations(size)?;
        let size_report = profiler.generate_memory_report(&snapshots);

        full_report.push_str(&format!("## Array Size: {} elements\n\n", size));
        full_report.push_str(&size_report);
        full_report.push_str("\n---\n\n");
    }

    Ok(full_report)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_memory_profiler_creation() {
        let profiler = NumArrayMemoryProfiler::new();
        assert_eq!(profiler.tracker.current_allocated(), 0);
    }

    #[test]
    fn test_memory_snapshot_calculations() {
        let snapshot = MemorySnapshot {
            operation: "test".to_string(),
            array_size: 1000,
            initial_memory: 8000,
            peak_memory: 12000,
            final_memory: 8000,
            allocation_count: 5,
            memory_efficiency_score: 0.8,
        };

        // Memory overhead should be 50% (4000 bytes overhead on 8000 bytes data)
        assert!((snapshot.memory_overhead_percent() - 50.0).abs() < 0.1);

        // Memory churn should be 0.005 (5 allocations for 1000 elements)
        assert!((snapshot.memory_churn() - 0.005).abs() < 0.001);
    }

    #[test]
    fn test_quick_memory_analysis() {
        let result = quick_memory_analysis(100);
        assert!(result.is_ok());
    }
}
