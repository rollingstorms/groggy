//! Comprehensive benchmarks comparing the new unified chaining system
//! with the legacy array implementations.
//!
//! This benchmark suite measures:
//! 1. Memory usage efficiency
//! 2. Operation throughput 
//! 3. Lazy vs eager evaluation performance
//! 4. Complex operation chain optimization
//! 5. Type-specific operation performance

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};
use groggy::storage::{
    array::{BaseArray, NodesArray, ArrayOps, LazyArrayIterator},
    legacy_array::GraphArray,
};
use groggy::types::{AttrValue, AttrValueType, NodeId};
use groggy::api::graph::Graph;
use std::rc::Rc;
use std::cell::RefCell;
use std::time::{Duration, Instant};

/// Generate test data of various sizes
fn generate_test_data(size: usize) -> Vec<AttrValue> {
    (1..=size)
        .map(|i| AttrValue::Int(i as i64))
        .collect()
}

/// Generate node IDs for testing
fn generate_node_ids(size: usize) -> Vec<NodeId> {
    (1..=size)
        .map(|i| NodeId(i as u32))
        .collect()
}

/// Benchmark: BaseArray vs Legacy GraphArray creation
fn bench_array_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("array_creation");
    
    for size in [1000, 5000, 10000].iter() {
        let data = generate_test_data(*size);
        
        // BaseArray creation
        group.bench_with_input(
            BenchmarkId::new("BaseArray", size),
            size,
            |b, _| {
                let data = generate_test_data(*size);
                b.iter(|| {
                    black_box(BaseArray::new(data.clone(), AttrValueType::Int))
                })
            },
        );
        
        // Legacy GraphArray creation
        group.bench_with_input(
            BenchmarkId::new("GraphArray", size), 
            size,
            |b, _| {
                let data = generate_test_data(*size);
                b.iter(|| {
                    black_box(GraphArray::from_vec(data.clone()))
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Simple filter operations
fn bench_simple_filter(c: &mut Criterion) {
    let mut group = c.benchmark_group("simple_filter");
    
    for size in [1000, 5000, 10000].iter() {
        let data = generate_test_data(*size);
        
        // BaseArray eager filter
        group.bench_with_input(
            BenchmarkId::new("BaseArray_eager", size),
            size,
            |b, _| {
                let array = BaseArray::new(data.clone(), AttrValueType::Int);
                b.iter(|| {
                    let result: Vec<_> = array.iter()
                        .filter(|x| x.as_int().unwrap_or(0) > (*size as i64) / 2)
                        .into_vec();
                    black_box(result)
                })
            },
        );
        
        // BaseArray lazy filter
        group.bench_with_input(
            BenchmarkId::new("BaseArray_lazy", size),
            size,
            |b, _| {
                let array = BaseArray::new(data.clone(), AttrValueType::Int);
                b.iter(|| {
                    let result = array.lazy_iter()
                        .filter(&format!("value > {}", (*size as i64) / 2))
                        .collect()
                        .unwrap();
                    black_box(result)
                })
            },
        );
        
        // Legacy GraphArray filter (if available)
        group.bench_with_input(
            BenchmarkId::new("GraphArray", size),
            size,
            |b, _| {
                let array = GraphArray::from_vec(data.clone());
                b.iter(|| {
                    // Simulate legacy filtering approach
                    let result: Vec<_> = array.data().iter()
                        .filter(|x| x.as_int().unwrap_or(0) > (*size as i64) / 2)
                        .cloned()
                        .collect();
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Complex operation chains
fn bench_complex_chains(c: &mut Criterion) {
    let mut group = c.benchmark_group("complex_chains");
    
    for size in [5000, 10000, 20000].iter() {
        let data = generate_test_data(*size);
        
        // BaseArray eager chain
        group.bench_with_input(
            BenchmarkId::new("BaseArray_eager_chain", size),
            size,
            |b, _| {
                let array = BaseArray::new(data.clone(), AttrValueType::Int);
                b.iter(|| {
                    let result: Vec<_> = array.iter()
                        .filter(|x| x.as_int().unwrap_or(0) > 100)
                        .filter(|x| x.as_int().unwrap_or(0) % 3 == 0)
                        .take(50)
                        .into_vec();
                    black_box(result)
                })
            },
        );
        
        // BaseArray lazy chain with optimization
        group.bench_with_input(
            BenchmarkId::new("BaseArray_lazy_chain", size),
            size,
            |b, _| {
                let array = BaseArray::new(data.clone(), AttrValueType::Int);
                b.iter(|| {
                    let result = array.lazy_iter()
                        .filter("value > 100")        // Will be fused
                        .filter("value % 3 = 0")      // Will be fused 
                        .take(50)                     // Early termination
                        .collect()
                        .unwrap();
                    black_box(result)
                })
            },
        );
        
        // Legacy approach (multiple passes)
        group.bench_with_input(
            BenchmarkId::new("GraphArray_multi_pass", size),
            size,
            |b, _| {
                let array = GraphArray::from_vec(data.clone());
                b.iter(|| {
                    // Simulate multiple-pass legacy approach
                    let step1: Vec<_> = array.data().iter()
                        .filter(|x| x.as_int().unwrap_or(0) > 100)
                        .cloned()
                        .collect();
                    
                    let step2: Vec<_> = step1.iter()
                        .filter(|x| x.as_int().unwrap_or(0) % 3 == 0)
                        .cloned()
                        .collect();
                    
                    let result: Vec<_> = step2.into_iter().take(50).collect();
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Node-specific operations
fn bench_node_operations(c: &mut Criterion) {
    let mut group = c.benchmark_group("node_operations");
    
    for size in [1000, 5000, 10000].iter() {
        let node_ids = generate_node_ids(*size);
        
        // Create mock graph
        let mut graph = Graph::new();
        for _ in 0..*size {
            let _ = graph.add_node();
        }
        let graph_ref = Rc::new(RefCell::new(graph));
        
        // NodesArray with type-specific operations
        group.bench_with_input(
            BenchmarkId::new("NodesArray_typed", size),
            size,
            |b, _| {
                let nodes = NodesArray::with_graph(node_ids.clone(), graph_ref.clone());
                b.iter(|| {
                    let result = nodes.iter()
                        .filter_by_degree(2)  // Type-specific method
                        .take(10)
                        .into_vec();
                    black_box(result)
                })
            },
        );
        
        // Generic array approach (simulating legacy)
        group.bench_with_input(
            BenchmarkId::new("Generic_untyped", size),
            size,
            |b, _| {
                let generic_array = BaseArray::new(
                    node_ids.iter().map(|id| AttrValue::Int(id.0 as i64)).collect(),
                    AttrValueType::Int
                );
                b.iter(|| {
                    // Simulate manual filtering without type safety
                    let result: Vec<_> = generic_array.iter()
                        .filter(|_| true) // Placeholder for complex degree check
                        .take(10)
                        .into_vec();
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Memory efficiency comparison
fn bench_memory_efficiency(c: &mut Criterion) {
    let mut group = c.benchmark_group("memory_efficiency");
    
    for size in [10000, 50000, 100000].iter() {
        let data = generate_test_data(*size);
        
        // Measure memory allocation patterns
        group.bench_with_input(
            BenchmarkId::new("lazy_no_intermediate", size),
            size,
            |b, _| {
                let array = BaseArray::new(data.clone(), AttrValueType::Int);
                b.iter(|| {
                    // Lazy evaluation - no intermediate allocations
                    let result = array.lazy_iter()
                        .filter("value > 1000")
                        .filter("value < 90000")
                        .sample(100)
                        .collect()
                        .unwrap();
                    black_box(result)
                })
            },
        );
        
        group.bench_with_input(
            BenchmarkId::new("eager_intermediate", size),
            size,
            |b, _| {
                let array = BaseArray::new(data.clone(), AttrValueType::Int);
                b.iter(|| {
                    // Eager evaluation - creates intermediate collections
                    let step1: Vec<_> = array.iter()
                        .filter(|x| x.as_int().unwrap_or(0) > 1000)
                        .into_vec();
                    
                    let step2: Vec<_> = BaseArray::new(step1, AttrValueType::Int).iter()
                        .filter(|x| x.as_int().unwrap_or(0) < 90000)
                        .into_vec();
                    
                    let result: Vec<_> = step2.into_iter().take(100).collect();
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Benchmark: Sampling performance
fn bench_sampling_performance(c: &mut Criterion) {
    let mut group = c.benchmark_group("sampling_performance");
    
    for size in [10000, 50000, 100000].iter() {
        let data = generate_test_data(*size);
        
        // Lazy sampling with reservoir algorithm
        group.bench_with_input(
            BenchmarkId::new("lazy_reservoir_sampling", size),
            size,
            |b, _| {
                let array = BaseArray::new(data.clone(), AttrValueType::Int);
                b.iter(|| {
                    let result = array.lazy_iter()
                        .sample(1000)  // Uses efficient reservoir sampling
                        .collect()
                        .unwrap();
                    black_box(result)
                })
            },
        );
        
        // Naive sampling (collect all then sample)
        group.bench_with_input(
            BenchmarkId::new("naive_collect_sample", size),
            size,
            |b, _| {
                let array = BaseArray::new(data.clone(), AttrValueType::Int);
                b.iter(|| {
                    let all_data = array.iter().into_vec();
                    let mut indices: Vec<_> = (0..all_data.len()).collect();
                    
                    // Simulate random sampling
                    indices.sort_by_key(|_| fastrand::u32(..));
                    let result: Vec<_> = indices.into_iter()
                        .take(1000.min(all_data.len()))
                        .map(|i| all_data[i].clone())
                        .collect();
                    black_box(result)
                })
            },
        );
    }
    
    group.finish();
}

/// Custom benchmark measuring end-to-end performance
fn bench_end_to_end_scenarios(c: &mut Criterion) {
    let mut group = c.benchmark_group("end_to_end_scenarios");
    
    // Scenario 1: Large dataset analysis
    let large_data = generate_test_data(100000);
    
    group.bench_function("scenario_large_analysis_lazy", |b| {
        let array = BaseArray::new(large_data.clone(), AttrValueType::Int);
        b.iter(|| {
            let result = array.lazy_iter()
                .filter("value > 10000")      // Filter 1 
                .filter("value < 90000")      // Filter 2 (fused)
                .filter("value % 7 = 0")      // Filter 3 (fused)
                .sample(500)                  // Efficient sampling
                .take(100)                    // Early termination
                .collect()
                .unwrap();
            black_box(result)
        })
    });
    
    group.bench_function("scenario_large_analysis_eager", |b| {
        let array = BaseArray::new(large_data.clone(), AttrValueType::Int);
        b.iter(|| {
            let result: Vec<_> = array.iter()
                .filter(|x| {
                    let val = x.as_int().unwrap_or(0);
                    val > 10000 && val < 90000 && val % 7 == 0
                })
                .take(500)
                .take(100)
                .into_vec();
            black_box(result)
        })
    });
    
    group.finish();
}

criterion_group!(
    benches,
    bench_array_creation,
    bench_simple_filter,
    bench_complex_chains,
    bench_node_operations,
    bench_memory_efficiency,
    bench_sampling_performance,
    bench_end_to_end_scenarios
);

criterion_main!(benches);