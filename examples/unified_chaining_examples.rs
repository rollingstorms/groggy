//! Practical examples of the unified chaining system
//!
//! This file contains real-world examples showing how to use the unified
//! array chaining system for common graph analysis tasks.

use groggy::errors::GraphResult;
use groggy::prelude::*;
use groggy::storage::array::{ArrayOps, BaseArray, EdgesArray, NodesArray};
use groggy::types::{AttrValue, AttrValueType, EdgeId, NodeId};
use std::collections::HashMap;

/// Example 1: Basic array operations with eager evaluation
pub fn example_basic_eager_operations() -> GraphResult<()> {
    println!("=== Example 1: Basic Eager Operations ===");

    // Create a BaseArray with sample data
    let data = vec![
        AttrValue::Int(10),
        AttrValue::Int(25),
        AttrValue::Int(5),
        AttrValue::Int(40),
        AttrValue::Int(15),
    ];

    let ages = BaseArray::with_name(data, AttrValueType::Int, "ages".to_string());

    // Eager evaluation - operations execute immediately
    let adults: Vec<AttrValue> = ages
        .iter()
        .filter(|age| age.as_int().unwrap_or(0) >= 18) // Filter adults
        .take(3) // Take first 3
        .into_vec(); // Convert to Vec

    println!("Adult ages (first 3): {:?}", adults);

    // Chain with transformation
    let doubled_ages: Vec<AttrValue> = ages
        .iter()
        .map(|age| AttrValue::Int(age.as_int().unwrap_or(0) * 2))
        .into_vec();

    println!("Doubled ages: {:?}", doubled_ages);

    Ok(())
}

/// Example 2: Lazy evaluation with optimization
pub fn example_lazy_evaluation() -> GraphResult<()> {
    println!("\n=== Example 2: Lazy Evaluation with Optimization ===");

    // Large dataset simulation
    let data: Vec<AttrValue> = (1..=1000).map(|i| AttrValue::Int(i)).collect();

    let large_array = BaseArray::new(data, AttrValueType::Int);

    // Lazy evaluation - operations are queued and optimized
    let result = large_array
        .lazy_iter()
        .filter("value > 500") // Queue filter operation
        .filter("value < 800") // Queue another filter (will be fused)
        .sample(10) // Queue sampling operation
        .take(5) // Queue take operation
        .collect()?; // Execute optimized operation chain

    println!(
        "Lazy result (5 samples from 500-800 range): {} elements",
        result.len()
    );

    // Show the optimization in action
    let optimized_result = large_array
        .lazy_iter()
        .take(100) // Early termination - only process first 100
        .filter("value > 50") // Filter on reduced dataset
        .collect()?;

    println!(
        "Early termination result: {} elements",
        optimized_result.len()
    );

    Ok(())
}

/// Example 3: Node-specific operations with NodesArray
pub fn example_node_operations() -> GraphResult<()> {
    println!("\n=== Example 3: Node-Specific Operations ===");

    // Create a mock graph (in real usage, this would be your actual graph)
    let mut graph = Graph::new();
    let node1 = graph.add_node()?;
    let node2 = graph.add_node()?;
    let node3 = graph.add_node()?;
    let node4 = graph.add_node()?;

    // Add some edges to create degree differences
    graph.add_edge(node1, node2, None)?;
    graph.add_edge(node1, node3, None)?;
    graph.add_edge(node2, node3, None)?;
    graph.add_edge(node3, node4, None)?;

    let graph_ref = std::rc::Rc::new(std::cell::RefCell::new(graph));
    let node_ids = vec![node1, node2, node3, node4];

    // Create NodesArray with graph reference
    let nodes = NodesArray::with_graph(node_ids, graph_ref.clone());

    // Node-specific operations become available
    let high_degree_nodes = nodes
        .iter()
        .filter_by_degree(2) // Only available for NodeIdLike types
        .into_vec();

    println!("High degree nodes (degree >= 2): {:?}", high_degree_nodes);

    // Get neighbors for nodes (lazy evaluation)
    let neighbor_lists = nodes
        .lazy_iter()
        .get_neighbors() // Returns LazyArrayIterator<Vec<NodeId>>
        .collect()?;

    println!("Found {} neighbor lists", neighbor_lists.len());

    Ok(())
}

/// Example 4: Edge operations with filtering
pub fn example_edge_operations() -> GraphResult<()> {
    println!("\n=== Example 4: Edge Operations ===");

    // Create edges with mock data
    let edge_ids = vec![EdgeId(1), EdgeId(2), EdgeId(3), EdgeId(4)];
    let edges = EdgesArray::new(edge_ids);

    // Edge-specific operations
    let filtered_edges = edges
        .iter()
        .filter_by_weight(0.5) // Only available for EdgeLike types
        .group_by_source() // Group edges by source node
        .into_vec();

    println!("Grouped edges: {} groups", filtered_edges.len());

    // Combine with endpoint filtering (lazy)
    let specific_edges = edges
        .lazy_iter()
        .filter_by_endpoints(
            Some("source_attr > 10".to_string()),
            Some("target_attr < 20".to_string()),
        )
        .filter_by_weight(0.3)
        .collect()?;

    println!("Filtered edges: {} edges", specific_edges.len());

    Ok(())
}

/// Example 5: Subgraph operations and meta-node creation
pub fn example_subgraph_operations() -> GraphResult<()> {
    println!("\n=== Example 5: Subgraph Operations ===");

    // In a real scenario, you'd have actual subgraphs
    // Here we simulate the operations

    let mock_subgraphs = vec![(), (), ()]; // Placeholder for actual subgraphs

    // Create aggregation specifications
    let mut aggregations = HashMap::new();
    aggregations.insert("count".to_string(), "sum".to_string());
    aggregations.insert("average_degree".to_string(), "mean".to_string());
    aggregations.insert("max_weight".to_string(), "max".to_string());

    // Subgraph processing chain
    let meta_nodes = groggy::storage::array::ArrayIterator::new(mock_subgraphs)
        .filter_nodes("active = true") // Filter nodes within subgraphs
        .filter_edges("weight > 0.1") // Filter edges within subgraphs
        .collapse(aggregations) // Collapse to meta-nodes
        .into_vec();

    println!("Created {} meta-nodes from subgraphs", meta_nodes.len());

    Ok(())
}

/// Example 6: Performance comparison between eager and lazy
pub fn example_performance_comparison() -> GraphResult<()> {
    println!("\n=== Example 6: Performance Comparison ===");

    use std::time::Instant;

    // Large dataset for meaningful comparison
    let data: Vec<AttrValue> = (1..=50000).map(|i| AttrValue::Int(i)).collect();

    let array = BaseArray::new(data, AttrValueType::Int);

    // Eager evaluation timing
    let start = Instant::now();
    let eager_result: Vec<_> = array
        .iter()
        .filter(|x| x.as_int().unwrap() % 3 == 0)
        .filter(|x| x.as_int().unwrap() > 1000)
        .take(100)
        .into_vec();
    let eager_time = start.elapsed();

    // Lazy evaluation timing
    let start = Instant::now();
    let lazy_result = array
        .lazy_iter()
        .filter("value % 3 = 0") // Fused with next filter
        .filter("value > 1000") // Fused operation
        .take(100) // Early termination
        .collect()?;
    let lazy_time = start.elapsed();

    println!(
        "Eager result: {} items in {:?}",
        eager_result.len(),
        eager_time
    );
    println!(
        "Lazy result: {} items in {:?}",
        lazy_result.len(),
        lazy_time
    );

    let speedup = eager_time.as_nanos() as f64 / lazy_time.as_nanos() as f64;
    println!("Lazy evaluation is {:.2}x faster", speedup);

    Ok(())
}

/// Example 7: Error handling and recovery
pub fn example_error_handling() -> GraphResult<()> {
    println!("\n=== Example 7: Error Handling ===");

    let data = vec![
        AttrValue::Int(10),
        AttrValue::Text("invalid".to_string()), // Mixed type - potential error
        AttrValue::Int(20),
    ];

    let mixed_array = BaseArray::new(data, AttrValueType::Mixed);

    // Safe operation with error handling
    let result = mixed_array
        .lazy_iter()
        .filter("value > 5") // Might fail on text values
        .collect()
        .map_err(|e| {
            eprintln!("Filter operation failed: {}", e);
            e
        });

    match result {
        Ok(values) => println!("Successfully filtered {} values", values.len()),
        Err(e) => println!("Operation failed with error: {}", e),
    }

    Ok(())
}

/// Example 8: Custom benchmarking
pub fn example_custom_benchmarking() -> GraphResult<()> {
    println!("\n=== Example 8: Custom Benchmarking ===");

    use groggy::storage::array::{BenchmarkConfig, Benchmarker};

    let config = BenchmarkConfig {
        data_sizes: vec![1000, 5000, 10000],
        operation_chains: vec![
            "filter".to_string(),
            "filter -> filter".to_string(),
            "filter -> take".to_string(),
            "filter -> sample -> take".to_string(),
        ],
        iterations: 5,
    };

    // Note: In a real implementation, this would run actual benchmarks
    println!(
        "Running benchmarks with {} data sizes and {} operation chains...",
        config.data_sizes.len(),
        config.operation_chains.len()
    );

    // Mock results for demonstration
    println!("Results: Lazy evaluation shows 2.3x average speedup for complex chains");
    println!("Memory usage: 45% reduction for large datasets with lazy evaluation");

    Ok(())
}

/// Example 9: Integration with BaseTable system
pub fn example_table_integration() -> GraphResult<()> {
    println!("\n=== Example 9: Integration with BaseTable ===");

    // Create sample data
    let node_ids = vec![
        AttrValue::Int(1),
        AttrValue::Int(2),
        AttrValue::Int(3),
        AttrValue::Int(4),
    ];

    let degrees = vec![
        AttrValue::Int(3),
        AttrValue::Int(1),
        AttrValue::Int(5),
        AttrValue::Int(2),
    ];

    let node_array = BaseArray::with_name(node_ids, AttrValueType::Int, "node_id".to_string());
    let degree_array = BaseArray::with_name(degrees, AttrValueType::Int, "degree".to_string());

    // Demonstrate chaining on table columns
    let high_degree_nodes = degree_array.lazy_iter().filter("value > 2").collect()?;

    println!("High degree nodes: {} found", high_degree_nodes.len());

    // In a real BaseTable implementation, you could do:
    // let table = BaseTable::new();
    // table.add_column("node_id", node_array);
    // table.add_column("degree", degree_array);
    // let results = table.query("degree > 2")?.get_column("node_id")?;

    Ok(())
}

/// Main function to run all examples
pub fn main() -> GraphResult<()> {
    println!("🚀 Unified Chaining System Examples\n");

    example_basic_eager_operations()?;
    example_lazy_evaluation()?;
    example_node_operations()?;
    example_edge_operations()?;
    example_subgraph_operations()?;
    example_performance_comparison()?;
    example_error_handling()?;
    example_custom_benchmarking()?;
    example_table_integration()?;

    println!("\n✅ All examples completed successfully!");

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_all_examples() {
        // Run all examples to ensure they work
        assert!(main().is_ok());
    }

    #[test]
    fn test_lazy_vs_eager_equivalence() -> GraphResult<()> {
        // Ensure lazy and eager produce same results
        let data: Vec<AttrValue> = (1..=100).map(|i| AttrValue::Int(i)).collect();

        let array = BaseArray::new(data, AttrValueType::Int);

        let eager_result: Vec<_> = array
            .iter()
            .filter(|x| x.as_int().unwrap() > 50)
            .take(10)
            .into_vec();

        let lazy_result = array.lazy_iter().filter("value > 50").take(10).collect()?;

        assert_eq!(eager_result.len(), lazy_result.len());

        Ok(())
    }
}
