//! Examples and demonstrations of the trait-based delegation system
//!
//! This module shows how the delegation architecture enables infinite
//! method chaining and universal operation availability.

use super::error_handling::TryMapOps;
use super::forwarding::ForwardingArray;
use super::traits::{BaseArrayOps, DelegatingIterator, NumArrayOps};
use pyo3::PyResult;

/// Example demonstrating basic trait-based delegation
pub fn example_subgraph_operations() -> PyResult<()> {
    println!("=== Trait-Based Delegation Example ===");

    // This would be real PySubgraph objects in practice
    // For demonstration, we use placeholder logic

    println!("1. Subgraph density calculation via trait:");
    // subgraph.density() -> f64

    println!("2. Connected component check via trait:");
    // subgraph.is_connected() -> bool

    println!("3. Neighborhood expansion via trait:");
    // subgraph.neighborhood(Some(2)) -> PySubgraph

    Ok(())
}

/// Example demonstrating array operations with statistical capabilities
pub fn example_num_array_operations() -> PyResult<()> {
    println!("=== Statistical Array Operations Example ===");

    // Create a numerical array with statistical capabilities
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0];
    let num_array = ForwardingArray::new(data);

    // Basic array operations
    println!("Array length: {}", num_array.len());
    println!("Is empty: {}", num_array.is_empty());
    println!("First element: {:?}", num_array.get(0));

    // Statistical operations
    if let Ok(Some(mean)) = num_array.mean() {
        println!("Mean: {:.2}", mean);
    }

    if let Ok(Some(std_dev)) = num_array.std_dev() {
        println!("Standard deviation: {:.2}", std_dev);
    }

    if let Ok(Some(median)) = num_array.median() {
        println!("Median: {:.2}", median);
    }

    // Array transformations
    if let Ok(doubled) = num_array.multiply(2.0) {
        println!("Doubled array length: {}", doubled.len());
        if let Ok(Some(doubled_mean)) = doubled.mean() {
            println!("Doubled array mean: {:.2}", doubled_mean);
        }
    }

    Ok(())
}

/// Example demonstrating delegating iterator with method chaining
pub fn example_delegating_iterator() -> PyResult<()> {
    println!("=== Delegating Iterator Example ===");

    let numbers = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
    let iter = DelegatingIterator::new(numbers.into_iter());

    // Chain operations on the iterator
    let result: Vec<i32> = iter
        .filter(|x| *x > 3) // Filter > 3
        .map(|x| x * 2) // Double each
        .take(3) // Take first 3
        .collect_vec();

    println!("Chained operations result: {:?}", result);
    // Should print: [8, 10, 12]

    Ok(())
}

/// Example demonstrating error handling with try_map
pub fn example_error_handling() -> PyResult<()> {
    println!("=== Error Handling Example ===");

    let numbers = vec![1, 2, 0, 4, 5]; // 0 will cause division error

    // Try mapping with error collection
    let (successes, errors) = numbers.try_map_collect_errors(|x| {
        if *x == 0 {
            Err("Division by zero")
        } else {
            Ok(10 / x)
        }
    });

    println!("Successful operations: {:?}", successes);
    println!("Number of errors: {}", errors.len());

    Ok(())
}

/// Example showing how complex chaining would work in practice
pub fn example_complex_chaining_pattern() -> PyResult<()> {
    println!("=== Complex Chaining Pattern Example ===");

    // This demonstrates the kind of chaining that becomes possible
    // with the delegation architecture:

    println!("Conceptual chaining pattern:");
    println!("g.connected_components()     // GraphOps -> SubgraphArray");
    println!("  .iter()                    // -> DelegatingIterator<Subgraph>");
    println!("  .neighborhood(Some(2))     // SubgraphOps -> DelegatingIterator<Subgraph>");
    println!("  .table()                   // SubgraphOps -> DelegatingIterator<NodesTable>");
    println!("  .filter(\"age > 25\")        // TableOps -> DelegatingIterator<NodesTable>");
    println!("  .agg(\"mean\")               // TableOps -> DelegatingIterator<BaseTable>");
    println!("  .collect()                 // -> TableArray");
    println!("  .stats()                   // -> NumArray");
    println!("  .mean()                    // NumArrayOps -> f64");

    println!("\nThis creates infinite composability where any valid sequence works!");

    Ok(())
}

/// Demonstration of the delegation system's key benefits
pub fn demonstrate_delegation_benefits() -> PyResult<()> {
    println!("=== Delegation Architecture Benefits ===");

    println!("✅ 1. ZERO ALGORITHM DUPLICATION");
    println!("   - Algorithms implemented once in core types");
    println!("   - Arrays/iterators only forward operations");
    println!("   - Optimized code stays optimized");

    println!("✅ 2. TYPE SAFETY");
    println!("   - Compile-time method availability checking");
    println!("   - No runtime 'method not found' errors");
    println!("   - Clear transformation paths");

    println!("✅ 3. INFINITE COMPOSABILITY");
    println!("   - Any valid sequence of transformations works");
    println!("   - Repository becomes a 'graph of possibilities'");
    println!("   - Easy to discover new patterns");

    println!("✅ 4. PERFORMANCE");
    println!("   - Lazy evaluation in iterators");
    println!("   - Zero-copy where possible");
    println!("   - Parallel processing ready");

    println!("✅ 5. PYTHON ERGONOMICS");
    println!("   - Fluent, chainable API");
    println!("   - Natural method discovery");
    println!("   - Seamless integration with existing code");

    Ok(())
}

/// Utility function to run all examples
pub fn run_all_examples() -> PyResult<()> {
    example_subgraph_operations()?;
    println!();

    example_num_array_operations()?;
    println!();

    example_delegating_iterator()?;
    println!();

    example_error_handling()?;
    println!();

    example_complex_chaining_pattern()?;
    println!();

    demonstrate_delegation_benefits()?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_num_array_operations() {
        let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let array = ForwardingArray::new(data);

        assert_eq!(array.len(), 5);
        assert!(!array.is_empty());
        assert_eq!(array.get(0), Some(&1.0));

        // Test statistical operations
        assert!(array.mean().unwrap().is_some());
        let mean = array.mean().unwrap().unwrap();
        assert!((mean - 3.0).abs() < 0.001);

        assert!(array.sum().unwrap() == 15.0);
    }

    #[test]
    fn test_delegating_iterator() {
        let numbers = vec![1, 2, 3, 4, 5];
        let iter = DelegatingIterator::new(numbers.into_iter());

        let result: Vec<i32> = iter.filter(|x| *x > 2).map(|x| x * 2).collect_vec();

        assert_eq!(result, vec![6, 8, 10]);
    }

    #[test]
    fn test_array_transformations() {
        let data = vec![2.0, 4.0, 6.0];
        let array = ForwardingArray::new(data);

        let doubled = array.multiply(2.0).unwrap();
        assert_eq!(doubled.len(), 3);

        let doubled_mean = doubled.mean().unwrap().unwrap();
        assert!((doubled_mean - 8.0).abs() < 0.001);
    }
}
