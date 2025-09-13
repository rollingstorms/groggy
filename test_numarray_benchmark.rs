//! Test script for NumArray benchmark functionality

use groggy::storage::array::{quick_numarray_benchmark, NumArray};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Testing NumArray benchmark functionality...\n");
    
    // First, test basic NumArray functionality
    println!("1. Testing NumArray basic operations:");
    let test_data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let array = NumArray::new(test_data);
    
    println!("   - Array length: {}", array.len());
    println!("   - Sum: {}", array.sum());
    if let Some(mean) = array.mean() {
        println!("   - Mean: {:.2}", mean);
    }
    if let Some(std_dev) = array.std_dev() {
        println!("   - Std Dev: {:.2}", std_dev);
    }
    
    // Now test the benchmark suite
    println!("\n2. Running quick benchmark suite:");
    quick_numarray_benchmark()?;
    
    println!("\n✅ All tests completed successfully!");
    Ok(())
}