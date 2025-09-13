//! Test script for memory profiler functionality

use groggy::storage::array::{quick_memory_analysis, NumArray};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧠 Testing NumArray Memory Profiler functionality...\n");
    
    // Test memory analysis with different sizes
    println!("1. Quick memory analysis (small array):");
    quick_memory_analysis(100)?;
    
    println!("\n2. Quick memory analysis (medium array):");
    quick_memory_analysis(1000)?;
    
    println!("\n✅ Memory profiler tests completed successfully!");
    Ok(())
}