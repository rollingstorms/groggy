/// Test CSV column name flexibility fixes
use groggy::storage::table::{BaseTable, NodesTable, EdgesTable};
use groggy::types::AttrValue;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("Testing CSV column name flexibility fixes...");
    
    // Test 1: NodesTable with 'node_ids' (plural) instead of 'node_id' 
    println!("\n=== Test 1: NodesTable with 'node_ids' column ===");
    
    // Create a BaseTable with 'node_ids' column
    let mut columns = HashMap::new();
    let node_ids = vec![AttrValue::Int(0), AttrValue::Int(1), AttrValue::Int(2)];
    columns.insert("node_ids".to_string(), groggy::storage::array::BaseArray::from_attr_values(node_ids));
    
    let base_table = BaseTable::from_columns(columns)?;
    
    // This should now work with our fix
    match NodesTable::from_base_table(base_table) {
        Ok(nodes_table) => {
            println!("✅ SUCCESS: NodesTable accepted 'node_ids' column and renamed it to 'node_id'");
            // Verify the column was renamed
            if nodes_table.base_table().has_column("node_id") {
                println!("✅ Column successfully renamed to 'node_id'");
            } else {
                println!("❌ ERROR: Column was not renamed properly");
            }
        },
        Err(e) => {
            println!("❌ FAILED: {}", e);
            return Err(e.into());
        }
    }
    
    // Test 2: EdgesTable with alternative column names
    println!("\n=== Test 2: EdgesTable with alternative column names ===");
    
    // Create a BaseTable with 'edge_ids', 'src', 'tgt' columns
    let mut columns = HashMap::new();
    let edge_ids = vec![AttrValue::Int(0), AttrValue::Int(1)];
    let sources = vec![AttrValue::Int(0), AttrValue::Int(1)]; 
    let targets = vec![AttrValue::Int(1), AttrValue::Int(2)];
    
    columns.insert("edge_ids".to_string(), groggy::storage::array::BaseArray::from_attr_values(edge_ids));
    columns.insert("src".to_string(), groggy::storage::array::BaseArray::from_attr_values(sources));
    columns.insert("tgt".to_string(), groggy::storage::array::BaseArray::from_attr_values(targets));
    
    let base_table = BaseTable::from_columns(columns)?;
    
    // This should now work with our fix
    match EdgesTable::from_base_table(base_table) {
        Ok(edges_table) => {
            println!("✅ SUCCESS: EdgesTable accepted alternative column names");
            // Verify the columns were renamed
            let base = edges_table.base_table();
            if base.has_column("edge_id") && base.has_column("source") && base.has_column("target") {
                println!("✅ Columns successfully renamed: edge_ids->edge_id, src->source, tgt->target");
            } else {
                println!("❌ ERROR: Columns were not renamed properly");
                println!("  Available columns: {:?}", base.column_names());
            }
        },
        Err(e) => {
            println!("❌ FAILED: {}", e);
            return Err(e.into());
        }
    }
    
    println!("\n🎉 All CSV flexibility tests passed!");
    Ok(())
}