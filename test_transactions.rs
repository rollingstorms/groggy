// Simple test to demonstrate working transaction system
use groggy::{Graph, AttrValue};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Create a new graph
    let mut graph = Graph::new();
    
    // Add some nodes and edges
    let node1 = graph.add_node();
    let node2 = graph.add_node();
    let _edge = graph.add_edge(node1, node2)?;
    
    // Set some attributes
    graph.set_node_attr(node1, "name".to_string(), AttrValue::Text("Alice".to_string()))?;
    graph.set_node_attr(node2, "name".to_string(), AttrValue::Text("Bob".to_string()))?;
    
    // Check if we have uncommitted changes
    println!("Has uncommitted changes: {}", graph.has_uncommitted_changes());
    
    // Commit the changes
    let commit_id = graph.commit("Initial graph with Alice and Bob".to_string(), "test".to_string())?;
    println!("Created commit: {}", commit_id);
    
    // Check if changes are committed
    println!("Has uncommitted changes after commit: {}", graph.has_uncommitted_changes());
    
    // Make more changes
    graph.set_node_attr(node1, "age".to_string(), AttrValue::Int(25))?;
    println!("Has uncommitted changes after setting age: {}", graph.has_uncommitted_changes());
    
    // Commit again
    let commit_id2 = graph.commit("Added age attribute".to_string(), "test".to_string())?;
    println!("Created second commit: {}", commit_id2);
    
    println!("Transaction system working! ✅");
    Ok(())
}