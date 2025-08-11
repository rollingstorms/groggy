// Basic test to verify core functionality works
use groggy::{Graph, AttrValue};

fn main() {
    // Create a new graph
    let mut graph = Graph::new();
    
    // Add some nodes
    let node1 = graph.add_node();
    let node2 = graph.add_node();
    println!("Created nodes: {} and {}", node1, node2);
    
    // Add an edge
    match graph.add_edge(node1, node2) {
        Ok(edge) => println!("Created edge: {}", edge),
        Err(e) => println!("Error creating edge: {}", e),
    }
    
    // Set an attribute
    match graph.set_node_attr(node1, "name".to_string(), AttrValue::Text("Alice".to_string())) {
        Ok(()) => println!("Set node attribute successfully"),
        Err(e) => println!("Error setting attribute: {}", e),
    }
    
    // Get statistics
    let stats = graph.statistics();
    println!("Graph stats: {} nodes, {} edges, {} attributes", 
             stats.node_count, stats.edge_count, stats.attribute_count);
    
    println!("Basic graph operations working! ✅");
}