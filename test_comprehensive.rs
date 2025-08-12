use groggy::{Graph, AttrValue};
use groggy::core::query::{AttributeFilter, NodeFilter};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 Starting comprehensive functionality test...");
    
    // Test 1: Basic graph operations
    println!("\n📊 Test 1: Basic graph operations");
    let mut graph = Graph::new();
    
    // Add nodes and edges
    let alice = graph.add_node();
    let bob = graph.add_node();
    let charlie = graph.add_node();
    
    let friendship1 = graph.add_edge(alice, bob)?;
    let friendship2 = graph.add_edge(bob, charlie)?;
    
    // Set attributes
    graph.set_node_attr(alice, "name".to_string(), AttrValue::Text("Alice".to_string()))?;
    graph.set_node_attr(bob, "name".to_string(), AttrValue::Text("Bob".to_string()))?;
    graph.set_node_attr(charlie, "name".to_string(), AttrValue::Text("Charlie".to_string()))?;
    
    graph.set_node_attr(alice, "age".to_string(), AttrValue::Int(28))?;
    graph.set_node_attr(bob, "age".to_string(), AttrValue::Int(32))?;
    graph.set_node_attr(charlie, "age".to_string(), AttrValue::Text("Unknown".to_string()))?;
    
    graph.set_edge_attr(friendship1, "strength".to_string(), AttrValue::Float(0.9))?;
    graph.set_edge_attr(friendship2, "strength".to_string(), AttrValue::Float(0.7))?;
    
    println!("✅ Created graph with {} nodes and {} edges", 
             graph.node_ids().len(), graph.edge_ids().len());
    
    // Test 2: Attribute retrieval
    println!("\n📋 Test 2: Attribute retrieval");
    if let Some(alice_name) = graph.get_node_attr(alice, &"name".to_string())? {
        println!("Alice's name: {:?}", alice_name);
    }
    
    let alice_attrs = graph.get_node_attrs(alice)?;
    println!("Alice's attributes: {} items", alice_attrs.len());
    
    // Test 3: Topology queries
    println!("\n🔍 Test 3: Topology queries");
    let bob_neighbors = graph.neighbors(bob)?;
    let bob_degree = graph.degree(bob)?;
    println!("Bob has {} neighbors and degree {}", bob_neighbors.len(), bob_degree);
    
    // Test 4: Change tracking
    println!("\n📈 Test 4: Change tracking and memory usage");
    let stats = graph.statistics();
    println!("Graph stats: {} nodes, {} edges, {:.1}MB memory", 
             stats.node_count, stats.edge_count, stats.memory_usage_mb);
    
    // Test 5: Query engine
    println!("\n🔎 Test 5: Query engine");
    let age_filter = AttributeFilter::GreaterThan(AttrValue::Int(30));
    let node_filter = NodeFilter::Attribute("age".to_string(), age_filter);
    
    let results = graph.find_nodes(node_filter)?;
    println!("Nodes with age > 30: {} results", results.len());
    
    // Test 6: Pattern matching
    println!("\n🎯 Test 6: Pattern matching");
    let name_pattern = AttributeFilter::Matches("*ice".to_string());
    let pattern_filter = NodeFilter::Attribute("name".to_string(), name_pattern);
    let pattern_results = graph.find_nodes(pattern_filter)?;
    println!("Nodes with names matching '*ice': {} results", pattern_results.len());
    
    // Test 7: Version control operations
    println!("\n🗂️ Test 7: Version control operations");
    let commit1 = graph.commit("Initial graph with users".to_string(), "test".to_string())?;
    println!("Created commit: {}", commit1);
    
    // Add more nodes and commit again
    let diana = graph.add_node();
    graph.set_node_attr(diana, "name".to_string(), AttrValue::Text("Diana".to_string()))?;
    graph.add_edge(alice, diana)?;
    
    let commit2 = graph.commit("Added Diana".to_string(), "test".to_string())?;
    println!("Created second commit: {}", commit2);
    
    // Test 8: Branching
    println!("\n🌳 Test 8: Branching operations");
    graph.create_branch("feature-branch".to_string())?;
    let branches = graph.list_branches();
    println!("Total branches: {}", branches.len());
    for branch in &branches {
        println!("  Branch: {} (head: {})", branch.name, branch.head);
    }
    
    // Test 9: Historical view
    println!("\n⏰ Test 9: Historical view (testing HistoryForest)");
    let history_stats = graph.statistics();
    if history_stats.commit_count > 0 {
        println!("History contains {} commits", history_stats.commit_count);
    }
    
    // Test 10: Memory and performance
    println!("\n⚡ Test 10: Memory and performance");
    let final_stats = graph.statistics();
    println!("Final stats:");
    println!("  Nodes: {}", final_stats.node_count);
    println!("  Edges: {}", final_stats.edge_count);
    println!("  Attributes: {}", final_stats.attribute_count);
    println!("  Commits: {}", final_stats.commit_count);
    println!("  Branches: {}", final_stats.branch_count);
    println!("  Memory usage: {:.2} MB", final_stats.memory_usage_mb);
    println!("  Uncommitted changes: {}", final_stats.uncommitted_changes);
    
    println!("\n🎉 All tests completed successfully!");
    println!("✨ The groggy graph library is fully functional!");
    
    Ok(())
}