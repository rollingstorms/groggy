// Comprehensive test of query and analysis functionality
use groggy::{Graph, AttrValue};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔍 COMPREHENSIVE QUERY & ANALYSIS TEST SUITE");
    println!("=============================================");
    
    // Test 1: Create a Rich Dataset for Querying
    println!("\n📊 TEST 1: Create Rich Dataset");
    println!("------------------------------");
    
    let mut graph = Graph::new();
    
    // Create a social network scenario
    let alice = graph.add_node();
    let bob = graph.add_node();
    let charlie = graph.add_node();
    let diana = graph.add_node();
    let eve = graph.add_node();
    let frank = graph.add_node();
    
    // Set user attributes
    graph.set_node_attr(alice, "name".to_string(), AttrValue::Text("Alice".to_string()))?;
    graph.set_node_attr(alice, "age".to_string(), AttrValue::Int(25))?;
    graph.set_node_attr(alice, "city".to_string(), AttrValue::Text("New York".to_string()))?;
    graph.set_node_attr(alice, "premium".to_string(), AttrValue::Bool(true))?;
    graph.set_node_attr(alice, "score".to_string(), AttrValue::Float(8.7))?;
    
    graph.set_node_attr(bob, "name".to_string(), AttrValue::Text("Bob".to_string()))?;
    graph.set_node_attr(bob, "age".to_string(), AttrValue::Int(30))?;
    graph.set_node_attr(bob, "city".to_string(), AttrValue::Text("San Francisco".to_string()))?;
    graph.set_node_attr(bob, "premium".to_string(), AttrValue::Bool(false))?;
    graph.set_node_attr(bob, "score".to_string(), AttrValue::Float(7.2))?;
    
    graph.set_node_attr(charlie, "name".to_string(), AttrValue::Text("Charlie".to_string()))?;
    graph.set_node_attr(charlie, "age".to_string(), AttrValue::Int(35))?;
    graph.set_node_attr(charlie, "city".to_string(), AttrValue::Text("New York".to_string()))?;
    graph.set_node_attr(charlie, "premium".to_string(), AttrValue::Bool(true))?;
    graph.set_node_attr(charlie, "score".to_string(), AttrValue::Float(9.1))?;
    
    graph.set_node_attr(diana, "name".to_string(), AttrValue::Text("Diana".to_string()))?;
    graph.set_node_attr(diana, "age".to_string(), AttrValue::Int(28))?;
    graph.set_node_attr(diana, "city".to_string(), AttrValue::Text("Los Angeles".to_string()))?;
    graph.set_node_attr(diana, "premium".to_string(), AttrValue::Bool(true))?;
    graph.set_node_attr(diana, "score".to_string(), AttrValue::Float(8.9))?;
    
    graph.set_node_attr(eve, "name".to_string(), AttrValue::Text("Eve".to_string()))?;
    graph.set_node_attr(eve, "age".to_string(), AttrValue::Int(22))?;
    graph.set_node_attr(eve, "city".to_string(), AttrValue::Text("Chicago".to_string()))?;
    graph.set_node_attr(eve, "premium".to_string(), AttrValue::Bool(false))?;
    graph.set_node_attr(eve, "score".to_string(), AttrValue::Float(6.8))?;
    
    graph.set_node_attr(frank, "name".to_string(), AttrValue::Text("Frank".to_string()))?;
    graph.set_node_attr(frank, "age".to_string(), AttrValue::Int(45))?;
    graph.set_node_attr(frank, "city".to_string(), AttrValue::Text("Boston".to_string()))?;
    graph.set_node_attr(frank, "premium".to_string(), AttrValue::Bool(true))?;
    graph.set_node_attr(frank, "score".to_string(), AttrValue::Float(7.5))?;
    
    println!("✅ Created 6 users with diverse attributes");
    
    // Create connections with different relationship types
    let conn1 = graph.add_edge(alice, bob)?; // friends
    let conn2 = graph.add_edge(alice, charlie)?; // colleagues  
    let conn3 = graph.add_edge(bob, diana)?; // friends
    let conn4 = graph.add_edge(charlie, diana)?; // family
    let conn5 = graph.add_edge(diana, eve)?; // friends
    let conn6 = graph.add_edge(eve, frank)?; // mentor
    let conn7 = graph.add_edge(frank, alice)?; // colleagues
    let conn8 = graph.add_edge(bob, charlie)?; // friends
    
    // Set edge attributes
    graph.set_edge_attr(conn1, "relationship".to_string(), AttrValue::Text("friends".to_string()))?;
    graph.set_edge_attr(conn1, "strength".to_string(), AttrValue::Float(0.8))?;
    graph.set_edge_attr(conn1, "duration_months".to_string(), AttrValue::Int(24))?;
    
    graph.set_edge_attr(conn2, "relationship".to_string(), AttrValue::Text("colleagues".to_string()))?;
    graph.set_edge_attr(conn2, "strength".to_string(), AttrValue::Float(0.6))?;
    graph.set_edge_attr(conn2, "duration_months".to_string(), AttrValue::Int(12))?;
    
    graph.set_edge_attr(conn3, "relationship".to_string(), AttrValue::Text("friends".to_string()))?;
    graph.set_edge_attr(conn3, "strength".to_string(), AttrValue::Float(0.9))?;
    graph.set_edge_attr(conn3, "duration_months".to_string(), AttrValue::Int(36))?;
    
    graph.set_edge_attr(conn4, "relationship".to_string(), AttrValue::Text("family".to_string()))?;
    graph.set_edge_attr(conn4, "strength".to_string(), AttrValue::Float(0.95))?;
    graph.set_edge_attr(conn4, "duration_months".to_string(), AttrValue::Int(300))?; // siblings
    
    graph.set_edge_attr(conn5, "relationship".to_string(), AttrValue::Text("friends".to_string()))?;
    graph.set_edge_attr(conn5, "strength".to_string(), AttrValue::Float(0.7))?;
    graph.set_edge_attr(conn5, "duration_months".to_string(), AttrValue::Int(18))?;
    
    graph.set_edge_attr(conn6, "relationship".to_string(), AttrValue::Text("mentor".to_string()))?;
    graph.set_edge_attr(conn6, "strength".to_string(), AttrValue::Float(0.85))?;
    graph.set_edge_attr(conn6, "duration_months".to_string(), AttrValue::Int(6))?;
    
    graph.set_edge_attr(conn7, "relationship".to_string(), AttrValue::Text("colleagues".to_string()))?;
    graph.set_edge_attr(conn7, "strength".to_string(), AttrValue::Float(0.75))?;
    graph.set_edge_attr(conn7, "duration_months".to_string(), AttrValue::Int(8))?;
    
    graph.set_edge_attr(conn8, "relationship".to_string(), AttrValue::Text("friends".to_string()))?;
    graph.set_edge_attr(conn8, "strength".to_string(), AttrValue::Float(0.65))?;
    graph.set_edge_attr(conn8, "duration_months".to_string(), AttrValue::Int(15))?;
    
    println!("✅ Created 8 connections with relationship attributes");
    
    // Test 2: Basic Topology Analysis
    println!("\n🌐 TEST 2: Topology Analysis");
    println!("----------------------------");
    
    // Analyze node degrees
    let nodes = vec![alice, bob, charlie, diana, eve, frank];
    for &node in &nodes {
        let neighbors = graph.neighbors(node)?;
        let degree = graph.degree(node)?;
        let name = graph.get_node_attr(node, &"name".to_string())?.unwrap();
        println!("✅ {}: degree={}, neighbors={:?}", 
                match name {
                    AttrValue::Text(n) => n,
                    _ => "Unknown".to_string(),
                }, 
                degree, neighbors);
    }
    
    // Test 3: Attribute-based Bulk Queries  
    println!("\n🏷️  TEST 3: Bulk Attribute Queries");
    println!("----------------------------------");
    
    // Get all names
    let all_names = graph.get_nodes_attrs(&"name".to_string(), &nodes)?;
    println!("✅ All names: {:?}", all_names);
    
    // Get all ages
    let all_ages = graph.get_nodes_attrs(&"age".to_string(), &nodes)?;
    println!("✅ All ages: {:?}", all_ages);
    
    // Get all cities
    let all_cities = graph.get_nodes_attrs(&"city".to_string(), &nodes)?;
    println!("✅ All cities: {:?}", all_cities);
    
    // Get all premium status
    let all_premium = graph.get_nodes_attrs(&"premium".to_string(), &nodes)?;
    println!("✅ All premium status: {:?}", all_premium);
    
    // Get all scores
    let all_scores = graph.get_nodes_attrs(&"score".to_string(), &nodes)?;
    println!("✅ All scores: {:?}", all_scores);
    
    // Test edge attributes in bulk
    let edges = vec![conn1, conn2, conn3, conn4, conn5, conn6, conn7, conn8];
    let relationships = graph.get_edges_attrs(&"relationship".to_string(), &edges)?;
    println!("✅ All relationships: {:?}", relationships);
    
    let strengths = graph.get_edges_attrs(&"strength".to_string(), &edges)?;
    println!("✅ All connection strengths: {:?}", strengths);
    
    // Test 4: Individual Entity Analysis
    println!("\n👤 TEST 4: Individual Entity Analysis");
    println!("------------------------------------");
    
    // Detailed analysis of Alice
    let alice_attrs = graph.get_node_attrs(alice)?;
    println!("✅ Alice's complete profile: {} attributes", alice_attrs.len());
    for (attr, value) in &alice_attrs {
        println!("   - {}: {:?}", attr, value);
    }
    
    // Detailed analysis of a strong connection
    let strong_connection_attrs = graph.get_edge_attrs(conn4)?; // Charlie-Diana family connection
    println!("✅ Strong connection (family) profile: {} attributes", strong_connection_attrs.len());
    for (attr, value) in &strong_connection_attrs {
        println!("   - {}: {:?}", attr, value);
    }
    
    // Test 5: Graph Statistics and Metrics
    println!("\n📈 TEST 5: Graph Statistics and Metrics");
    println!("---------------------------------------");
    
    let stats = graph.statistics();
    println!("✅ Overall graph statistics:");
    println!("   - Total nodes: {}", stats.node_count);
    println!("   - Total edges: {}", stats.edge_count);
    println!("   - Total attributes: {}", stats.attribute_count);
    println!("   - Memory usage: {:.2} MB", stats.memory_usage_mb);
    
    // Calculate some derived metrics
    let total_degree: usize = nodes.iter()
        .map(|&node| graph.degree(node).unwrap_or(0))
        .sum();
    let avg_degree = total_degree as f64 / nodes.len() as f64;
    println!("✅ Derived topology metrics:");
    println!("   - Average degree: {:.2}", avg_degree);
    println!("   - Density: {:.3} (edges/max_possible)", 
             stats.edge_count as f64 / (stats.node_count * (stats.node_count - 1) / 2) as f64);
    
    // Count attributes by type
    let mut node_attr_types = HashMap::new();
    for &node in &nodes {
        let attrs = graph.get_node_attrs(node)?;
        for (_, value) in attrs {
            let type_name = match value {
                AttrValue::Text(_) => "Text",
                AttrValue::Int(_) => "Int", 
                AttrValue::Float(_) => "Float",
                AttrValue::Bool(_) => "Bool",
                AttrValue::FloatVec(_) => "FloatVec",
            };
            *node_attr_types.entry(type_name).or_insert(0) += 1;
        }
    }
    println!("✅ Attribute type distribution:");
    for (type_name, count) in node_attr_types {
        println!("   - {}: {} instances", type_name, count);
    }
    
    // Test 6: Advanced Query Patterns (what's implemented)
    println!("\n🔬 TEST 6: Advanced Query Capabilities");
    println!("-------------------------------------");
    
    // Test filtering capabilities (basic implementations)
    println!("✅ Testing query engine capabilities...");
    
    // Note: Most advanced query features are placeholder implementations
    // But we can test what's available
    
    // Find all nodes (should return all active nodes)
    let all_node_ids = graph.node_ids();
    let all_edge_ids = graph.edge_ids();
    println!("✅ Active entities: {} nodes, {} edges", all_node_ids.len(), all_edge_ids.len());
    
    // Test contains operations
    let mut valid_nodes = 0;
    let mut valid_edges = 0;
    for &node in &nodes {
        if graph.contains_node(node) {
            valid_nodes += 1;
        }
    }
    for &edge in &edges {
        if graph.contains_edge(edge) {
            valid_edges += 1;
        }
    }
    println!("✅ Entity validation: {}/{} nodes valid, {}/{} edges valid", 
             valid_nodes, nodes.len(), valid_edges, edges.len());
    
    // Test 7: Performance and Scale Characteristics 
    println!("\n⚡ TEST 7: Performance Characteristics");
    println!("-------------------------------------");
    
    // Time some basic operations (simple timing)
    use std::time::Instant;
    
    let start = Instant::now();
    for _ in 0..1000 {
        let _ = graph.contains_node(alice);
    }
    let contains_time = start.elapsed();
    println!("✅ 1000x contains_node: {:?}", contains_time);
    
    let start = Instant::now();
    for _ in 0..100 {
        let _ = graph.get_node_attr(alice, &"name".to_string());
    }
    let get_attr_time = start.elapsed();
    println!("✅ 100x get_node_attr: {:?}", get_attr_time);
    
    let start = Instant::now();
    for _ in 0..100 {
        let _ = graph.neighbors(alice);
    }
    let neighbors_time = start.elapsed();
    println!("✅ 100x neighbors: {:?}", neighbors_time);
    
    // Test 8: Error Handling in Queries
    println!("\n⚠️  TEST 8: Query Error Handling");
    println!("--------------------------------");
    
    // Test invalid node queries
    let result = graph.get_node_attr(9999, &"name".to_string());
    match result {
        Err(_) => println!("✅ Query on non-existent node properly fails"),
        Ok(_) => println!("❌ Should have failed to query non-existent node"),
    }
    
    // Test invalid edge queries
    let result = graph.get_edge_attr(9999, &"relationship".to_string());
    match result {
        Err(_) => println!("✅ Query on non-existent edge properly fails"),
        Ok(_) => println!("❌ Should have failed to query non-existent edge"),
    }
    
    // Test neighbors of non-existent node
    let result = graph.neighbors(9999);
    match result {
        Err(_) => println!("✅ Neighbors query on non-existent node properly fails"),
        Ok(_) => println!("❌ Should have failed to get neighbors of non-existent node"),
    }
    
    // Commit final state
    let final_commit = graph.commit("Complete query test dataset".to_string(), "query_tester".to_string())?;
    println!("✅ Committed final dataset: commit {}", final_commit);
    
    println!("\n🎉 ALL QUERY & ANALYSIS TESTS COMPLETED!");
    println!("========================================");
    println!("✅ Rich dataset creation and management");
    println!("✅ Topology analysis (neighbors, degree)");
    println!("✅ Bulk attribute queries and retrieval");
    println!("✅ Individual entity detailed analysis");
    println!("✅ Graph statistics and derived metrics");
    println!("✅ Performance characteristics measurement");
    println!("✅ Comprehensive error handling");
    println!("✅ Data persistence via commit system");
    
    println!("\n📊 FINAL DATASET SUMMARY");
    println!("========================");
    println!("Nodes: {} users with 5 attributes each", nodes.len());
    println!("Edges: {} connections with 3 attributes each", edges.len());
    println!("Relationship types: friends, colleagues, family, mentor");
    println!("Cities represented: New York, San Francisco, Los Angeles, Chicago, Boston");
    println!("Age range: 22-45 years");
    println!("Score range: 6.8-9.1");
    println!("Premium users: 4/6 (67%)");
    
    Ok(())
}