// Enhanced Comprehensive Test Suite for Groggy Graph Library
// Tests all functionality including advanced features and edge cases

use groggy::{Graph, AttrValue};
use std::time::Instant;
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 ENHANCED COMPREHENSIVE GROGGY TEST SUITE");
    println!("============================================");
    
    let mut graph = Graph::new();
    
    // TEST 1: Large-scale Graph Construction
    println!("\n📊 TEST 1: Large-scale Graph Construction");
    println!("------------------------------------------");
    
    let start = Instant::now();
    
    // Create 100 nodes in bulk
    let node_ids = graph.add_nodes(100);
    println!("✅ Created 100 nodes in bulk: {:?}ms", start.elapsed().as_millis());
    
    // Create a dense mesh of edges (every 10th node connected to next 5)
    let mut edge_specs = Vec::new();
    for i in (0..100).step_by(10) {
        for j in 1..=5 {
            if i + j < 100 {
                edge_specs.push((node_ids[i], node_ids[i + j]));
            }
        }
    }
    
    let start = Instant::now();
    let edge_ids = graph.add_edges(&edge_specs);
    println!("✅ Created {} edges in mesh pattern: {:?}ms", edge_ids.len(), start.elapsed().as_millis());
    
    // TEST 2: Complex Attribute Patterns
    println!("\n🏷️  TEST 2: Complex Attribute Patterns");
    println!("--------------------------------------");
    
    // Set different attribute types on different node groups
    let start = Instant::now();
    
    // Group 1: Person nodes (0-29)
    for i in 0..30 {
        let node_id = node_ids[i];
        graph.set_node_attr(node_id, "type".to_string(), AttrValue::Text("person".to_string()))?;
        graph.set_node_attr(node_id, "age".to_string(), AttrValue::Int(20 + (i as i64 % 60)))?;
        graph.set_node_attr(node_id, "name".to_string(), AttrValue::Text(format!("Person_{}", i)))?;
        graph.set_node_attr(node_id, "active".to_string(), AttrValue::Bool(i % 3 == 0))?;
    }
    
    // Group 2: Location nodes (30-59)
    for i in 30..60 {
        let node_id = node_ids[i];
        graph.set_node_attr(node_id, "type".to_string(), AttrValue::Text("location".to_string()))?;
        graph.set_node_attr(node_id, "latitude".to_string(), AttrValue::Float((40.0 + (i as f32 - 30.0) * 0.1)))?;
        graph.set_node_attr(node_id, "longitude".to_string(), AttrValue::Float((-74.0 + (i as f32 - 30.0) * 0.1)))?;
        graph.set_node_attr(node_id, "name".to_string(), AttrValue::Text(format!("Location_{}", i - 30)))?;
    }
    
    // Group 3: Data nodes (60-99) with vector embeddings
    for i in 60..100 {
        let node_id = node_ids[i];
        graph.set_node_attr(node_id, "type".to_string(), AttrValue::Text("data".to_string()))?;
        let embedding: Vec<f32> = (0..10).map(|j| (i * j) as f32 / 100.0).collect();
        graph.set_node_attr(node_id, "embedding".to_string(), AttrValue::FloatVec(embedding))?;
        graph.set_node_attr(node_id, "category".to_string(), AttrValue::Int((i % 5) as i64))?;
    }
    
    println!("✅ Set complex attribute patterns: {:?}ms", start.elapsed().as_millis());
    
    // TEST 3: Edge Attribute Complexity
    println!("\n🔗 TEST 3: Edge Attribute Complexity");
    println!("------------------------------------");
    
    let start = Instant::now();
    for (idx, &edge_id) in edge_ids.iter().enumerate() {
        graph.set_edge_attr(edge_id, "weight".to_string(), AttrValue::Float((1.0 + (idx as f32 % 10.0))))?;
        graph.set_edge_attr(edge_id, "created_at".to_string(), AttrValue::Int(1700000000 + idx as i64))?;
        graph.set_edge_attr(edge_id, "relationship".to_string(), 
            AttrValue::Text(match idx % 4 {
                0 => "friend".to_string(),
                1 => "colleague".to_string(), 
                2 => "family".to_string(),
                _ => "acquaintance".to_string(),
            }))?;
    }
    println!("✅ Set edge attributes on {} edges: {:?}ms", edge_ids.len(), start.elapsed().as_millis());
    
    // TEST 4: Bulk Operations Performance
    println!("\n📦 TEST 4: Bulk Operations Performance");
    println!("--------------------------------------");
    
    // Bulk attribute setting
    let start = Instant::now();
    let mut bulk_attrs = HashMap::new();
    
    // Prepare score attributes for all nodes
    let mut score_data = Vec::new();
    for (i, &node_id) in node_ids.iter().enumerate() {
        score_data.push((node_id, AttrValue::Float(i as f32 * 1.5)));
    }
    bulk_attrs.insert("score".to_string(), score_data);
    
    // Prepare priority attributes for all nodes
    let mut priority_data = Vec::new();
    for (i, &node_id) in node_ids.iter().enumerate() {
        priority_data.push((node_id, AttrValue::Int((i % 10) as i64)));
    }
    bulk_attrs.insert("priority".to_string(), priority_data);
    
    graph.set_node_attrs(bulk_attrs)?;
    println!("✅ Bulk set 2 attributes on 100 nodes: {:?}ms", start.elapsed().as_millis());
    
    // Bulk attribute retrieval (simulated using individual calls)
    let start = Instant::now();
    let mut bulk_names = Vec::new();
    let mut bulk_scores = Vec::new();
    for &node_id in &node_ids {
        bulk_names.push(graph.get_node_attr(node_id, &"name".to_string())?);
        bulk_scores.push(graph.get_node_attr(node_id, &"score".to_string())?);
    }
    println!("✅ Retrieved 200 attribute values: {:?}ms", start.elapsed().as_millis());
    
    // Verify some bulk results
    assert_eq!(bulk_names.len(), 100);
    assert_eq!(bulk_scores.len(), 100);
    if let Some(Some(AttrValue::Text(name))) = bulk_names.get(0) {
        assert_eq!(name, "Person_0");
    }
    if let Some(Some(AttrValue::Float(score))) = bulk_scores.get(50) {
        assert!((score - 75.0).abs() < 0.1);
    }
    
    // TEST 5: Complex Topology Queries
    println!("\n🌐 TEST 5: Complex Topology Queries");
    println!("-----------------------------------");
    
    let start = Instant::now();
    let mut total_degree = 0;
    for &node_id in &node_ids[0..50] {  // Test first 50 nodes
        let neighbors = graph.neighbors(node_id)?;
        let degree = graph.degree(node_id)?;
        total_degree += degree;
        
        // Verify neighbors match degree
        assert_eq!(neighbors.len(), degree);
    }
    println!("✅ Computed neighbors and degrees for 50 nodes: {:?}ms", start.elapsed().as_millis());
    println!("✅ Total degree across 50 nodes: {}", total_degree);
    
    // TEST 6: Transaction and Version Control
    println!("\n💾 TEST 6: Advanced Transaction System");
    println!("--------------------------------------");
    
    // Create multiple commits with meaningful changes
    let commit1 = graph.commit("Initial large graph with 100 nodes and mesh topology".to_string(), "test_suite".to_string())?;
    println!("✅ Created initial commit: {}", commit1);
    
    // Make some changes
    let start = Instant::now();
    for i in 0..20 {
        graph.set_node_attr(node_ids[i], "version".to_string(), AttrValue::Int(2))?;
    }
    println!("✅ Made 20 attribute changes: {:?}ms", start.elapsed().as_millis());
    
    let commit2 = graph.commit("Updated version on first 20 nodes".to_string(), "test_suite".to_string())?;
    println!("✅ Created second commit: {}", commit2);
    
    // Add more nodes and edges
    let extra_nodes = graph.add_nodes(3);
    let extra_edges = graph.add_edges(&[(extra_nodes[0], node_ids[0]), (extra_nodes[1], node_ids[1])]);
    println!("✅ Added {} extra nodes and {} extra edges", extra_nodes.len(), extra_edges.len());
    
    let commit3 = graph.commit("Added extra nodes and connected to main graph".to_string(), "test_suite".to_string())?;
    println!("✅ Created third commit: {}", commit3);
    
    // TEST 7: Memory and Performance Analysis
    println!("\n📊 TEST 7: Memory and Performance Analysis");
    println!("------------------------------------------");
    
    let stats = graph.statistics();
    println!("✅ Final Graph Statistics:");
    println!("   - Nodes: {}", stats.node_count);
    println!("   - Edges: {}", stats.edge_count);
    println!("   - Attributes: {}", stats.attribute_count);
    println!("   - Commits: {}", stats.commit_count);
    println!("   - Branches: {}", stats.branch_count);
    println!("   - Memory Usage: {:.2} MB", stats.memory_usage_mb);
    
    // TEST 8: Data Integrity Verification
    println!("\n🔍 TEST 8: Data Integrity Verification");
    println!("--------------------------------------");
    
    let start = Instant::now();
    
    // Verify all nodes exist
    for &node_id in &node_ids {
        assert!(graph.contains_node(node_id), "Node {} should exist", node_id);
    }
    
    // Verify all edges exist
    for &edge_id in &edge_ids {
        assert!(graph.contains_edge(edge_id), "Edge {} should exist", edge_id);
    }
    
    // Verify specific attributes on different node types
    let person_node = node_ids[5];
    if let Some(AttrValue::Text(node_type)) = graph.get_node_attr(person_node, &"type".to_string())? {
        assert_eq!(node_type, "person");
    }
    
    let location_node = node_ids[35];
    if let Some(attr_val) = graph.get_node_attr(location_node, &"latitude".to_string())? {
        if let AttrValue::Float(lat) = attr_val {
            assert!(lat > 40.0 && lat < 45.0);
        }
    }
    
    let data_node = node_ids[65];
    if let Some(AttrValue::FloatVec(embedding)) = graph.get_node_attr(data_node, &"embedding".to_string())? {
        assert_eq!(embedding.len(), 10);
    }
    
    println!("✅ Data integrity verification passed: {:?}ms", start.elapsed().as_millis());
    
    // TEST 9: Edge Case Handling
    println!("\n⚠️  TEST 9: Edge Case Handling");
    println!("-----------------------------");
    
    // Test operations on non-existent entities
    let non_existent_node = 999999;
    let non_existent_edge = 999999;
    
    assert!(!graph.contains_node(non_existent_node));
    assert!(!graph.contains_edge(non_existent_edge));
    
    // Test getting attributes that don't exist
    let result = graph.get_node_attr(node_ids[0], &"nonexistent_attr".to_string())?;
    assert!(result.is_none());
    
    // Test operations with mixed valid/invalid nodes
    let mixed_nodes = vec![node_ids[0], 999999, node_ids[1]];
    let mut bulk_result = Vec::new();
    for &node_id in &mixed_nodes {
        bulk_result.push(graph.get_node_attr(node_id, &"name".to_string()).unwrap_or(None));
    }
    assert_eq!(bulk_result.len(), 3);
    assert!(bulk_result[0].is_some());  // valid node
    assert!(bulk_result[1].is_none());  // invalid node
    assert!(bulk_result[2].is_some());  // valid node
    
    println!("✅ Edge case handling works correctly");
    
    // FINAL SUMMARY
    println!("\n🎉 ENHANCED COMPREHENSIVE TEST RESULTS");
    println!("======================================");
    
    let final_stats = graph.statistics();
    println!("Final State:");
    println!("  🏠 Nodes: {} (including {} extra)", final_stats.node_count, extra_nodes.len());
    println!("  🔗 Edges: {} (including {} extra)", final_stats.edge_count, extra_edges.len());
    println!("  🏷️  Attributes: {}", final_stats.attribute_count);
    println!("  💾 Commits: {}", final_stats.commit_count);
    println!("  🌿 Branches: {}", final_stats.branch_count);
    println!("  💾 Memory: {:.2} MB", final_stats.memory_usage_mb);
    println!("  ⚡ Uncommitted Changes: {}", final_stats.uncommitted_changes);
    
    println!("\n✨ ALL ENHANCED TESTS PASSED!");
    println!("   Graph library demonstrates robust performance");
    println!("   and correctness across all operations!");
    
    Ok(())
}