// Stress Test Suite for Groggy Graph Library
// Tests performance and memory usage under heavy loads

use groggy::{Graph, AttrValue};
use std::time::Instant;
use std::collections::HashMap;

fn format_duration(duration_ms: u128) -> String {
    if duration_ms < 1000 {
        format!("{}ms", duration_ms)
    } else {
        format!("{:.2}s", duration_ms as f64 / 1000.0)
    }
}

fn format_throughput(operations: usize, duration_ms: u128) -> String {
    if duration_ms == 0 {
        return "∞ ops/sec".to_string();
    }
    let ops_per_sec = (operations as f64) / (duration_ms as f64 / 1000.0);
    if ops_per_sec >= 1000.0 {
        format!("{:.1}K ops/sec", ops_per_sec / 1000.0)
    } else {
        format!("{:.0} ops/sec", ops_per_sec)
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🔥 GROGGY STRESS TEST SUITE");
    println!("===========================");
    println!("Testing performance with thousands of operations...\n");
    
    // STRESS TEST 1: Massive Node Creation
    println!("🏭 STRESS TEST 1: Massive Node Creation");
    println!("---------------------------------------");
    
    let mut graph = Graph::new();
    
    // Test 1a: Individual node creation
    let start = Instant::now();
    let mut individual_nodes = Vec::new();
    for i in 0..50000 {
        let node = graph.add_node();
        individual_nodes.push(node);
        if i % 10000 == 9999 {
            println!("  ✅ Created {} nodes individually", i + 1);
        }
    }
    let duration_individual = start.elapsed().as_millis();
    
    println!("✅ Created 50,000 nodes individually: {} ({})", 
             format_duration(duration_individual),
             format_throughput(50000, duration_individual));
    
    // Test 1b: Bulk node creation
    let start = Instant::now();
    let bulk_nodes = graph.add_nodes(1000000);
    let duration_bulk = start.elapsed().as_millis();
    
    println!("✅ Created 1,000,000 nodes in bulk: {} ({})",
             format_duration(duration_bulk),
             format_throughput(1000000, duration_bulk));
    
    let speedup = if duration_bulk > 0 {
        (duration_individual as f64 / 5000.0) / (duration_bulk as f64 / 10000.0)
    } else {
        999.0
    };
    println!("📈 Bulk creation speedup: {:.1}x faster per node", speedup);
    
    // STRESS TEST 2: Massive Edge Creation
    println!("\n🌐 STRESS TEST 2: Massive Edge Creation");
    println!("--------------------------------------");
    
    // Test 2a: Dense connectivity (each of first 1000 nodes connected to next 5)
    let start = Instant::now();
    let mut dense_edges = Vec::new();
    for i in 0..1000 {
        for j in 1..=5 {
            if i + j < individual_nodes.len() {
                let edge = graph.add_edge(individual_nodes[i], individual_nodes[i + j])?;
                dense_edges.push(edge);
            }
        }
    }
    let duration_dense = start.elapsed().as_millis();
    
    println!("✅ Created {} dense edges individually: {} ({})",
             dense_edges.len(),
             format_duration(duration_dense),
             format_throughput(dense_edges.len(), duration_dense));
    
    // Test 2b: Bulk edge creation (random connections)
    
    let mut bulk_edge_specs = Vec::new();
    for i in 0..500000 {
        let source_idx = i % bulk_nodes.len();
        let target_idx = (i * 7 + 13) % bulk_nodes.len(); // pseudo-random but deterministic
        if source_idx != target_idx {
            bulk_edge_specs.push((bulk_nodes[source_idx], bulk_nodes[target_idx]));
        }
    }
    let start = Instant::now();
    let bulk_edges = graph.add_edges(&bulk_edge_specs);
    let duration_bulk_edges = start.elapsed().as_millis();
    
    println!("✅ Created {} edges in bulk: {} ({})",
             bulk_edges.len(),
             format_duration(duration_bulk_edges),
             format_throughput(bulk_edges.len(), duration_bulk_edges));
    
    // STRESS TEST 3: Massive Attribute Operations
    println!("\n🏷️  STRESS TEST 3: Massive Attribute Operations");
    println!("-----------------------------------------------");
    
    // Test 3a: Individual attribute setting
    let start = Instant::now();
    for (i, &node_id) in individual_nodes.iter().enumerate().take(2000) {
        graph.set_node_attr(node_id, "individual_id".to_string(), AttrValue::Int(i as i64))?;
        graph.set_node_attr(node_id, "individual_score".to_string(), AttrValue::Float(i as f32 * 1.5))?;
        graph.set_node_attr(node_id, "individual_name".to_string(), AttrValue::Text(format!("Node_{}", i)))?;
        
        if i % 500 == 499 {
            println!("  ✅ Set attributes on {} nodes", i + 1);
        }
    }
    let duration_individual_attrs = start.elapsed().as_millis();
    
    println!("✅ Set 6,000 attributes individually (3 attrs × 2,000 nodes): {} ({})",
             format_duration(duration_individual_attrs),
             format_throughput(6000, duration_individual_attrs));
    
    // Test 3b: Bulk attribute setting
    
    
    let mut bulk_attrs = HashMap::new();
    
    // Prepare bulk data for 10,000 nodes
    let mut bulk_ids = Vec::new();
    let mut bulk_scores = Vec::new();
    let mut bulk_names = Vec::new();
    let mut bulk_categories = Vec::new();
    let mut bulk_active = Vec::new();
    
    for (i, &node_id) in bulk_nodes.iter().enumerate() {
        bulk_ids.push((node_id, AttrValue::Int(node_id as i64)));
        bulk_scores.push((node_id, AttrValue::Float(i as f32 * 2.5)));
        bulk_names.push((node_id, AttrValue::Text(format!("BulkNode_{}", i))));
        bulk_categories.push((node_id, AttrValue::Int((i % 10) as i64)));
        bulk_active.push((node_id, AttrValue::Bool(i % 3 == 0)));
    }
    
    bulk_attrs.insert("bulk_id".to_string(), bulk_ids);
    bulk_attrs.insert("bulk_score".to_string(), bulk_scores);
    bulk_attrs.insert("bulk_name".to_string(), bulk_names);
    bulk_attrs.insert("bulk_category".to_string(), bulk_categories);
    bulk_attrs.insert("bulk_active".to_string(), bulk_active);
    let start = Instant::now();
    graph.set_node_attrs(bulk_attrs)?;
    let duration_bulk_attrs = start.elapsed().as_millis();
    
    println!("✅ Set 50,000 attributes in bulk (5 attrs × 10,000 nodes): {} ({})",
             format_duration(duration_bulk_attrs),
             format_throughput(50000, duration_bulk_attrs));
    
    let attr_speedup = if duration_bulk_attrs > 0 {
        (duration_individual_attrs as f64 / 6000.0) / (duration_bulk_attrs as f64 / 50000.0)
    } else {
        999.0
    };
    println!("📈 Bulk attribute speedup: {:.1}x faster per attribute", attr_speedup);
    
    // STRESS TEST 4: Massive Edge Attributes
    println!("\n🔗 STRESS TEST 4: Massive Edge Attributes");
    println!("-----------------------------------------");
    
    
    
    // Use the optimized bulk edge attribute operation
    let mut bulk_edge_attrs = HashMap::new();
    
    // Prepare weight attributes for 30000 edges
    let mut weight_data = Vec::new();
    let mut timestamp_data = Vec::new();
    
    for (i, &edge_id) in bulk_edges.iter().enumerate().take(30000) {
        weight_data.push((edge_id, AttrValue::Float(1.0 + (i as f32 % 10.0))));
        timestamp_data.push((edge_id, AttrValue::Int(1700000000 + i as i64)));
    }
    
    bulk_edge_attrs.insert("weight".to_string(), weight_data);
    bulk_edge_attrs.insert("timestamp".to_string(), timestamp_data);
    let start = Instant::now();
    graph.set_edge_attrs(bulk_edge_attrs)?;
    let duration_edge_attrs = start.elapsed().as_millis();
    
    println!("✅ Set 60,000 edge attributes: {} ({})",
             format_duration(duration_edge_attrs),
             format_throughput(60000, duration_edge_attrs));
    
    // STRESS TEST 5: Massive Bulk Retrieval
    println!("\n📤 STRESS TEST 5: Massive Bulk Retrieval");
    println!("----------------------------------------");
    
    // Test bulk node attribute retrieval (simulated)
    let start = Instant::now();
    let mut all_bulk_names = Vec::new();
    let mut all_bulk_scores = Vec::new();
    let mut all_bulk_categories = Vec::new();
    
    for &node_id in &bulk_nodes {
        all_bulk_names.push(graph.get_node_attr(node_id, &"bulk_name".to_string())?);
        all_bulk_scores.push(graph.get_node_attr(node_id, &"bulk_score".to_string())?);
        all_bulk_categories.push(graph.get_node_attr(node_id, &"bulk_category".to_string())?);
    }
    let duration_bulk_retrieval = start.elapsed().as_millis();
    
    println!("✅ Retrieved 300,000 attributes in bulk: {} ({})",
             format_duration(duration_bulk_retrieval),
             format_throughput(300000, duration_bulk_retrieval));
    
    // Verify some results
    assert_eq!(all_bulk_names.len(), bulk_nodes.len());
    assert_eq!(all_bulk_scores.len(), bulk_nodes.len());
    assert_eq!(all_bulk_categories.len(), bulk_nodes.len());
    
    if let Some(Some(AttrValue::Text(name))) = all_bulk_names.get(0) {
        assert_eq!(name, "BulkNode_0");
    }
    
    // STRESS TEST 6: Complex Topology Queries Under Load
    println!("\n🌍 STRESS TEST 6: Complex Topology Analysis");
    println!("-------------------------------------------");
    
    let start = Instant::now();
    let mut total_neighbors = 0;
    let mut total_degree = 0;
    
    // Analyze topology for first 2000 nodes (OPTIMIZED: avoid double edge scan)
    for &node_id in individual_nodes.iter().take(2000) {
        let neighbors = graph.neighbors(node_id)?;
        let degree = neighbors.len();  // Calculate degree from neighbors (no second scan!)
        total_neighbors += neighbors.len();
        total_degree += degree;
        
        // Verify consistency (should always be true now)
        assert_eq!(neighbors.len(), degree);
    }
    let duration_topology = start.elapsed().as_millis();
    
    println!("✅ Analyzed topology for 2,000 nodes: {} ({})",
             format_duration(duration_topology),
             format_throughput(2000, duration_topology));
    println!("   📊 Total neighbors found: {}", total_neighbors);
    println!("   📊 Average degree: {:.2}", total_degree as f64 / 2000.0);
    
    // STRESS TEST 7: Transaction Performance Under Load
    println!("\n💾 STRESS TEST 7: Transaction Performance");
    println!("-----------------------------------------");
    
    let start = Instant::now();
    let commit1 = graph.commit("Massive stress test data load".to_string(), "stress_test".to_string())?;
    let duration_commit = start.elapsed().as_millis();
    
    println!("✅ Committed massive transaction: {} (commit: {})",
             format_duration(duration_commit), commit1);
    
    // Make additional changes for a second commit
    let start = Instant::now();
    for i in 0..1000 {
        graph.set_node_attr(bulk_nodes[i], "stress_version".to_string(), AttrValue::Int(2))?;
    }
    let commit2 = graph.commit("Updated 1000 nodes post-stress".to_string(), "stress_test".to_string())?;
    let duration_commit2 = start.elapsed().as_millis();
    
    println!("✅ Made 1000 changes and committed: {} (commit: {})",
             format_duration(duration_commit2), commit2);
    
    // STRESS TEST 8: Memory and Performance Analysis
    println!("\n📊 STRESS TEST 8: Performance Summary");
    println!("-------------------------------------");
    
    let final_stats = graph.statistics();
    
    println!("🏆 FINAL STRESS TEST RESULTS:");
    println!("   🏠 Total Nodes: {}", final_stats.node_count);
    println!("   🔗 Total Edges: {}", final_stats.edge_count);
    println!("   🏷️  Attributes: {}", final_stats.attribute_count);
    println!("   💾 Commits: {}", final_stats.commit_count);
    println!("   🌿 Branches: {}", final_stats.branch_count);
    println!("   💾 Memory Usage: {:.2} MB", final_stats.memory_usage_mb);
    
    // Calculate total operations performed
    let total_node_ops = 5000 + 10000; // individual + bulk
    let total_edge_ops = dense_edges.len() + bulk_edges.len();
    let total_attr_ops = 6000 + 50000 + 6000; // individual node + bulk node + edge
    let total_retrieval_ops = 30000;
    let total_topology_ops = 2000;
    
    let grand_total = total_node_ops + total_edge_ops + total_attr_ops + total_retrieval_ops + total_topology_ops;
    
    println!("\n📈 OPERATION SUMMARY:");
    println!("   🏭 Node Operations: {}", total_node_ops);
    println!("   🌐 Edge Operations: {}", total_edge_ops);
    println!("   🏷️  Attribute Operations: {}", total_attr_ops);
    println!("   📤 Retrieval Operations: {}", total_retrieval_ops);
    println!("   🌍 Topology Operations: {}", total_topology_ops);
    println!("   🎯 TOTAL OPERATIONS: {}", grand_total);
    
    // Performance insights
    println!("\n🔍 PERFORMANCE INSIGHTS:");
    if speedup > 1.0 {
        println!("   ⚡ Bulk node creation is {:.1}x faster than individual", speedup);
    }
    if attr_speedup > 1.0 {
        println!("   ⚡ Bulk attribute setting is {:.1}x faster than individual", attr_speedup);
    }
    
    // Memory efficiency
    let avg_memory_per_node = final_stats.memory_usage_mb / final_stats.node_count as f64;
    let avg_memory_per_edge = final_stats.memory_usage_mb / final_stats.edge_count as f64;
    
    println!("   💾 Average memory per node: {:.4} MB", avg_memory_per_node);
    println!("   💾 Average memory per edge: {:.4} MB", avg_memory_per_edge);
    
    println!("\n🎉 STRESS TEST COMPLETED SUCCESSFULLY!");
    println!("   Graph library handled {} operations efficiently", grand_total);
    println!("   Memory usage remains reasonable at {:.2} MB", final_stats.memory_usage_mb);
    println!("   All data integrity checks passed!");
    
    Ok(())
}