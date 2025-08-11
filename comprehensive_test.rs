// Comprehensive test of all groggy graph functionality
use groggy::{Graph, AttrValue};
use std::collections::HashMap;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🧪 COMPREHENSIVE GROGGY TEST SUITE");
    println!("=====================================");
    
    // Test 1: Basic Graph Creation and Node Operations
    println!("\n📊 TEST 1: Basic Graph Operations");
    println!("----------------------------------");
    
    let mut graph = Graph::new();
    println!("✅ Created new graph");
    
    // Add nodes
    let node1 = graph.add_node();
    let node2 = graph.add_node();
    let node3 = graph.add_node();
    println!("✅ Added 3 nodes: {}, {}, {}", node1, node2, node3);
    
    // Test bulk node creation
    let bulk_nodes = graph.add_nodes(5);
    println!("✅ Added 5 bulk nodes: {:?}", bulk_nodes);
    
    // Check node existence
    assert!(graph.contains_node(node1));
    assert!(graph.contains_node(node2));
    assert!(!graph.contains_node(999)); // Non-existent node
    println!("✅ Node existence checks passed");
    
    // Test 2: Edge Operations
    println!("\n🔗 TEST 2: Edge Operations");
    println!("--------------------------");
    
    // Add edges
    let edge1 = graph.add_edge(node1, node2)?;
    let edge2 = graph.add_edge(node2, node3)?;
    let edge3 = graph.add_edge(node1, node3)?;
    println!("✅ Added 3 edges: {}, {}, {}", edge1, edge2, edge3);
    
    // Test bulk edge creation
    let bulk_edge_pairs = vec![
        (bulk_nodes[0], bulk_nodes[1]),
        (bulk_nodes[1], bulk_nodes[2]),
        (bulk_nodes[2], bulk_nodes[3]),
    ];
    let bulk_edges = graph.add_edges(&bulk_edge_pairs);
    println!("✅ Added {} bulk edges: {:?}", bulk_edges.len(), bulk_edges);
    
    // Check edge existence
    assert!(graph.contains_edge(edge1));
    assert!(graph.contains_edge(edge2));
    assert!(!graph.contains_edge(999)); // Non-existent edge
    println!("✅ Edge existence checks passed");
    
    // Test edge endpoints
    let (source, target) = graph.edge_endpoints(edge1)?;
    assert_eq!(source, node1);
    assert_eq!(target, node2);
    println!("✅ Edge endpoints correct: {} -> {}", source, target);
    
    // Test 3: Node Attribute Operations
    println!("\n🏷️  TEST 3: Node Attribute Operations");
    println!("------------------------------------");
    
    // Set individual attributes
    graph.set_node_attr(node1, "name".to_string(), AttrValue::Text("Alice".to_string()))?;
    graph.set_node_attr(node1, "age".to_string(), AttrValue::Int(25))?;
    graph.set_node_attr(node1, "height".to_string(), AttrValue::Float(5.6))?;
    graph.set_node_attr(node1, "active".to_string(), AttrValue::Bool(true))?;
    println!("✅ Set 4 different attribute types on node1");
    
    graph.set_node_attr(node2, "name".to_string(), AttrValue::Text("Bob".to_string()))?;
    graph.set_node_attr(node2, "age".to_string(), AttrValue::Int(30))?;
    graph.set_node_attr(node3, "name".to_string(), AttrValue::Text("Charlie".to_string()))?;
    graph.set_node_attr(node3, "age".to_string(), AttrValue::Int(35))?;
    println!("✅ Set attributes on node2 and node3");
    
    // Test bulk attribute setting
    let mut bulk_name_attrs = HashMap::new();
    bulk_name_attrs.insert("nickname".to_string(), vec![
        (node1, AttrValue::Text("Al".to_string())),
        (node2, AttrValue::Text("Bobby".to_string())),
        (node3, AttrValue::Text("Chuck".to_string())),
    ]);
    graph.set_node_attrs(bulk_name_attrs)?;
    println!("✅ Set bulk nickname attributes");
    
    // Test attribute retrieval
    let name1 = graph.get_node_attr(node1, &"name".to_string())?;
    assert_eq!(name1, Some(AttrValue::Text("Alice".to_string())));
    println!("✅ Retrieved node1 name: {:?}", name1);
    
    let age2 = graph.get_node_attr(node2, &"age".to_string())?;
    assert_eq!(age2, Some(AttrValue::Int(30)));
    println!("✅ Retrieved node2 age: {:?}", age2);
    
    // Test non-existent attribute
    let nonexistent = graph.get_node_attr(node1, &"nonexistent".to_string())?;
    assert_eq!(nonexistent, None);
    println!("✅ Non-existent attribute returns None");
    
    // Test get all attributes for a node
    let node1_attrs = graph.get_node_attrs(node1)?;
    println!("✅ Retrieved all node1 attributes (count: {})", node1_attrs.len());
    
    // Test 4: Edge Attribute Operations
    println!("\n🔗🏷️  TEST 4: Edge Attribute Operations");
    println!("-------------------------------------");
    
    // Set edge attributes
    graph.set_edge_attr(edge1, "weight".to_string(), AttrValue::Float(1.5))?;
    graph.set_edge_attr(edge1, "type".to_string(), AttrValue::Text("friendship".to_string()))?;
    graph.set_edge_attr(edge2, "weight".to_string(), AttrValue::Float(2.0))?;
    graph.set_edge_attr(edge3, "weight".to_string(), AttrValue::Float(0.8))?;
    println!("✅ Set edge attributes on 3 edges");
    
    // Test bulk edge attributes
    let mut bulk_edge_attrs = HashMap::new();
    bulk_edge_attrs.insert("category".to_string(), vec![
        (edge1, AttrValue::Text("social".to_string())),
        (edge2, AttrValue::Text("work".to_string())),
        (edge3, AttrValue::Text("family".to_string())),
    ]);
    graph.set_edge_attrs(bulk_edge_attrs)?;
    println!("✅ Set bulk edge category attributes");
    
    // Test edge attribute retrieval
    let weight1 = graph.get_edge_attr(edge1, &"weight".to_string())?;
    assert_eq!(weight1, Some(AttrValue::Float(1.5)));
    println!("✅ Retrieved edge1 weight: {:?}", weight1);
    
    let edge1_attrs = graph.get_edge_attrs(edge1)?;
    println!("✅ Retrieved all edge1 attributes (count: {})", edge1_attrs.len());
    
    // Test 5: Bulk Attribute Retrieval
    println!("\n📊 TEST 5: Bulk Attribute Retrieval");
    println!("-----------------------------------");
    
    // Test bulk node attribute retrieval
    let nodes_to_query = vec![node1, node2, node3];
    let names = graph.get_nodes_attrs(&"name".to_string(), &nodes_to_query)?;
    println!("✅ Bulk retrieved names: {:?}", names);
    
    let ages = graph.get_nodes_attrs(&"age".to_string(), &nodes_to_query)?;
    println!("✅ Bulk retrieved ages: {:?}", ages);
    
    // Test bulk edge attribute retrieval
    let edges_to_query = vec![edge1, edge2, edge3];
    let weights = graph.get_edges_attrs(&"weight".to_string(), &edges_to_query)?;
    println!("✅ Bulk retrieved edge weights: {:?}", weights);
    
    // Test 6: Topology Queries
    println!("\n🌐 TEST 6: Topology Queries");
    println!("---------------------------");
    
    // Get all node and edge IDs
    let all_nodes = graph.node_ids();
    let all_edges = graph.edge_ids();
    println!("✅ Total nodes: {}, Total edges: {}", all_nodes.len(), all_edges.len());
    
    // Test neighbors and degree (basic implementations)
    let node1_neighbors = graph.neighbors(node1)?;
    let node1_degree = graph.degree(node1)?;
    println!("✅ Node1 neighbors: {:?}, degree: {}", node1_neighbors, node1_degree);
    
    // Test 7: Graph Statistics
    println!("\n📈 TEST 7: Graph Statistics");
    println!("---------------------------");
    
    let stats = graph.statistics();
    println!("✅ Graph Statistics:");
    println!("   - Nodes: {}", stats.node_count);
    println!("   - Edges: {}", stats.edge_count);
    println!("   - Attributes: {}", stats.attribute_count);
    println!("   - Commits: {}", stats.commit_count);
    println!("   - Branches: {}", stats.branch_count);
    println!("   - Uncommitted changes: {}", stats.uncommitted_changes);
    println!("   - Memory usage: {:.2} MB", stats.memory_usage_mb);
    
    // Test 8: Transaction System
    println!("\n💾 TEST 8: Transaction System");
    println!("-----------------------------");
    
    // Check if we have changes to commit
    let has_changes_before = graph.has_uncommitted_changes();
    println!("✅ Has uncommitted changes: {}", has_changes_before);
    
    if has_changes_before {
        // Commit current changes
        let commit1 = graph.commit("Initial graph with nodes and edges".to_string(), "test_user".to_string())?;
        println!("✅ Created commit: {}", commit1);
        
        let has_changes_after = graph.has_uncommitted_changes();
        println!("✅ Has changes after commit: {}", has_changes_after);
    }
    
    // Make more changes and commit again
    graph.set_node_attr(node1, "last_updated".to_string(), AttrValue::Text("2024-01-01".to_string()))?;
    let commit2 = graph.commit("Updated last_updated field".to_string(), "test_user".to_string())?;
    println!("✅ Created second commit: {}", commit2);
    
    // Test 9: Node and Edge Removal
    println!("\n🗑️  TEST 9: Node and Edge Removal");
    println!("--------------------------------");
    
    // Test edge removal
    let initial_edge_count = graph.edge_ids().len();
    graph.remove_edge(edge3)?;
    let after_edge_removal = graph.edge_ids().len();
    assert_eq!(after_edge_removal, initial_edge_count - 1);
    println!("✅ Removed edge3, edges: {} -> {}", initial_edge_count, after_edge_removal);
    
    // Test bulk edge removal
    let edges_to_remove = vec![edge1, edge2];
    graph.remove_edges(&edges_to_remove)?;
    let after_bulk_removal = graph.edge_ids().len();
    println!("✅ Bulk removed 2 edges, edges now: {}", after_bulk_removal);
    
    // Test node removal (this should also remove incident edges)
    let initial_node_count = graph.node_ids().len();
    graph.remove_node(bulk_nodes[4])?; // Remove the last bulk node
    let after_node_removal = graph.node_ids().len();
    assert_eq!(after_node_removal, initial_node_count - 1);
    println!("✅ Removed node, nodes: {} -> {}", initial_node_count, after_node_removal);
    
    // Test bulk node removal
    let nodes_to_remove = vec![bulk_nodes[0], bulk_nodes[1]];
    graph.remove_nodes(&nodes_to_remove)?;
    let after_bulk_node_removal = graph.node_ids().len();
    println!("✅ Bulk removed 2 nodes, nodes now: {}", after_bulk_node_removal);
    
    // Test 10: Error Handling
    println!("\n⚠️  TEST 10: Error Handling");
    println!("---------------------------");
    
    // Test operations on non-existent nodes
    let result = graph.set_node_attr(999, "test".to_string(), AttrValue::Text("fail".to_string()));
    assert!(result.is_err());
    println!("✅ Setting attribute on non-existent node properly fails");
    
    let result = graph.get_node_attr(999, &"test".to_string());
    assert!(result.is_err());
    println!("✅ Getting attribute from non-existent node properly fails");
    
    // Test operations on non-existent edges
    let result = graph.set_edge_attr(999, "test".to_string(), AttrValue::Text("fail".to_string()));
    assert!(result.is_err());
    println!("✅ Setting attribute on non-existent edge properly fails");
    
    let result = graph.remove_edge(999);
    assert!(result.is_err());
    println!("✅ Removing non-existent edge properly fails");
    
    // Test adding edge between non-existent nodes
    let result = graph.add_edge(999, 998);
    assert!(result.is_err());
    println!("✅ Adding edge between non-existent nodes properly fails");
    
    // Final commit
    if graph.has_uncommitted_changes() {
        let final_commit = graph.commit("Final commit after removals".to_string(), "test_user".to_string())?;
        println!("✅ Final commit: {}", final_commit);
    }
    
    // Final statistics
    println!("\n📊 FINAL STATISTICS");
    println!("===================");
    let final_stats = graph.statistics();
    println!("Final graph state:");
    println!("   - Nodes: {}", final_stats.node_count);
    println!("   - Edges: {}", final_stats.edge_count);
    println!("   - Attributes: {}", final_stats.attribute_count);
    println!("   - Total commits: {}", final_stats.commit_count);
    println!("   - Memory usage: {:.2} MB", final_stats.memory_usage_mb);
    
    println!("\n🎉 ALL TESTS PASSED! Graph functionality is working correctly!");
    
    Ok(())
}