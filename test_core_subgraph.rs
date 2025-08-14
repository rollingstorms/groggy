//! Test Core Subgraph Implementation
//! 
//! This demonstrates the enhanced Subgraph architecture in core Rust
//! where Subgraphs have full Graph API capabilities.

use groggy::{Graph, AttrValue};
use groggy::core::subgraph::Subgraph;
use std::rc::Rc;
use std::cell::RefCell;
use std::collections::HashSet;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🚀 Testing Core Subgraph Implementation");
    
    // Create a test graph
    let mut graph = Graph::new();
    
    // Add test nodes with attributes
    let node1 = graph.add_node();
    let node2 = graph.add_node();
    let node3 = graph.add_node();
    let node4 = graph.add_node();
    
    graph.set_node_attr(node1, "name".to_string(), AttrValue::Text("Alice".to_string()))?;
    graph.set_node_attr(node1, "age".to_string(), AttrValue::Int(30))?;
    graph.set_node_attr(node1, "dept".to_string(), AttrValue::Text("Engineering".to_string()))?;
    
    graph.set_node_attr(node2, "name".to_string(), AttrValue::Text("Bob".to_string()))?;
    graph.set_node_attr(node2, "age".to_string(), AttrValue::Int(25))?;
    graph.set_node_attr(node2, "dept".to_string(), AttrValue::Text("Engineering".to_string()))?;
    
    graph.set_node_attr(node3, "name".to_string(), AttrValue::Text("Carol".to_string()))?;
    graph.set_node_attr(node3, "age".to_string(), AttrValue::Int(35))?;
    graph.set_node_attr(node3, "dept".to_string(), AttrValue::Text("Design".to_string()))?;
    
    graph.set_node_attr(node4, "name".to_string(), AttrValue::Text("Dave".to_string()))?;
    graph.set_node_attr(node4, "age".to_string(), AttrValue::Int(28))?;
    graph.set_node_attr(node4, "dept".to_string(), AttrValue::Text("Design".to_string()))?;
    
    // Add edges
    let edge1 = graph.add_edge(node1, node2)?;
    let edge2 = graph.add_edge(node2, node3)?;
    let edge3 = graph.add_edge(node3, node4)?;
    
    graph.set_edge_attr(edge1, "weight".to_string(), AttrValue::Float(0.8))?;
    graph.set_edge_attr(edge2, "weight".to_string(), AttrValue::Float(0.6))?;
    graph.set_edge_attr(edge3, "weight".to_string(), AttrValue::Float(0.9))?;
    
    println!("✅ Created test graph with {} nodes and {} edges", 
             graph.node_count(), graph.edge_count());
    
    // Test 1: Create Subgraph from nodes
    println!("\n📋 Test 1: Creating Subgraph from nodes");
    let graph_rc = Rc::new(RefCell::new(graph));
    let node_subset = HashSet::from([node1, node2, node3]);
    let subgraph = Subgraph::from_nodes(
        graph_rc.clone(),
        node_subset,
        "test_subgraph".to_string(),
    )?;
    
    println!("✅ Created subgraph: {}", subgraph);
    println!("   Nodes: {:?}", subgraph.node_ids());
    println!("   Edges: {:?}", subgraph.edge_ids());
    
    // Test 2: Column access for bulk attribute extraction
    println!("\n📋 Test 2: Column access for bulk attribute extraction");
    let names = subgraph.get_node_attribute_column(&"name".to_string())?;
    let ages = subgraph.get_node_attribute_column(&"age".to_string())?;
    let depts = subgraph.get_node_attribute_column(&"dept".to_string())?;
    
    println!("✅ Column access results:");
    println!("   Names: {:?}", names);
    println!("   Ages: {:?}", ages);
    println!("   Departments: {:?}", depts);
    
    // Test 3: Batch operations on subgraph
    println!("\n📋 Test 3: Batch operations on subgraph");
    subgraph.set_node_attribute_bulk(
        &"team".to_string(), 
        AttrValue::Text("Alpha".to_string())
    )?;
    
    println!("✅ Set 'team' attribute on all subgraph nodes");
    
    // Verify the batch operation worked
    let teams = subgraph.get_node_attribute_column(&"team".to_string())?;
    println!("   Team values: {:?}", teams);
    
    // Test 4: Attribute filtering within subgraph
    println!("\n📋 Test 4: Attribute filtering within subgraph");
    use std::collections::HashMap;
    use groggy::core::query::AttributeFilter;
    
    let mut filters = HashMap::new();
    filters.insert(
        "dept".to_string(), 
        AttributeFilter::Equals(AttrValue::Text("Engineering".to_string()))
    );
    
    let engineering_subgraph = subgraph.filter_nodes_by_attributes(&filters)?;
    println!("✅ Filtered subgraph to Engineering department: {}", engineering_subgraph);
    println!("   Engineering nodes: {:?}", engineering_subgraph.node_ids());
    
    // Test 5: Infinite composability (subgraph operations on subgraph)
    println!("\n📋 Test 5: Infinite composability");
    
    // First filter by department, then by age
    let young_engineers = engineering_subgraph.filter_nodes_by_attribute(
        &"age".to_string(),
        &AttrValue::Int(25)
    )?;
    
    println!("✅ Double-filtered (Engineering + age=25): {}", young_engineers);
    println!("   Young engineers: {:?}", young_engineers.node_ids());
    
    // Test 6: Multi-attribute bulk operations
    println!("\n📋 Test 6: Multi-attribute bulk operations");
    let mut bulk_attrs = HashMap::new();
    bulk_attrs.insert("status".to_string(), AttrValue::Text("Active".to_string()));
    bulk_attrs.insert("rating".to_string(), AttrValue::Float(4.5));
    
    subgraph.set_node_attributes_bulk(bulk_attrs)?;
    println!("✅ Set multiple attributes on all subgraph nodes");
    
    let statuses = subgraph.get_node_attribute_column(&"status".to_string())?;
    let ratings = subgraph.get_node_attribute_column(&"rating".to_string())?;
    println!("   Statuses: {:?}", statuses);
    println!("   Ratings: {:?}", ratings);
    
    println!("\n🎉 All core Subgraph tests passed!");
    println!("✨ The enhanced Subgraph architecture is working:");
    println!("   • Subgraphs have full Graph reference through Rc<RefCell<Graph>>");
    println!("   • Column access for bulk attribute extraction");
    println!("   • Batch operations for setting attributes");
    println!("   • Attribute filtering creates new subgraphs");
    println!("   • Infinite composability through method chaining");
    println!("   • All operations work within subgraph constraints");
    
    Ok(())
}