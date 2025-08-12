// Rust native stress test - identical operations to Python version
use groggy::{Graph, AttrValue};
use groggy::core::query::NodeFilter;
use groggy::core::traversal::TraversalOptions;
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🦀 RUST NATIVE STRESS TEST");
    println!("==========================");
    
    let test_sizes = [(10000, 5000), (50000, 25000), (100000, 50000)];
    
    for (num_nodes, num_edges) in test_sizes {
        println!("\n📊 Testing {} nodes, {} edges", num_nodes, num_edges);
        
        // === GRAPH CREATION ===
        let start = Instant::now();
        let mut graph = Graph::new();
        
        // Create nodes with bulk operations
        let nodes = graph.add_nodes(num_nodes);
        
        // Set attributes with bulk operations
        let mut bulk_attrs = std::collections::HashMap::new();
        let mut type_data = Vec::new();
        let mut dept_data = Vec::new();
        let mut active_data = Vec::new();
        let mut age_data = Vec::new();
        let mut salary_data = Vec::new();
        
        for (i, &node_id) in nodes.iter().enumerate() {
            type_data.push((node_id, AttrValue::Text("user".to_string())));
            dept_data.push((node_id, AttrValue::Text(match i % 6 {
                0 => "Engineering",
                1 => "Marketing",
                2 => "Sales", 
                3 => "HR",
                4 => "Finance",
                _ => "Operations"
            }.to_string())));
            active_data.push((node_id, AttrValue::Bool(i % 3 != 0)));
            age_data.push((node_id, AttrValue::Int(25 + (i % 40) as i64)));
            salary_data.push((node_id, AttrValue::Int(50000 + (i % 100000) as i64)));
        }
        
        bulk_attrs.insert("type".to_string(), type_data);
        bulk_attrs.insert("department".to_string(), dept_data);
        bulk_attrs.insert("active".to_string(), active_data);
        bulk_attrs.insert("age".to_string(), age_data);
        bulk_attrs.insert("salary".to_string(), salary_data);
        
        graph.set_node_attrs(bulk_attrs)?;
        
        // Create edges with bulk operations
        let mut edge_specs = Vec::new();
        for _ in 0..num_edges {
            let from_idx = fastrand::usize(..num_nodes);
            let to_idx = fastrand::usize(..num_nodes);
            if from_idx != to_idx {
                edge_specs.push((nodes[from_idx], nodes[to_idx]));
            }
        }
        graph.add_edges(&edge_specs);
        
        let creation_time = start.elapsed();
        println!("   ✅ Graph creation: {:?}", creation_time);
        
        // === FILTERING TESTS ===
        println!("   🔍 Filtering tests:");
        
        // Single attribute filter
        let start = Instant::now();
        let type_filter = NodeFilter::AttributeEquals {
            name: "type".to_string(),
            value: AttrValue::Text("user".to_string()),
        };
        let type_results = graph.find_nodes(type_filter)?;
        let type_time = start.elapsed();
        println!("      Single attribute: {:?} ({} results)", type_time, type_results.len());
        
        // Complex AND filter
        let start = Instant::now();
        let and_filter = NodeFilter::And(vec![
            NodeFilter::AttributeEquals {
                name: "type".to_string(),
                value: AttrValue::Text("user".to_string()),
            },
            NodeFilter::AttributeEquals {
                name: "department".to_string(),
                value: AttrValue::Text("Engineering".to_string()),
            },
            NodeFilter::AttributeEquals {
                name: "active".to_string(),
                value: AttrValue::Bool(true),
            }
        ]);
        let and_results = graph.find_nodes(and_filter)?;
        let and_time = start.elapsed();
        println!("      Complex AND: {:?} ({} results)", and_time, and_results.len());
        
        // OR filter
        let start = Instant::now();
        let or_filter = NodeFilter::Or(vec![
            NodeFilter::AttributeEquals {
                name: "department".to_string(),
                value: AttrValue::Text("Engineering".to_string()),
            },
            NodeFilter::AttributeEquals {
                name: "department".to_string(),
                value: AttrValue::Text("Marketing".to_string()),
            }
        ]);
        let or_results = graph.find_nodes(or_filter)?;
        let or_time = start.elapsed();
        println!("      Complex OR: {:?} ({} results)", or_time, or_results.len());
        
        // === TRAVERSAL TESTS ===
        println!("   🌐 Traversal tests:");
        
        // Connected components
        let start = Instant::now();
        let components = graph.connected_components(TraversalOptions::default())?;
        let cc_time = start.elapsed();
        println!("      Connected components: {:?} ({} components)", 
                cc_time, components.total_components);
        
        // BFS from random node
        if !nodes.is_empty() {
            let start = Instant::now();
            let start_node = nodes[fastrand::usize(..nodes.len())];
            let bfs_result = graph.bfs(start_node, TraversalOptions::default())?;
            let bfs_time = start.elapsed();
            println!("      BFS traversal: {:?} ({} nodes)", bfs_time, bfs_result.nodes.len());
        }
        
        // === AGGREGATION TESTS ===
        println!("   📊 Aggregation tests:");
        
        // Basic statistics
        let start = Instant::now();
        let stats = graph.statistics();
        let stats_time = start.elapsed();
        println!("      Basic statistics: {:?}", stats_time);
        println!("         Nodes: {}, Edges: {}", stats.node_count, stats.edge_count);
        
        // Performance summary
        let nodes_per_ms = num_nodes as f64 / creation_time.as_millis() as f64;
        println!("   📈 Performance: {:.1} nodes/ms creation", nodes_per_ms);
    }
    
    Ok(())
}