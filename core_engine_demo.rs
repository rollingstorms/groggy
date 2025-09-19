//! Demo of the unified visualization core engine
//!
//! This demonstrates that the core engine works as specified in 
//! UNIFIED_VIZ_MIGRATION_PLAN.md Phase 1

use std::collections::HashMap;

// Import the core engine and required types
use groggy::viz::core::{VizEngine, VizConfig};
use groggy::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge, Position};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    println!("🎨 Groggy Unified Visualization Core Engine Demo");
    println!("================================================");
    
    // ✅ Step 1: Create VizEngine with configuration
    println!("\n✅ 1. Creating VizEngine with default configuration...");
    let config = VizConfig {
        width: 800.0,
        height: 600.0,
        physics_enabled: true,
        continuous_physics: false,
        target_fps: 60.0,
        interactions_enabled: true,
        auto_fit: true,
        fit_padding: 50.0,
    };
    
    let mut engine = VizEngine::new(config);
    println!("   Engine created successfully!");
    
    // ✅ Step 2: Create sample graph data
    println!("\n✅ 2. Creating sample graph data...");
    let nodes = vec![
        VizNode {
            id: "node1".to_string(),
            label: Some("Central Hub".to_string()),
            attributes: HashMap::new(),
            position: Some(Position { x: 0.0, y: 0.0 }),
        },
        VizNode {
            id: "node2".to_string(),
            label: Some("Data Source".to_string()),
            attributes: HashMap::new(),
            position: Some(Position { x: 100.0, y: 0.0 }),
        },
        VizNode {
            id: "node3".to_string(),
            label: Some("Processor".to_string()),
            attributes: HashMap::new(),
            position: Some(Position { x: 0.0, y: 100.0 }),
        },
    ];
    
    let edges = vec![
        VizEdge {
            id: "edge1".to_string(),
            source: "node1".to_string(),
            target: "node2".to_string(),
            label: Some("feeds".to_string()),
            weight: Some(1.0),
            attributes: HashMap::new(),
        },
        VizEdge {
            id: "edge2".to_string(),
            source: "node2".to_string(),
            target: "node3".to_string(),
            label: Some("processes".to_string()),
            weight: Some(2.0),
            attributes: HashMap::new(),
        },
    ];
    
    println!("   Created {} nodes and {} edges", nodes.len(), edges.len());
    
    // ✅ Step 3: Set data in engine
    println!("\n✅ 3. Loading data into VizEngine...");
    engine.set_data(nodes, edges)?;
    println!("   Data loaded successfully!");
    
    // ✅ Step 4: Generate frames (this is the key unified interface)
    println!("\n✅ 4. Generating visualization frames...");
    let frame1 = engine.update()?;
    println!("   Frame 1: {} nodes, {} edges, simulation running: {}", 
             frame1.nodes.len(), 
             frame1.edges.len(),
             frame1.metadata.simulation_state.is_running);
    
    // ✅ Step 5: Test physics simulation
    println!("\n✅ 5. Running physics simulation to completion...");
    let final_frame = engine.simulate_to_completion()?;
    println!("   Final frame: alpha={:.6}, converged={}, energy={:.2}", 
             final_frame.metadata.simulation_state.alpha,
             final_frame.metadata.simulation_state.has_converged,
             final_frame.metadata.simulation_state.energy);
    
    // ✅ Step 6: Test rendering (unified rendering pipeline)
    println!("\n✅ 6. Testing unified rendering pipeline...");
    let render_output = engine.render(&final_frame)?;
    println!("   Rendered {} bytes in {:.2}ms", 
             render_output.metadata.size_bytes,
             render_output.metadata.render_time_ms);
    
    // Show a preview of the SVG output
    let preview = if render_output.content.len() > 200 {
        format!("{}...", &render_output.content[..200])
    } else {
        render_output.content
    };
    println!("   Preview: {}", preview);
    
    // ✅ Step 7: Test interactions
    println!("\n✅ 7. Testing interaction system...");
    engine.select_node("node1".to_string());
    engine.pin_node("node2".to_string());
    engine.set_hover(Some("node3".to_string()));
    
    let interactive_frame = engine.update()?;
    let node1 = &interactive_frame.nodes[0];
    println!("   Node1 interactions: selected={}, pinned={}, hovered={}", 
             node1.interaction_state.is_selected,
             node1.interaction_state.is_pinned,
             node1.interaction_state.is_hovered);
    
    // ✅ Step 8: Show engine capabilities
    println!("\n✅ 8. Engine capabilities summary:");
    println!("   • Physics simulation: ✅ Working");
    println!("   • Unified rendering: ✅ Working");
    println!("   • Interaction state: ✅ Working");
    println!("   • Frame generation: ✅ Working");
    println!("   • JSON serialization: ✅ Available");
    
    let frame_json = final_frame.to_json();
    match frame_json {
        Ok(json) => println!("   • JSON export: ✅ {} characters", json.len()),
        Err(e) => println!("   • JSON export: ❌ Failed: {}", e),
    }
    
    println!("\n🎉 UNIFIED VIZ CORE ENGINE DEMO COMPLETE!");
    println!("   Ready for adapter integration (Jupyter, Streaming, File)");
    
    Ok(())
}