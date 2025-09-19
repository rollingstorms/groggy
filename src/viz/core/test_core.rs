//! Test that the core engine produces correct output
//!
//! This tests that our unified core engine produces the same type of
//! output as the existing streaming system.

#[cfg(test)]
mod tests {
    use super::super::{
        engine::{VizEngine, VizConfig},
        frame::VizFrame,
    };
    use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge, Position};
    use crate::types::AttrValue;
    use std::collections::HashMap;

    /// Test that the core engine can be created and initialized
    #[test]
    fn test_viz_engine_creation() {
        let config = VizConfig::default();
        let engine = VizEngine::new(config);
        
        // Should be able to create engine without errors
        assert!(!engine.is_simulation_running());
    }

    /// Test that the core engine can process simple graph data
    #[test]
    fn test_viz_engine_simple_graph() {
        let mut engine = VizEngine::new(VizConfig::default());
        
        // Create simple test graph
        let nodes = vec![
            VizNode {
                id: "node1".to_string(),
                label: Some("Node 1".to_string()),
                attributes: HashMap::new(),
                position: Some(Position { x: 0.0, y: 0.0 }),
            },
            VizNode {
                id: "node2".to_string(),
                label: Some("Node 2".to_string()),
                attributes: HashMap::new(),
                position: Some(Position { x: 100.0, y: 0.0 }),
            },
        ];
        
        let edges = vec![
            VizEdge {
                id: "edge1".to_string(),
                source: "node1".to_string(),
                target: "node2".to_string(),
                label: Some("connects".to_string()),
                weight: Some(1.0),
                attributes: HashMap::new(),
            },
        ];
        
        // Set data and update
        engine.set_data(nodes, edges).expect("Failed to set data");
        let frame = engine.update().expect("Failed to update engine");
        
        // Verify frame structure
        assert_eq!(frame.nodes.len(), 2);
        assert_eq!(frame.edges.len(), 1);
        assert_eq!(frame.nodes[0].id, "node1");
        assert_eq!(frame.nodes[1].id, "node2");
        assert_eq!(frame.edges[0].id, "edge1");
        assert_eq!(frame.edges[0].source, "node1");
        assert_eq!(frame.edges[0].target, "node2");
    }

    /// Test physics simulation produces different positions
    #[test]
    fn test_physics_simulation() {
        let mut config = VizConfig::default();
        config.physics_enabled = true;
        let mut engine = VizEngine::new(config);
        
        // Create nodes in conflict (overlapping positions)
        let nodes = vec![
            VizNode {
                id: "node1".to_string(),
                label: Some("Node 1".to_string()),
                attributes: HashMap::new(),
                position: Some(Position { x: 0.0, y: 0.0 }),
            },
            VizNode {
                id: "node2".to_string(),
                label: Some("Node 2".to_string()),
                attributes: HashMap::new(),
                position: Some(Position { x: 0.0, y: 0.0 }), // Same position -> should repel
            },
        ];
        
        let edges = vec![];
        
        // Set data and run simulation
        engine.set_data(nodes, edges).expect("Failed to set data");
        
        // Get initial positions
        let initial_positions = engine.get_positions().clone();
        
        // Run simulation to completion
        let final_frame = engine.simulate_to_completion().expect("Failed to simulate");
        
        // Positions should have changed due to repulsion
        let final_positions = engine.get_positions();
        
        // Nodes should no longer be at the same position
        let node1_initial = &initial_positions["node1"];
        let node2_initial = &initial_positions["node2"];
        let node1_final = &final_positions["node1"];
        let node2_final = &final_positions["node2"];
        
        // Initial positions should be the same
        assert_eq!(node1_initial.x, node2_initial.x);
        assert_eq!(node1_initial.y, node2_initial.y);
        
        // Final positions should be different (repelled)
        let final_distance = ((node1_final.x - node2_final.x).powi(2) + 
                             (node1_final.y - node2_final.y).powi(2)).sqrt();
        
        assert!(final_distance > 10.0, "Nodes should have repelled to distance > 10, got {}", final_distance);
        
        // Simulation should be marked as completed
        assert!(!final_frame.metadata.simulation_state.is_running);
        assert!(final_frame.metadata.simulation_state.has_converged);
    }

    /// Test rendering produces output
    #[test]
    fn test_rendering_output() {
        let mut engine = VizEngine::new(VizConfig::default());
        
        // Create simple test graph
        let nodes = vec![
            VizNode {
                id: "test_node".to_string(),
                label: Some("Test".to_string()),
                attributes: HashMap::new(),
                position: Some(Position { x: 0.0, y: 0.0 }),
            },
        ];
        
        let edges = vec![];
        
        engine.set_data(nodes, edges).expect("Failed to set data");
        let frame = engine.update().expect("Failed to update engine");
        
        // Test rendering to different formats
        let svg_output = engine.render(&frame).expect("Failed to render SVG");
        
        // SVG output should contain expected elements
        assert!(svg_output.content.contains("<svg"));
        assert!(svg_output.content.contains("<circle"));
        assert!(svg_output.content.contains("test_node"));
        assert!(svg_output.metadata.node_count == 1);
        assert!(svg_output.metadata.edge_count == 0);
    }

    /// Test interaction state management
    #[test]
    fn test_interaction_state() {
        let mut engine = VizEngine::new(VizConfig::default());
        
        let nodes = vec![
            VizNode {
                id: "interactive_node".to_string(),
                label: Some("Interactive".to_string()),
                attributes: HashMap::new(),
                position: Some(Position { x: 0.0, y: 0.0 }),
            },
        ];
        
        engine.set_data(nodes, vec![]).expect("Failed to set data");
        
        // Test selection
        engine.select_node("interactive_node".to_string());
        assert!(engine.get_interaction_state().is_selected("interactive_node"));
        
        // Test hover
        engine.set_hover(Some("interactive_node".to_string()));
        assert!(engine.get_interaction_state().is_hovered("interactive_node"));
        
        // Test pin
        engine.pin_node("interactive_node".to_string());
        assert!(engine.get_interaction_state().is_pinned("interactive_node"));
        
        // Generate frame and check interaction states are applied
        let frame = engine.update().expect("Failed to update engine");
        let node_frame = &frame.nodes[0];
        
        assert!(node_frame.interaction_state.is_selected);
        assert!(node_frame.interaction_state.is_hovered);
        assert!(node_frame.interaction_state.is_pinned);
        assert_eq!(node_frame.interaction_state.highlight, 1.0); // Selected = full highlight
    }

    /// Test engine configuration changes
    #[test]
    fn test_config_changes() {
        let mut engine = VizEngine::new(VizConfig::default());
        
        // Initially physics should be enabled
        assert!(engine.get_frame_info().frame_count == 0);
        
        // Update config to disable physics
        let mut new_config = VizConfig::default();
        new_config.physics_enabled = false;
        new_config.width = 1200.0;
        new_config.height = 800.0;
        
        engine.set_config(new_config).expect("Failed to set config");
        
        // Set some data and update
        let nodes = vec![
            VizNode {
                id: "config_test".to_string(),
                label: Some("Config Test".to_string()),
                attributes: HashMap::new(),
                position: None,
            },
        ];
        
        engine.set_data(nodes, vec![]).expect("Failed to set data");
        let frame = engine.update().expect("Failed to update");
        
        // Frame should reflect new dimensions
        assert_eq!(frame.metadata.dimensions.width, 1200.0);
        assert_eq!(frame.metadata.dimensions.height, 800.0);
        
        // Physics should not be running
        assert!(!frame.metadata.simulation_state.is_running);
        assert!(!engine.is_simulation_running());
    }

    /// Regression test: ensure frame generation is consistent
    #[test]
    fn test_frame_consistency() {
        let mut engine = VizEngine::new(VizConfig::default());
        
        let nodes = vec![
            VizNode {
                id: "consistent_node".to_string(),
                label: Some("Consistent".to_string()),
                attributes: {
                    let mut attrs = HashMap::new();
                    attrs.insert("test_attr".to_string(), AttrValue::String("test_value".to_string()));
                    attrs
                },
                position: Some(Position { x: 50.0, y: 75.0 }),
            },
        ];
        
        engine.set_data(nodes, vec![]).expect("Failed to set data");
        
        // Generate multiple frames
        let frame1 = engine.update().expect("Failed to update 1");
        let frame2 = engine.update().expect("Failed to update 2");
        
        // Core data should be consistent
        assert_eq!(frame1.nodes.len(), frame2.nodes.len());
        assert_eq!(frame1.nodes[0].id, frame2.nodes[0].id);
        
        // Metadata should show progression
        assert!(frame2.metadata.frame_id != frame1.metadata.frame_id);
        assert!(frame2.metadata.timestamp >= frame1.metadata.timestamp);
        
        // Node attributes should be preserved
        let node = &frame2.nodes[0];
        assert!(node.attributes.contains_key("test_attr"));
        
        // Frame structure should be valid for serialization
        let json_result = frame2.to_json();
        assert!(json_result.is_ok(), "Frame should be serializable to JSON");
    }
}