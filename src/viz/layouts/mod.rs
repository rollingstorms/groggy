//! Graph layout algorithms for visualization

use crate::errors::GraphResult;
use crate::viz::streaming::data_source::{GraphNode as VizNode, GraphEdge as VizEdge, Position};

/// Trait for graph layout algorithms
pub trait LayoutEngine {
    /// Compute positions for nodes given the graph structure
    fn compute_layout(&self, nodes: &[VizNode], edges: &[VizEdge]) -> GraphResult<Vec<(String, Position)>>;
    
    /// Get the name of this layout algorithm
    fn name(&self) -> &str;
    
    /// Check if this layout supports incremental updates
    fn supports_incremental(&self) -> bool {
        false
    }
}

/// Force-directed layout implementation (placeholder)
pub struct ForceDirectedLayout {
    pub charge: f64,
    pub distance: f64,
    pub iterations: usize,
}

impl Default for ForceDirectedLayout {
    fn default() -> Self {
        Self {
            charge: -300.0,
            distance: 50.0,
            iterations: 100,
        }
    }
}

impl LayoutEngine for ForceDirectedLayout {
    fn compute_layout(&self, nodes: &[VizNode], _edges: &[VizEdge]) -> GraphResult<Vec<(String, Position)>> {
        // Placeholder: arrange nodes in a circle for now
        let mut positions = Vec::new();
        let radius = 200.0;
        let angle_step = 2.0 * std::f64::consts::PI / nodes.len() as f64;
        
        for (i, node) in nodes.iter().enumerate() {
            let angle = i as f64 * angle_step;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            positions.push((node.id.clone(), Position { x, y }));
        }
        
        Ok(positions)
    }
    
    fn name(&self) -> &str {
        "force-directed"
    }
}

/// Circular layout implementation
pub struct CircularLayout {
    pub radius: Option<f64>,
}

impl Default for CircularLayout {
    fn default() -> Self {
        Self { radius: None }
    }
}

impl LayoutEngine for CircularLayout {
    fn compute_layout(&self, nodes: &[VizNode], _edges: &[VizEdge]) -> GraphResult<Vec<(String, Position)>> {
        let mut positions = Vec::new();
        let radius = self.radius.unwrap_or(200.0);
        let angle_step = 2.0 * std::f64::consts::PI / nodes.len() as f64;
        
        for (i, node) in nodes.iter().enumerate() {
            let angle = i as f64 * angle_step;
            let x = radius * angle.cos();
            let y = radius * angle.sin();
            positions.push((node.id.clone(), Position { x, y }));
        }
        
        Ok(positions)
    }
    
    fn name(&self) -> &str {
        "circular"
    }
}