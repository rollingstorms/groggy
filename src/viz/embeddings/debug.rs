//! Debug utilities for embedding monitoring and validation

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::storage::matrix::GraphMatrix;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::time::{SystemTime, UNIX_EPOCH};

/// Debug data collector for embedding computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingDebugData {
    /// Snapshots of embedding during computation
    pub embedding_snapshots: Vec<EmbeddingSnapshot>,
    /// Energy values over time (for energy-based methods)
    pub energy_history: Vec<f64>,
    /// Gradient norms over time
    pub gradient_norms: Vec<f64>,
    /// Performance metrics
    pub performance_metrics: PerformanceMetrics,
    /// Graph metadata
    pub graph_metadata: GraphMetadata,
    /// Embedding configuration
    pub config_info: HashMap<String, String>,
}

/// Snapshot of embedding state at a specific time
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct EmbeddingSnapshot {
    /// Timestamp when snapshot was taken
    pub timestamp: f64,
    /// Iteration number (for iterative methods)
    pub iteration: usize,
    /// Current embedding matrix (serialized as nested arrays)
    pub embedding_data: Vec<Vec<f64>>,
    /// Current energy (if applicable)
    pub energy: Option<f64>,
    /// Gradient norm (if applicable)
    pub gradient_norm: Option<f64>,
}

/// Performance metrics for embedding computation
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct PerformanceMetrics {
    /// Total computation time in seconds
    pub total_time_sec: f64,
    /// Time per iteration (for iterative methods)
    pub time_per_iteration: Vec<f64>,
    /// Memory usage snapshots
    pub memory_usage_mb: Vec<f64>,
    /// Peak memory usage
    pub peak_memory_mb: f64,
}

/// Graph metadata for context
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GraphMetadata {
    /// Number of nodes
    pub node_count: usize,
    /// Number of edges
    pub edge_count: usize,
    /// Whether graph is connected
    pub is_connected: bool,
    /// Graph density (edges / max_possible_edges)
    pub density: f64,
    /// Average degree
    pub average_degree: f64,
    /// Clustering coefficient
    pub clustering_coefficient: Option<f64>,
}

impl EmbeddingDebugData {
    /// Create new debug data collector
    pub fn new(graph: &Graph, config_info: HashMap<String, String>) -> GraphResult<Self> {
        let graph_metadata = GraphMetadata::from_graph(graph)?;

        Ok(Self {
            embedding_snapshots: Vec::new(),
            energy_history: Vec::new(),
            gradient_norms: Vec::new(),
            performance_metrics: PerformanceMetrics::new(),
            graph_metadata,
            config_info,
        })
    }

    /// Add an embedding snapshot
    pub fn add_snapshot(
        &mut self,
        embedding: &GraphMatrix,
        iteration: usize,
        energy: Option<f64>,
        gradient_norm: Option<f64>,
    ) -> GraphResult<()> {
        let timestamp = current_timestamp();

        // Convert matrix to nested array for serialization
        let (n_rows, n_cols) = embedding.shape();
        let mut embedding_data = Vec::with_capacity(n_rows);

        for i in 0..n_rows {
            let mut row = Vec::with_capacity(n_cols);
            for j in 0..n_cols {
                row.push(embedding.get_checked(i, j)?);
            }
            embedding_data.push(row);
        }

        let snapshot = EmbeddingSnapshot {
            timestamp,
            iteration,
            embedding_data,
            energy,
            gradient_norm,
        };

        self.embedding_snapshots.push(snapshot);

        // Also update history arrays
        if let Some(energy) = energy {
            self.energy_history.push(energy);
        }
        if let Some(grad_norm) = gradient_norm {
            self.gradient_norms.push(grad_norm);
        }

        Ok(())
    }

    /// Record timing for an iteration
    pub fn record_iteration_time(&mut self, time_sec: f64) {
        self.performance_metrics.time_per_iteration.push(time_sec);
    }

    /// Record memory usage
    pub fn record_memory_usage(&mut self, memory_mb: f64) {
        self.performance_metrics.memory_usage_mb.push(memory_mb);
        if memory_mb > self.performance_metrics.peak_memory_mb {
            self.performance_metrics.peak_memory_mb = memory_mb;
        }
    }

    /// Finalize debug data collection
    pub fn finalize(&mut self, total_time_sec: f64) {
        self.performance_metrics.total_time_sec = total_time_sec;
    }

    /// Export debug data to JSON file
    pub fn export_to_file(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let json_data = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json_data)?;
        Ok(())
    }

    /// Create HTML visualization of the debug data
    pub fn create_visualization_html(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        let html = format!(
            r#"
<!DOCTYPE html>
<html>
<head>
    <title>Embedding Debug Visualization</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .plot-container {{ margin: 20px 0; }}
        .metadata {{ background: #f5f5f5; padding: 15px; border-radius: 5px; margin: 20px 0; }}
        .metric {{ margin: 5px 0; }}
    </style>
</head>
<body>
    <h1>Embedding Debug Visualization</h1>

    <div class="metadata">
        <h2>Graph Metadata</h2>
        <div class="metric">Nodes: {}</div>
        <div class="metric">Edges: {}</div>
        <div class="metric">Connected: {}</div>
        <div class="metric">Density: {:.4}</div>
        <div class="metric">Average Degree: {:.2}</div>
    </div>

    <div class="metadata">
        <h2>Performance Metrics</h2>
        <div class="metric">Total Time: {:.2} seconds</div>
        <div class="metric">Peak Memory: {:.1} MB</div>
        <div class="metric">Iterations: {}</div>
    </div>

    <div id="energy-plot" class="plot-container"></div>
    <div id="gradient-plot" class="plot-container"></div>
    <div id="performance-plot" class="plot-container"></div>
    <div id="embedding-evolution" class="plot-container"></div>

    <script>
        const debugData = {};

        // Energy convergence plot
        if (debugData.energy_history.length > 0) {{
            const energyTrace = {{
                x: Array.from({{length: debugData.energy_history.length}}, (_, i) => i),
                y: debugData.energy_history,
                type: 'scatter',
                mode: 'lines',
                name: 'Energy',
                line: {{color: 'blue'}}
            }};

            Plotly.newPlot('energy-plot', [energyTrace], {{
                title: 'Energy Convergence',
                xaxis: {{title: 'Iteration'}},
                yaxis: {{title: 'Energy', type: 'log'}}
            }});
        }}

        // Gradient norm plot
        if (debugData.gradient_norms.length > 0) {{
            const gradientTrace = {{
                x: Array.from({{length: debugData.gradient_norms.length}}, (_, i) => i),
                y: debugData.gradient_norms,
                type: 'scatter',
                mode: 'lines',
                name: 'Gradient Norm',
                line: {{color: 'red'}}
            }};

            Plotly.newPlot('gradient-plot', [gradientTrace], {{
                title: 'Gradient Norm Over Time',
                xaxis: {{title: 'Iteration'}},
                yaxis: {{title: 'Gradient Norm', type: 'log'}}
            }});
        }}

        // Performance metrics
        if (debugData.performance_metrics.time_per_iteration.length > 0) {{
            const timeTrace = {{
                x: Array.from({{length: debugData.performance_metrics.time_per_iteration.length}}, (_, i) => i),
                y: debugData.performance_metrics.time_per_iteration,
                type: 'scatter',
                mode: 'lines',
                name: 'Time per Iteration',
                line: {{color: 'green'}}
            }};

            Plotly.newPlot('performance-plot', [timeTrace], {{
                title: 'Performance Over Time',
                xaxis: {{title: 'Iteration'}},
                yaxis: {{title: 'Time (seconds)'}}
            }});
        }}

        // Embedding evolution (first 2 dimensions)
        if (debugData.embedding_snapshots.length > 1) {{
            const firstSnapshot = debugData.embedding_snapshots[0];
            const lastSnapshot = debugData.embedding_snapshots[debugData.embedding_snapshots.length - 1];

            const initialTrace = {{
                x: firstSnapshot.embedding_data.map(row => row[0]),
                y: firstSnapshot.embedding_data.map(row => row[1]),
                mode: 'markers',
                type: 'scatter',
                name: 'Initial',
                marker: {{color: 'lightblue', size: 8}}
            }};

            const finalTrace = {{
                x: lastSnapshot.embedding_data.map(row => row[0]),
                y: lastSnapshot.embedding_data.map(row => row[1]),
                mode: 'markers',
                type: 'scatter',
                name: 'Final',
                marker: {{color: 'darkblue', size: 8}}
            }};

            Plotly.newPlot('embedding-evolution', [initialTrace, finalTrace], {{
                title: 'Embedding Evolution (First 2 Dimensions)',
                xaxis: {{title: 'Dimension 1'}},
                yaxis: {{title: 'Dimension 2'}}
            }});
        }}
    </script>
</body>
</html>
        "#,
            self.graph_metadata.node_count,
            self.graph_metadata.edge_count,
            self.graph_metadata.is_connected,
            self.graph_metadata.density,
            self.graph_metadata.average_degree,
            self.performance_metrics.total_time_sec,
            self.performance_metrics.peak_memory_mb,
            self.embedding_snapshots.len(),
            serde_json::to_string(self)?
        );

        std::fs::write(path, html)?;
        Ok(())
    }

    /// Check if energy is decreasing (for convergence analysis)
    pub fn is_energy_decreasing(&self) -> bool {
        if self.energy_history.len() < 2 {
            return true; // Not enough data
        }

        let first = self.energy_history.first().unwrap();
        let last = self.energy_history.last().unwrap();
        last < first
    }

    /// Get energy decrease rate over the last N iterations
    pub fn energy_decrease_rate(&self, last_n: usize) -> Option<f64> {
        if self.energy_history.len() < last_n + 1 {
            return None;
        }

        let len = self.energy_history.len();
        let start_energy = self.energy_history[len - last_n - 1];
        let end_energy = self.energy_history[len - 1];

        Some((start_energy - end_energy) / last_n as f64)
    }

    /// Check convergence based on energy and gradient criteria
    pub fn has_converged(&self, energy_threshold: f64, gradient_threshold: f64) -> bool {
        // Check energy convergence
        if let Some(rate) = self.energy_decrease_rate(10) {
            if rate.abs() < energy_threshold {
                // Check gradient convergence
                if let Some(&last_gradient) = self.gradient_norms.last() {
                    return last_gradient < gradient_threshold;
                }
            }
        }
        false
    }
}

impl GraphMetadata {
    /// Create graph metadata from a Graph instance
    pub fn from_graph(graph: &Graph) -> GraphResult<Self> {
        let node_count = graph.space().node_count();
        let edge_count = graph.space().edge_count();

        let max_possible_edges = if node_count > 1 {
            node_count * (node_count - 1) / 2 // Undirected graph
        } else {
            1
        };

        let density = if max_possible_edges > 0 {
            edge_count as f64 / max_possible_edges as f64
        } else {
            0.0
        };

        let average_degree = if node_count > 0 {
            (2 * edge_count) as f64 / node_count as f64 // Each edge contributes to 2 nodes
        } else {
            0.0
        };

        let is_connected = if node_count > 0 {
            // TODO: Implement connectivity check
            true // Assume connected for now
        } else {
            false
        };

        // TODO: Implement clustering coefficient calculation
        let clustering_coefficient = None;

        Ok(Self {
            node_count,
            edge_count,
            is_connected,
            density,
            average_degree,
            clustering_coefficient,
        })
    }
}

impl PerformanceMetrics {
    pub fn new() -> Self {
        Self {
            total_time_sec: 0.0,
            time_per_iteration: Vec::new(),
            memory_usage_mb: Vec::new(),
            peak_memory_mb: 0.0,
        }
    }

    /// Get average time per iteration
    pub fn average_iteration_time(&self) -> f64 {
        if self.time_per_iteration.is_empty() {
            0.0
        } else {
            self.time_per_iteration.iter().sum::<f64>() / self.time_per_iteration.len() as f64
        }
    }

    /// Get current memory usage (simplified - returns 0 for now)
    pub fn current_memory_usage_mb() -> f64 {
        // TODO: Implement actual memory usage measurement
        // This would require platform-specific code or external crates
        0.0
    }
}

/// Get current timestamp as seconds since epoch
fn current_timestamp() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs_f64()
}

/// Debug-enabled embedding wrapper
#[derive(Debug)]
pub struct DebuggableEmbedding<T: super::EmbeddingEngine> {
    engine: T,
    debug_data: Option<EmbeddingDebugData>,
}

impl<T: super::EmbeddingEngine> DebuggableEmbedding<T> {
    pub fn new(engine: T) -> Self {
        Self {
            engine,
            debug_data: None,
        }
    }

    pub fn with_debug(
        mut self,
        graph: &Graph,
        config_info: HashMap<String, String>,
    ) -> GraphResult<Self> {
        self.debug_data = Some(EmbeddingDebugData::new(graph, config_info)?);
        Ok(self)
    }

    pub fn get_debug_data(&self) -> Option<&EmbeddingDebugData> {
        self.debug_data.as_ref()
    }

    pub fn export_debug_data(&self, path: &str) -> Result<(), Box<dyn std::error::Error>> {
        if let Some(debug_data) = &self.debug_data {
            debug_data.export_to_file(path)?;
        }
        Ok(())
    }
}

impl<T: super::EmbeddingEngine> super::EmbeddingEngine for DebuggableEmbedding<T> {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        // For now, just delegate to the wrapped engine
        // In a full implementation, this would intercept the computation
        // to collect debug data at each iteration
        self.engine.compute_embedding(graph, dimensions)
    }

    fn supports_incremental(&self) -> bool {
        self.engine.supports_incremental()
    }

    fn supports_streaming(&self) -> bool {
        self.engine.supports_streaming()
    }

    fn name(&self) -> &str {
        self.engine.name()
    }

    fn default_dimensions(&self) -> usize {
        self.engine.default_dimensions()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::viz::embeddings::{random::RandomEmbedding, EmbeddingEngine};

    fn path_graph(n: usize) -> Graph {
        let mut graph = Graph::new();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node()).collect();
        for i in 0..n - 1 {
            graph.add_edge(nodes[i], nodes[i + 1]).unwrap();
        }
        graph
    }

    fn cycle_graph(n: usize) -> Graph {
        let mut graph = path_graph(n);
        let nodes: Vec<_> = graph.space().node_ids();
        if n > 2 {
            graph.add_edge(nodes[n - 1], nodes[0]).unwrap();
        }
        graph
    }

    fn star_graph(n: usize) -> Graph {
        let mut graph = Graph::new();
        let center = graph.add_node();
        for _ in 1..n {
            let leaf = graph.add_node();
            graph.add_edge(center, leaf).unwrap();
        }
        graph
    }

    fn karate_club() -> Graph {
        let mut graph = Graph::new();
        let nodes: Vec<_> = (0..34).map(|_| graph.add_node()).collect();
        // Add some representative edges for karate club
        let edges = [
            (0, 1),
            (0, 2),
            (0, 3),
            (1, 2),
            (1, 3),
            (2, 3),
            (1, 7),
            (2, 7),
            (3, 7),
        ];
        for (i, j) in edges.iter() {
            graph.add_edge(nodes[*i], nodes[*j]).unwrap();
        }
        graph
    }

    #[test]
    fn test_debug_data_creation() {
        let graph = karate_club();
        let config = [("method".to_string(), "test".to_string())]
            .iter()
            .cloned()
            .collect();

        let debug_data = EmbeddingDebugData::new(&graph, config);
        assert!(debug_data.is_ok());

        let data = debug_data.unwrap();
        assert_eq!(data.graph_metadata.node_count, graph.space().node_count());
        assert_eq!(data.graph_metadata.edge_count, graph.space().edge_count());
    }

    #[test]
    fn test_embedding_snapshot() {
        let graph = path_graph(3);
        let config = HashMap::new();
        let mut debug_data = EmbeddingDebugData::new(&graph, config).unwrap();

        // Create a test embedding
        let embedding = RandomEmbedding::gaussian(0.0, 1.0)
            .with_seed(42)
            .compute_embedding(&graph, 2)
            .unwrap();

        // Add snapshot
        let result = debug_data.add_snapshot(&embedding, 0, Some(1.5), Some(0.1));
        assert!(result.is_ok());

        assert_eq!(debug_data.embedding_snapshots.len(), 1);
        assert_eq!(debug_data.energy_history.len(), 1);
        assert_eq!(debug_data.gradient_norms.len(), 1);

        let snapshot = &debug_data.embedding_snapshots[0];
        assert_eq!(snapshot.iteration, 0);
        assert_eq!(snapshot.energy, Some(1.5));
        assert_eq!(snapshot.gradient_norm, Some(0.1));
        assert_eq!(snapshot.embedding_data.len(), 3); // 3 nodes
        assert_eq!(snapshot.embedding_data[0].len(), 2); // 2 dimensions
    }

    #[test]
    fn test_convergence_analysis() {
        let graph = cycle_graph(4);
        let config = HashMap::new();
        let mut debug_data = EmbeddingDebugData::new(&graph, config).unwrap();

        // Simulate decreasing energy
        let energies = vec![10.0, 8.0, 6.5, 5.2, 4.1, 3.3, 2.8, 2.5, 2.3, 2.2];
        debug_data.energy_history = energies;

        assert!(debug_data.is_energy_decreasing());

        let rate = debug_data.energy_decrease_rate(5);
        assert!(rate.is_some());
        assert!(rate.unwrap() > 0.0); // Should be positive for decreasing energy
    }

    #[test]
    fn test_debuggable_embedding() {
        let graph = path_graph(4);
        let config = [("method".to_string(), "random".to_string())]
            .iter()
            .cloned()
            .collect();

        let engine = RandomEmbedding::gaussian(0.0, 1.0).with_seed(42);
        let debuggable = DebuggableEmbedding::new(engine).with_debug(&graph, config);

        assert!(debuggable.is_ok());

        let debug_engine = debuggable.unwrap();
        let embedding = debug_engine.compute_embedding(&graph, 3);

        assert!(embedding.is_ok());
        assert_eq!(embedding.unwrap().shape(), (4, 3));

        // Debug data should be available
        assert!(debug_engine.get_debug_data().is_some());
    }

    #[test]
    fn test_export_functionality() {
        let graph = star_graph(5);
        let config = HashMap::new();
        let debug_data = EmbeddingDebugData::new(&graph, config).unwrap();

        // Test JSON export
        let json_path = "test_debug_export.json";
        let result = debug_data.export_to_file(json_path);
        assert!(result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(json_path);

        // Test HTML export
        let html_path = "test_debug_visualization.html";
        let html_result = debug_data.create_visualization_html(html_path);
        assert!(html_result.is_ok());

        // Clean up
        let _ = std::fs::remove_file(html_path);
    }
}
