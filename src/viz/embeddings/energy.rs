//! Energy-based embedding implementations
//!
//! This module implements various energy-based graph embedding methods,
//! including force-directed layouts extended to n dimensions and custom
//! energy function optimization.

use super::{EmbeddingEngine, EnergyFunction};
use crate::api::graph::Graph;
use crate::errors::{GraphResult, GraphError};
use crate::storage::matrix::GraphMatrix;
use crate::types::NodeId;
use crate::traits::subgraph_operations::SubgraphOperations;
use std::collections::HashMap;

/// Energy-based embedding engine with customizable energy functions
#[derive(Debug)]
pub struct EnergyEmbedding {
    /// Number of optimization iterations
    iterations: usize,
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Whether to use simulated annealing
    annealing: bool,
    /// Custom energy function
    energy_function: Option<EnergyFunction>,
    /// Random seed for reproducible results
    seed: Option<u64>,
    /// Debug data collection
    debug_enabled: bool,
}

impl EnergyEmbedding {
    pub fn new(iterations: usize, learning_rate: f64, annealing: bool, energy_function: Option<EnergyFunction>) -> Self {
        Self {
            iterations,
            learning_rate,
            annealing,
            energy_function,
            seed: None,
            debug_enabled: false,
        }
    }

    pub fn with_debug(mut self, enabled: bool) -> Self {
        self.debug_enabled = enabled;
        self
    }

    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    /// Initialize random positions for nodes
    fn initialize_positions(&self, node_count: usize, dimensions: usize) -> GraphResult<GraphMatrix> {
        let mut rng = if let Some(seed) = self.seed {
            fastrand::Rng::with_seed(seed)
        } else {
            fastrand::Rng::new()
        };

        // Initialize with small random values in n-dimensional space
        let mut positions = GraphMatrix::zeros(node_count, dimensions);

        for i in 0..node_count {
            for j in 0..dimensions {
                let val: f64 = rng.f64() * 0.2 - 0.1; // Range [-0.1, 0.1]
                positions.set(i, j, val)?;
            }
        }

        Ok(positions)
    }

    /// Compute energy gradients for all nodes
    fn compute_gradients(&self, positions: &GraphMatrix, graph: &Graph, node_indices: &HashMap<NodeId, usize>) -> GraphResult<GraphMatrix> {
        let (n_nodes, n_dims) = positions.shape();
        let mut gradients = GraphMatrix::zeros(n_nodes, n_dims);

        match &self.energy_function {
            Some(EnergyFunction::SpringElectric { attraction_strength, repulsion_strength, ideal_distance }) => {
                self.compute_spring_electric_gradients(&mut gradients, positions, graph, node_indices, *attraction_strength, *repulsion_strength, *ideal_distance)?;
            }

            Some(EnergyFunction::StressMinimization { stress_type }) => {
                self.compute_stress_gradients(&mut gradients, positions, *stress_type)?;
            }

            Some(EnergyFunction::Custom { attraction_fn, repulsion_fn }) => {
                // TODO: Implement custom function evaluation
                return Err(GraphError::NotImplemented {
                    feature: "Custom energy functions".to_string(),
                    tracking_issue: None,
                });
            }

            None => {
                // Default: spring-electric model
                self.compute_spring_electric_gradients(&mut gradients, positions, graph, node_indices, 1.0, 1.0, 1.0)?;
            }
        }

        Ok(gradients)
    }

    /// Compute gradients for spring-electric energy model
    fn compute_spring_electric_gradients(
        &self,
        gradients: &mut GraphMatrix,
        positions: &GraphMatrix,
        graph: &Graph,
        node_indices: &HashMap<NodeId, usize>,
        attraction_strength: f64,
        repulsion_strength: f64,
        ideal_distance: f64,
    ) -> GraphResult<()> {
        let (n_nodes, n_dims) = positions.shape();

        // 1. Attraction forces from edges
        for edge_id in graph.space().edge_ids() {
            if let Some((source, target)) = graph.pool().get_edge_endpoints(edge_id) {
                let &i = node_indices.get(&source).ok_or_else(|| {
                    GraphError::InvalidInput("Source node not found in graph".to_string())
                })?;
                let &j = node_indices.get(&target).ok_or_else(|| {
                    GraphError::InvalidInput("Target node not found in graph".to_string())
                })?;

                if i == j { continue; } // Skip self-loops

                let weight = 1.0; // TODO: get actual edge weight if needed

                // Compute distance vector and magnitude
                let mut diff = vec![0.0; n_dims];
                let mut dist_sq = 0.0;

                for d in 0..n_dims {
                    diff[d] = positions.get_checked(j, d)? - positions.get_checked(i, d)?;
                    dist_sq += diff[d] * diff[d];
                }

                let distance = dist_sq.sqrt().max(1e-6); // Avoid division by zero

                // Spring force: F = k * (d - d0) * direction
                let force_magnitude = attraction_strength * weight * (distance - ideal_distance);

                for d in 0..n_dims {
                    let force_component = force_magnitude * diff[d] / distance;

                    // Apply force to both nodes (opposite directions)
                    let grad_i = gradients.get_checked(i, d)? + force_component;
                    let grad_j = gradients.get_checked(j, d)? - force_component;

                    gradients.set(i, d, grad_i)?;
                    gradients.set(j, d, grad_j)?;
                }
            }
        }

        // 2. Repulsion forces between all pairs
        for i in 0..n_nodes {
            for j in (i + 1)..n_nodes {
                // Compute distance vector and magnitude
                let mut diff = vec![0.0; n_dims];
                let mut dist_sq = 0.0;

                for d in 0..n_dims {
                    diff[d] = positions.get_checked(j, d)? - positions.get_checked(i, d)?;
                    dist_sq += diff[d] * diff[d];
                }

                let distance = dist_sq.sqrt().max(1e-6);

                // Repulsion force: F = k / d^2 * direction
                let force_magnitude = repulsion_strength / (distance * distance);

                for d in 0..n_dims {
                    let force_component = force_magnitude * diff[d] / distance;

                    // Apply repulsive force
                    let grad_i = gradients.get_checked(i, d)? - force_component;
                    let grad_j = gradients.get_checked(j, d)? + force_component;

                    gradients.set(i, d, grad_i)?;
                    gradients.set(j, d, grad_j)?;
                }
            }
        }

        Ok(())
    }

    /// Compute gradients for stress minimization
    fn compute_stress_gradients(
        &self,
        gradients: &mut GraphMatrix,
        positions: &GraphMatrix,
        stress_type: super::StressType,
    ) -> GraphResult<()> {
        // TODO: Implement stress minimization gradients
        Err(GraphError::NotImplemented {
            feature: "Stress minimization".to_string(),
            tracking_issue: None,
        })
    }

    /// Perform optimization using gradient descent with optional annealing
    fn optimize(&self, mut positions: GraphMatrix, graph: &Graph) -> GraphResult<(GraphMatrix, Vec<f64>)> {
        let node_ids: Vec<NodeId> = graph.space().node_ids();
        let node_indices: HashMap<NodeId, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(i, &id)| (id, i))
            .collect();

        let mut energy_history = Vec::new();
        let mut learning_rate = self.learning_rate;

        for iteration in 0..self.iterations {
            // Compute gradients
            let gradients = self.compute_gradients(&positions, graph, &node_indices)?;

            // Compute current energy (for monitoring)
            let energy = self.compute_total_energy(&positions, graph, &node_indices)?;
            energy_history.push(energy);

            // Apply gradients with current learning rate
            for i in 0..positions.shape().0 {
                for j in 0..positions.shape().1 {
                    let pos = positions.get_checked(i, j)?;
                    let grad = gradients.get_checked(i, j)?;
                    let new_pos = pos - learning_rate * grad;
                    positions.set(i, j, new_pos)?;
                }
            }

            // Simulated annealing: reduce learning rate over time
            if self.annealing {
                let progress = iteration as f64 / self.iterations as f64;
                learning_rate = self.learning_rate * (1.0 - progress);
            }

            // Optional: early stopping if energy is not decreasing
            if iteration > 50 && energy_history.len() >= 10 {
                let recent_energies = &energy_history[energy_history.len() - 10..];
                let energy_change = recent_energies.first().unwrap() - recent_energies.last().unwrap();
                if energy_change < 1e-8 {
                    break; // Converged
                }
            }
        }

        Ok((positions, energy_history))
    }

    /// Compute total energy of the current configuration
    fn compute_total_energy(&self, positions: &GraphMatrix, graph: &Graph, node_indices: &HashMap<NodeId, usize>) -> GraphResult<f64> {
        let mut total_energy = 0.0;

        match &self.energy_function {
            Some(EnergyFunction::SpringElectric { attraction_strength, repulsion_strength, ideal_distance }) => {
                // Spring energy from edges
                for edge_id in graph.space().edge_ids() {
                    if let Some((source, target)) = graph.pool().get_edge_endpoints(edge_id) {
                        let &i = node_indices.get(&source).unwrap();
                        let &j = node_indices.get(&target).unwrap();

                        if i == j { continue; }

                        let weight = 1.0; // TODO: get actual edge weight if needed
                        let distance = self.compute_distance(positions, i, j)?;
                        let displacement = distance - ideal_distance;

                        total_energy += 0.5 * attraction_strength * weight * displacement * displacement;
                    }
                }

                // Repulsion energy between all pairs
                let n_nodes = positions.shape().0;
                for i in 0..n_nodes {
                    for j in (i + 1)..n_nodes {
                        let distance = self.compute_distance(positions, i, j)?;
                        total_energy += repulsion_strength / distance;
                    }
                }
            }

            _ => {
                // Default energy computation
                return Ok(0.0);
            }
        }

        Ok(total_energy)
    }

    /// Compute Euclidean distance between two nodes
    fn compute_distance(&self, positions: &GraphMatrix, i: usize, j: usize) -> GraphResult<f64> {
        let mut dist_sq = 0.0;

        for d in 0..positions.shape().1 {
            let diff = positions.get_checked(i, d)? - positions.get_checked(j, d)?;
            dist_sq += diff * diff;
        }

        Ok(dist_sq.sqrt().max(1e-6))
    }
}

impl EmbeddingEngine for EnergyEmbedding {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        self.validate_graph(graph)?;

        if dimensions == 0 {
            return Err(GraphError::InvalidInput(
                "Cannot compute embedding with 0 dimensions".to_string()
            ));
        }

        // Initialize random positions
        let initial_positions = self.initialize_positions(graph.space().node_count(), dimensions)?;

        // Optimize positions using energy minimization
        let (final_positions, energy_history) = self.optimize(initial_positions, graph)?;

        if self.debug_enabled {
            println!("Energy optimization completed. Final energy: {:.6}",
                energy_history.last().unwrap_or(&0.0));
            println!("Energy decrease: {:.6}",
                energy_history.first().unwrap_or(&0.0) - energy_history.last().unwrap_or(&0.0));
        }

        Ok(final_positions)
    }

    fn supports_incremental(&self) -> bool {
        true // Can resume optimization from previous positions
    }

    fn supports_streaming(&self) -> bool {
        false // Requires full graph for energy computation
    }

    fn name(&self) -> &str {
        "energy_nd"
    }

    fn default_dimensions(&self) -> usize {
        8 // Good default for energy-based methods
    }
}

/// Force-directed embedding (specific case of energy embedding)
#[derive(Debug)]
pub struct ForceDirectedEmbedding {
    spring_constant: f64,
    repulsion_strength: f64,
    iterations: usize,
}

impl ForceDirectedEmbedding {
    pub fn new(spring_constant: f64, repulsion_strength: f64, iterations: usize) -> Self {
        Self {
            spring_constant,
            repulsion_strength,
            iterations,
        }
    }
}

impl EmbeddingEngine for ForceDirectedEmbedding {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        let energy_function = EnergyFunction::SpringElectric {
            attraction_strength: self.spring_constant,
            repulsion_strength: self.repulsion_strength,
            ideal_distance: 1.0,
        };

        let energy_embedding = EnergyEmbedding::new(
            self.iterations,
            0.01, // Learning rate
            true, // Annealing
            Some(energy_function),
        );

        energy_embedding.compute_embedding(graph, dimensions)
    }

    fn supports_incremental(&self) -> bool { true }
    fn supports_streaming(&self) -> bool { false }
    fn name(&self) -> &str { "force_directed_nd" }
    fn default_dimensions(&self) -> usize { 8 }
}

/// Builder for energy-based embeddings
pub struct EnergyEmbeddingBuilder {
    iterations: usize,
    learning_rate: f64,
    annealing: bool,
    energy_function: Option<EnergyFunction>,
    seed: Option<u64>,
    debug_enabled: bool,
}

impl EnergyEmbeddingBuilder {
    pub fn new() -> Self {
        Self {
            iterations: 1000,
            learning_rate: 0.01,
            annealing: true,
            energy_function: None,
            seed: None,
            debug_enabled: false,
        }
    }

    pub fn iterations(mut self, iterations: usize) -> Self {
        self.iterations = iterations;
        self
    }

    pub fn learning_rate(mut self, lr: f64) -> Self {
        self.learning_rate = lr;
        self
    }

    pub fn annealing(mut self, enabled: bool) -> Self {
        self.annealing = enabled;
        self
    }

    pub fn energy_function(mut self, func: EnergyFunction) -> Self {
        self.energy_function = Some(func);
        self
    }

    pub fn with_spring_electric(mut self, attraction: f64, repulsion: f64, ideal_distance: f64) -> Self {
        self.energy_function = Some(EnergyFunction::SpringElectric {
            attraction_strength: attraction,
            repulsion_strength: repulsion,
            ideal_distance,
        });
        self
    }

    pub fn seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }

    pub fn debug(mut self, enabled: bool) -> Self {
        self.debug_enabled = enabled;
        self
    }

    pub fn build(self) -> EnergyEmbedding {
        EnergyEmbedding {
            iterations: self.iterations,
            learning_rate: self.learning_rate,
            annealing: self.annealing,
            energy_function: self.energy_function,
            seed: self.seed,
            debug_enabled: self.debug_enabled,
        }
    }

    pub fn compute(self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        let engine = self.build();
        engine.compute_embedding(graph, dimensions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> Graph {
        let mut graph = Graph::new();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node()).collect();
        for i in 0..n-1 {
            graph.add_edge(nodes[i], nodes[i+1]).unwrap();
        }
        graph
    }

    fn cycle_graph(n: usize) -> Graph {
        let mut graph = path_graph(n);
        let nodes: Vec<_> = graph.space().node_ids();
        if n > 2 {
            graph.add_edge(nodes[n-1], nodes[0]).unwrap();
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
        let edges = [(0,1), (0,2), (0,3), (1,2), (1,3), (2,3), (1,7), (2,7), (3,7)];
        for (i, j) in edges.iter() {
            graph.add_edge(nodes[*i], nodes[*j]).unwrap();
        }
        graph
    }

    #[test]
    fn test_energy_embedding_basic() {
        let graph = path_graph(4);
        let builder = EnergyEmbeddingBuilder::new()
            .iterations(100)
            .learning_rate(0.1)
            .seed(42);

        let embedding = builder.compute(&graph, 3);
        assert!(embedding.is_ok());

        let matrix = embedding.unwrap();
        assert_eq!(matrix.shape(), (4, 3));
    }

    #[test]
    fn test_force_directed_embedding() {
        let graph = cycle_graph(5);
        let engine = ForceDirectedEmbedding::new(1.0, 0.5, 200);

        let embedding = engine.compute_embedding(&graph, 4);
        assert!(embedding.is_ok());

        let matrix = embedding.unwrap();
        assert_eq!(matrix.shape(), (5, 4));
    }

    #[test]
    fn test_energy_convergence() {
        let graph = karate_club();
        let engine = EnergyEmbeddingBuilder::new()
            .iterations(500)
            .learning_rate(0.05)
            .annealing(true)
            .with_spring_electric(1.0, 0.1, 1.0)
            .seed(123)
            .debug(false)
            .build();

        let embedding = engine.compute_embedding(&graph, 6);
        assert!(embedding.is_ok());

        let matrix = embedding.unwrap();
        assert_eq!(matrix.shape(), (graph.space().node_count(), 6));

        // Test that different nodes have different positions
        let pos1 = matrix.row(0).unwrap();
        let pos2 = matrix.row(1).unwrap();
        let diff = pos1.subtract(&pos2).unwrap().norm();
        assert!(diff > 1e-6, "Different nodes should have different positions");
    }

    #[test]
    fn test_reproducible_results() {
        let graph = star_graph(6);
        let seed = 999;

        let embedding1 = EnergyEmbeddingBuilder::new()
            .seed(seed)
            .iterations(100)
            .compute(&graph, 3);

        let embedding2 = EnergyEmbeddingBuilder::new()
            .seed(seed)
            .iterations(100)
            .compute(&graph, 3);

        assert!(embedding1.is_ok());
        assert!(embedding2.is_ok());

        let matrix1 = embedding1.unwrap();
        let matrix2 = embedding2.unwrap();

        // Results should be identical with same seed
        let diff = matrix1.subtract(&matrix2).unwrap().frobenius_norm();
        assert!(diff < 1e-10, "Results should be identical with same seed");
    }
}