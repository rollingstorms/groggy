//! Energy-based embedding implementations
//!
//! This module implements various energy-based graph embedding methods,
//! including force-directed layouts extended to n dimensions with
//! spring-electric energy optimization. Custom and stress modes reject explicitly.

use super::{row_node_ids, EmbeddingEngine, EnergyFunction};
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::matrix::GraphMatrix;
use crate::types::NodeId;
use std::collections::HashMap;

/// Energy-based embedding engine with customizable energy functions
#[derive(Debug)]
pub struct EnergyEmbedding {
    /// Number of optimization iterations
    iterations: usize,
    /// Learning rate for gradient descent
    learning_rate: f64,
    /// Whether to linearly reduce the gradient-descent learning rate
    annealing: bool,
    /// Custom energy function
    energy_function: Option<EnergyFunction>,
    /// Random seed for reproducible results
    seed: Option<u64>,
    /// Debug data collection
    debug_enabled: bool,
}

impl EnergyEmbedding {
    pub fn new(
        iterations: usize,
        learning_rate: f64,
        annealing: bool,
        energy_function: Option<EnergyFunction>,
    ) -> Self {
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
    fn initialize_positions(
        &self,
        node_count: usize,
        dimensions: usize,
    ) -> GraphResult<GraphMatrix> {
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

    /// Check controls before allocation or optimization, including zero-iteration runs.
    fn parameters(&self) -> GraphResult<(f64, f64, f64)> {
        if !self.learning_rate.is_finite() || self.learning_rate <= 0.0 {
            return Err(invalid("Energy learning rate must be finite and positive"));
        }
        let parameters = match &self.energy_function {
            None => (1.0, 1.0, 1.0),
            Some(EnergyFunction::SpringElectric {
                attraction_strength,
                repulsion_strength,
                ideal_distance,
            }) => (*attraction_strength, *repulsion_strength, *ideal_distance),
            Some(EnergyFunction::StressMinimization { .. }) => {
                return Err(GraphError::NotImplemented {
                    feature: "Energy embedding stress minimization".into(),
                    tracking_issue: None,
                });
            }
            Some(EnergyFunction::Custom { .. }) => {
                return Err(GraphError::NotImplemented {
                    feature: "Custom energy functions".into(),
                    tracking_issue: None,
                });
            }
        };
        if [parameters.0, parameters.1, parameters.2]
            .iter()
            .any(|value| !value.is_finite() || *value < 0.0)
        {
            return Err(invalid(
                "Spring strengths and ideal distance must be finite and nonnegative",
            ));
        }
        Ok(parameters)
    }

    /// Optimize an owned copy of finite coordinates, preserving labeled row order.
    /// Unlabeled rows use ascending active node IDs. Does not change graph state.
    pub fn optimize_positions(
        &self,
        positions: &GraphMatrix,
        graph: &Graph,
    ) -> GraphResult<GraphMatrix> {
        self.validate_graph(graph)?;
        let parameters = self.parameters()?;
        let (rows, dimensions) = positions.shape();
        if dimensions == 0 || rows.checked_mul(dimensions).is_none() {
            return Err(invalid(
                "Energy embedding requires a nonzero, representable dimension count",
            ));
        }
        let ids = row_node_ids(positions, graph)?;
        let node_indices: HashMap<NodeId, usize> =
            ids.iter().enumerate().map(|(row, &id)| (id, row)).collect();
        let mut edge_ids = graph.edge_ids();
        edge_ids.sort_unstable();
        let mut edges = Vec::with_capacity(edge_ids.len());
        for id in edge_ids {
            let (source, target) = graph.edge_endpoints(id)?;
            let row = |id| {
                node_indices
                    .get(&id)
                    .copied()
                    .ok_or_else(|| invalid("Active edge endpoint is absent from embedding rows"))
            };
            edges.push((row(source)?, row(target)?));
        }
        let data = (0..rows)
            .flat_map(|row| (0..dimensions).map(move |axis| positions.get_checked(row, axis)))
            .collect::<GraphResult<Vec<_>>>()?;
        let (data, history) = self.optimize(data, dimensions, &edges, parameters)?;
        if self.debug_enabled {
            println!(
                "Energy optimization: {} accepted steps; initial {:?}, final {:?}",
                history.len().saturating_sub(1),
                history.first(),
                history.last()
            );
        }
        let mut result = GraphMatrix::from_row_major_data(data, rows, dimensions, None)?;
        result.set_row_labels(Some(ids.iter().map(|id| format!("node_{id}")).collect()));
        result.set_column_names(
            (0..dimensions)
                .map(|axis| format!("energy_{axis}"))
                .collect(),
        );
        Ok(result)
    }

    fn optimize(
        &self,
        mut positions: Vec<f64>,
        dimensions: usize,
        edges: &[(usize, usize)],
        parameters: (f64, f64, f64),
    ) -> GraphResult<(Vec<f64>, Vec<f64>)> {
        let (mut energy, mut gradient) =
            spring_electric(&positions, dimensions, edges, parameters)?;
        let mut history = vec![energy];
        for iteration in 0..self.iterations {
            if gradient.iter().all(|value| *value == 0.0) {
                break;
            }
            let mut step = self.learning_rate;
            if self.annealing {
                step *= 1.0 - iteration as f64 / self.iterations as f64;
            }
            let mut accepted = None;
            for _ in 0..64 {
                let candidate: Vec<_> = positions
                    .iter()
                    .zip(&gradient)
                    .map(|(position, gradient)| position - step * gradient)
                    .collect();
                if candidate == positions {
                    return Ok((positions, history));
                }
                if let Ok((next_energy, next_gradient)) =
                    spring_electric(&candidate, dimensions, edges, parameters)
                {
                    if next_energy <= energy {
                        accepted = Some((candidate, next_energy, next_gradient));
                        break;
                    }
                }
                step *= 0.5;
            }
            let Some((candidate, next_energy, next_gradient)) = accepted else {
                return Err(invalid(
                    "Energy line search failed to find a finite nonincreasing step",
                ));
            };
            positions = candidate;
            energy = next_energy;
            gradient = next_gradient;
            history.push(energy);
        }
        Ok((positions, history))
    }
}

fn invalid(message: &str) -> GraphError {
    GraphError::InvalidInput(message.into())
}

/// Smooth pair distance sqrt(||p_i-p_j||² + 1e-6) avoids singular repulsion.
/// Edge instances contribute 0.5*a*(r-d)²; unordered node pairs contribute b/r.
/// Self-loops have zero spring energy. Directed edges act as geometric springs.
fn spring_electric(
    positions: &[f64],
    dimensions: usize,
    edges: &[(usize, usize)],
    (attraction, repulsion, ideal): (f64, f64, f64),
) -> GraphResult<(f64, Vec<f64>)> {
    if positions.iter().any(|value| !value.is_finite()) {
        return Err(invalid("Energy coordinates must be finite"));
    }
    let mut energy = 0.0;
    let mut gradient = vec![0.0; positions.len()];
    let mut pair = |i: usize, j: usize, spring: bool| -> GraphResult<()> {
        if i == j {
            return Ok(());
        }
        let mut squared = 1e-6;
        for axis in 0..dimensions {
            let delta = positions[i * dimensions + axis] - positions[j * dimensions + axis];
            squared += delta * delta;
        }
        if !squared.is_finite() {
            return Err(invalid(
                "Pair distance overflowed; rescale energy coordinates",
            ));
        }
        let distance = squared.sqrt();
        let coefficient = if spring {
            let extension = distance - ideal;
            energy += 0.5 * attraction * extension * extension;
            attraction * (extension / distance)
        } else {
            energy += repulsion / distance;
            -(repulsion / distance) / squared
        };
        for axis in 0..dimensions {
            let delta = positions[i * dimensions + axis] - positions[j * dimensions + axis];
            let force = coefficient * delta;
            gradient[i * dimensions + axis] += force;
            gradient[j * dimensions + axis] -= force;
        }
        Ok(())
    };
    if attraction != 0.0 {
        for &(i, j) in edges {
            pair(i, j, true)?;
        }
    }
    if repulsion != 0.0 {
        for i in 0..positions.len() / dimensions {
            for j in i + 1..positions.len() / dimensions {
                pair(i, j, false)?;
            }
        }
    }
    if !energy.is_finite() || gradient.iter().any(|value| !value.is_finite()) {
        return Err(invalid(
            "Energy or gradient overflowed; rescale coordinates or strengths",
        ));
    }
    Ok((energy, gradient))
}

impl EmbeddingEngine for EnergyEmbedding {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        self.validate_graph(graph)?;

        if dimensions == 0 {
            return Err(GraphError::InvalidInput(
                "Cannot compute embedding with 0 dimensions".to_string(),
            ));
        }

        self.parameters()?;
        if graph
            .space()
            .node_count()
            .checked_mul(dimensions)
            .and_then(|count| count.checked_mul(std::mem::size_of::<f64>()))
            .is_none_or(|bytes| bytes > isize::MAX as usize)
        {
            return Err(invalid("Energy embedding dimensions overflow"));
        }
        let initial_positions =
            self.initialize_positions(graph.space().node_count(), dimensions)?;
        self.optimize_positions(&initial_positions, graph)
    }

    fn supports_incremental(&self) -> bool {
        true // Explicit warm starts through optimize_positions.
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
    seed: Option<u64>,
}

impl ForceDirectedEmbedding {
    pub fn new(spring_constant: f64, repulsion_strength: f64, iterations: usize) -> Self {
        Self {
            spring_constant,
            repulsion_strength,
            iterations,
            seed: None,
        }
    }

    /// Use a local random generator for repeatable initialization.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = Some(seed);
        self
    }
}

impl EmbeddingEngine for ForceDirectedEmbedding {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        let energy_function = EnergyFunction::SpringElectric {
            attraction_strength: self.spring_constant,
            repulsion_strength: self.repulsion_strength,
            ideal_distance: 1.0,
        };

        let mut energy_embedding = EnergyEmbedding::new(
            self.iterations,
            0.01, // Learning rate
            true, // Annealing
            Some(energy_function),
        );

        if let Some(seed) = self.seed {
            energy_embedding = energy_embedding.with_seed(seed);
        }
        energy_embedding.compute_embedding(graph, dimensions)
    }

    fn supports_incremental(&self) -> bool {
        false
    }
    fn supports_streaming(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "force_directed_nd"
    }
    fn default_dimensions(&self) -> usize {
        8
    }
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

impl Default for EnergyEmbeddingBuilder {
    fn default() -> Self {
        Self::new()
    }
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

    pub fn with_spring_electric(
        mut self,
        attraction: f64,
        repulsion: f64,
        ideal_distance: f64,
    ) -> Self {
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
        for i in 0..n.saturating_sub(1) {
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

        assert!((0..6).any(
            |axis| (matrix.get(0, axis).unwrap() - matrix.get(1, axis).unwrap()).abs() > 1e-6
        ));
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

        assert_eq!(
            embedding1.unwrap().to_vec().unwrap(),
            embedding2.unwrap().to_vec().unwrap()
        );
    }
    #[test]
    fn spring_electric_derivatives_match_independent_finite_differences() {
        for dimensions in [1, 2, 5] {
            let data: Vec<_> = (0..4 * dimensions)
                .map(|i| (i as f64 * 1.7).sin())
                .collect();
            // Include parallel, reversed, and self-loop instances.
            let edges = [(0, 1), (0, 1), (1, 0), (2, 3), (2, 2)];
            for parameters in [(1.3, 0.0, 0.7), (0.0, 0.8, 0.7), (1.3, 0.8, 0.7)] {
                let (energy, gradients) =
                    spring_electric(&data, dimensions, &edges, parameters).unwrap();
                let (a, b, ideal) = parameters;
                let reference = |x: &[f64]| {
                    let radius = |u: usize, v: usize| {
                        (1e-6
                            + (0..dimensions)
                                .map(|axis| {
                                    (x[u * dimensions + axis] - x[v * dimensions + axis]).powi(2)
                                })
                                .sum::<f64>())
                        .sqrt()
                    };
                    let spring = edges
                        .iter()
                        .filter(|(u, v)| u != v)
                        .map(|&(u, v)| 0.5 * a * (radius(u, v) - ideal).powi(2))
                        .sum::<f64>();
                    let mut electric = 0.0;
                    for u in 0..4 {
                        for v in u + 1..4 {
                            electric += b / radius(u, v);
                        }
                    }
                    spring + electric
                };
                assert!((energy - reference(&data)).abs() < 1e-12);
                for axis in 0..data.len() {
                    let mut plus = data.clone();
                    let mut minus = data.clone();
                    plus[axis] += 1e-6;
                    minus[axis] -= 1e-6;
                    let finite_difference = (reference(&plus) - reference(&minus)) / 2e-6;
                    assert!(
                        (gradients[axis] - finite_difference).abs()
                            < 1e-5 * finite_difference.abs().max(1.0)
                    );
                }
                // Every force has an equal-and-opposite counterpart.
                for axis in 0..dimensions {
                    assert!(
                        (0..4)
                            .map(|row| gradients[row * dimensions + axis])
                            .sum::<f64>()
                            .abs()
                            < 1e-12
                    );
                }
                let translated: Vec<_> = data.iter().map(|x| x + 7.0).collect();
                let (shifted_energy, shifted_gradient) =
                    spring_electric(&translated, dimensions, &edges, parameters).unwrap();
                assert!((energy - shifted_energy).abs() < 1e-10);
                for (a, b) in gradients.iter().zip(shifted_gradient) {
                    assert!((a - b).abs() < 1e-10);
                }
            }
        }
    }

    #[test]
    fn accepted_steps_reduce_real_energy_with_and_without_annealing() {
        let data = vec![-0.1, 0.05, 0.2, -0.15, 0.04, 0.02];
        let edges = [(0, 1), (1, 2)];
        for annealing in [false, true] {
            // Deliberately large rate exercises backtracking.
            let engine = EnergyEmbedding::new(100, 100.0, annealing, None);
            let (output, history) = engine
                .optimize(data.clone(), 2, &edges, (1.0, 0.1, 1.0))
                .unwrap();
            assert!(history.len() > 2);
            assert!(history.windows(2).all(|pair| pair[1] <= pair[0]));
            assert!(history.last().unwrap() < &history[0]);
            assert_eq!(
                *history.last().unwrap(),
                spring_electric(&output, 2, &edges, (1.0, 0.1, 1.0))
                    .unwrap()
                    .0
            );
        }
        assert!(EnergyEmbedding::new(1, f64::MAX, false, None)
            .optimize(vec![0.0, 1.0], 1, &[(0, 1)], (1.0, 0.0, 0.0))
            .is_err());
        let (energy, gradient) = spring_electric(&[0.0, 0.0], 1, &[], (0.0, 1.0, 1.0)).unwrap();
        assert_eq!(energy, 1000.0);
        assert_eq!(gradient, [0.0, 0.0]); // Smoothed coincident points are stationary.
        assert!(spring_electric(&[f64::MAX, -f64::MAX], 1, &[], (0.0, 1.0, 1.0)).is_err());
    }
}
