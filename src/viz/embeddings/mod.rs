//! Multi-dimensional embedding engines for honeycomb layout
//!
//! This module provides various algorithms for embedding graph nodes into
//! high-dimensional space, integrated with the existing matrix ecosystem.

use crate::api::graph::Graph;
use crate::errors::GraphResult;
use crate::storage::matrix::GraphMatrix;

pub mod debug;
pub mod energy;
pub mod flat_embedding;
pub mod random;
pub mod spectral;

/// Core trait for computing high-dimensional node embeddings
pub trait EmbeddingEngine: std::fmt::Debug {
    /// Compute embedding matrix for the given graph
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix>;

    /// Whether this engine supports incremental updates
    fn supports_incremental(&self) -> bool;

    /// Whether this engine supports streaming computation
    fn supports_streaming(&self) -> bool;

    /// Human-readable name for this embedding method
    fn name(&self) -> &str;

    /// Get recommended default dimension count for this method
    fn default_dimensions(&self) -> usize {
        10
    }

    /// Validate that the graph is suitable for this embedding method
    fn validate_graph(&self, graph: &Graph) -> GraphResult<()> {
        if graph.space().node_count() == 0 {
            return Err(crate::errors::GraphError::InvalidInput(
                "Cannot compute embedding for empty graph".to_string(),
            ));
        }
        Ok(())
    }
}

/// Configuration for embedding computation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct EmbeddingConfig {
    /// Embedding method to use
    pub method: EmbeddingMethod,

    /// Number of dimensions in the embedding space
    pub dimensions: usize,

    /// Optional energy function for energy-based methods
    pub energy_function: Option<EnergyFunction>,

    /// Preprocessing transformations to apply to the graph
    pub preprocessing: Vec<GraphTransform>,

    /// Postprocessing transformations to apply to the embedding
    pub postprocessing: Vec<MatrixTransform>,

    /// Whether to enable debug data collection
    pub debug_enabled: bool,

    /// Random seed for reproducible results
    pub seed: Option<u64>,
}

/// Available embedding methods
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EmbeddingMethod {
    /// Spectral embedding using Laplacian eigenvectors
    Spectral {
        /// Whether to use normalized Laplacian
        normalized: bool,
        /// Minimum eigenvalue threshold
        eigenvalue_threshold: f64,
    },

    /// Energy-based optimization with custom energy functions
    EnergyND {
        /// Number of optimization iterations
        iterations: usize,
        /// Learning rate for optimization
        learning_rate: f64,
        /// Enable simulated annealing
        annealing: bool,
    },

    /// Random high-dimensional embedding (for testing)
    RandomND {
        /// Distribution type for random values
        distribution: RandomDistribution,
        /// Whether to normalize the embedding
        normalize: bool,
    },

    /// Force-directed layout extended to n dimensions
    ForceDirectedND {
        /// Spring constant for edges
        spring_constant: f64,
        /// Repulsion strength between nodes
        repulsion_strength: f64,
        /// Number of simulation iterations
        iterations: usize,
    },

    /// Custom embedding provided as a matrix
    CustomMatrix {
        /// Pre-computed embedding matrix
        #[serde(skip)]
        matrix: Box<GraphMatrix>,
    },

    /// Composite embedding combining multiple methods
    Composite {
        /// List of methods to combine
        methods: Vec<(EmbeddingMethod, f64)>, // (method, weight)
        /// How to combine the embeddings
        combination_strategy: CombinationStrategy,
    },
}

/// Energy function for energy-based embeddings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EnergyFunction {
    /// Standard spring-electric model
    SpringElectric {
        attraction_strength: f64,
        repulsion_strength: f64,
        ideal_distance: f64,
    },

    /// Custom energy function
    Custom {
        /// Attraction function: f(distance) -> force
        attraction_fn: String, // Script or formula
        /// Repulsion function: f(distance) -> force
        repulsion_fn: String,
    },

    /// Stress minimization (MDS-like)
    StressMinimization {
        /// Stress function type
        stress_type: StressType,
    },
}

/// Graph preprocessing transformations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum GraphTransform {
    /// Remove isolated nodes
    RemoveIsolated,
    /// Keep only largest connected component
    LargestComponent,
    /// Add artificial edges to ensure connectivity
    EnsureConnected,
    /// Apply edge weight transformations
    TransformWeights(String), // Script or formula
}

/// Matrix postprocessing transformations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum MatrixTransform {
    /// Principal component analysis
    PCA { target_dimensions: usize },
    /// Add Gaussian noise
    AddNoise { stddev: f64 },
    /// L2 normalize each row (node embedding)
    NormalizeRows,
    /// Center the embedding (zero mean)
    Center,
    /// Scale to unit variance
    StandardizeColumns,
    /// Apply custom matrix transformation
    Custom { transform_script: String },
}

/// Random distribution types
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum RandomDistribution {
    Gaussian { mean: f64, stddev: f64 },
    Uniform { min: f64, max: f64 },
    Spherical, // Points on unit sphere
}

/// Strategy for combining multiple embeddings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum CombinationStrategy {
    /// Concatenate embeddings horizontally
    Concatenate,
    /// Weighted average of embeddings
    WeightedAverage,
    /// Take first N dimensions from each embedding
    Interleave { chunk_size: usize },
}

/// Stress function types for MDS-like embeddings
#[derive(Debug, Clone, Copy, serde::Serialize, serde::Deserialize)]
pub enum StressType {
    /// Raw stress: sum((d_ij - delta_ij)^2)
    Raw,
    /// Kruskal stress: sqrt(sum((d_ij - delta_ij)^2) / sum(d_ij^2))
    Kruskal,
    /// Normalized stress
    Normalized,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            method: EmbeddingMethod::Spectral {
                normalized: true,
                eigenvalue_threshold: 1e-8,
            },
            dimensions: 10,
            energy_function: None,
            preprocessing: vec![],
            postprocessing: vec![MatrixTransform::NormalizeRows],
            debug_enabled: false,
            seed: None,
        }
    }
}

/// Factory for creating embedding engines
pub struct EmbeddingEngineFactory;

impl EmbeddingEngineFactory {
    /// Create an embedding engine from configuration
    pub fn create_engine(config: &EmbeddingConfig) -> GraphResult<Box<dyn EmbeddingEngine>> {
        match &config.method {
            EmbeddingMethod::Spectral {
                normalized,
                eigenvalue_threshold,
            } => Ok(Box::new(spectral::SpectralEmbedding::new(
                *normalized,
                *eigenvalue_threshold,
            ))),

            EmbeddingMethod::EnergyND {
                iterations,
                learning_rate,
                annealing,
            } => Ok(Box::new(energy::EnergyEmbedding::new(
                *iterations,
                *learning_rate,
                *annealing,
                config.energy_function.clone(),
            ))),

            EmbeddingMethod::RandomND {
                distribution,
                normalize,
            } => Ok(Box::new(random::RandomEmbedding::new(
                distribution.clone(),
                *normalize,
                config.seed,
            ))),

            EmbeddingMethod::ForceDirectedND {
                spring_constant,
                repulsion_strength,
                iterations,
            } => Ok(Box::new(energy::ForceDirectedEmbedding::new(
                *spring_constant,
                *repulsion_strength,
                *iterations,
            ))),

            EmbeddingMethod::CustomMatrix { matrix } => {
                Ok(Box::new(CustomMatrixEmbedding::new((**matrix).clone())))
            }

            EmbeddingMethod::Composite {
                methods,
                combination_strategy,
            } => Ok(Box::new(CompositeEmbedding::new(
                methods.clone(),
                combination_strategy.clone(),
            ))),
        }
    }

    /// Get list of available embedding methods
    pub fn available_methods() -> Vec<&'static str> {
        vec![
            "spectral",
            "energy_nd",
            "random_nd",
            "force_directed_nd",
            "custom_matrix",
            "composite",
        ]
    }
}

/// Wrapper for custom matrix embeddings
#[derive(Debug)]
struct CustomMatrixEmbedding {
    matrix: GraphMatrix,
}

impl CustomMatrixEmbedding {
    fn new(matrix: GraphMatrix) -> Self {
        Self { matrix }
    }
}

impl EmbeddingEngine for CustomMatrixEmbedding {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        self.validate_graph(graph)?;

        if self.matrix.shape().1 != dimensions {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Custom matrix has {} dimensions, but {} requested",
                self.matrix.shape().1,
                dimensions
            )));
        }

        if self.matrix.shape().0 != graph.space().node_count() {
            return Err(crate::errors::GraphError::InvalidInput(format!(
                "Custom matrix has {} nodes, but graph has {}",
                self.matrix.shape().0,
                graph.space().node_count()
            )));
        }

        Ok(self.matrix.clone())
    }

    fn supports_incremental(&self) -> bool {
        false
    }
    fn supports_streaming(&self) -> bool {
        false
    }
    fn name(&self) -> &str {
        "custom_matrix"
    }
}

/// Composite embedding that combines multiple methods
#[derive(Debug)]
struct CompositeEmbedding {
    methods: Vec<(EmbeddingMethod, f64)>,
    combination_strategy: CombinationStrategy,
}

impl CompositeEmbedding {
    fn new(methods: Vec<(EmbeddingMethod, f64)>, strategy: CombinationStrategy) -> Self {
        Self {
            methods,
            combination_strategy: strategy,
        }
    }
}

impl EmbeddingEngine for CompositeEmbedding {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        self.validate_graph(graph)?;

        // Compute embeddings for each method
        let mut embeddings = Vec::new();
        let mut total_weight = 0.0;

        for (method, weight) in &self.methods {
            let config = EmbeddingConfig {
                method: method.clone(),
                dimensions,
                ..Default::default()
            };

            let engine = EmbeddingEngineFactory::create_engine(&config)?;
            let embedding = engine.compute_embedding(graph, dimensions)?;

            embeddings.push((embedding, *weight));
            total_weight += weight;
        }

        // Combine embeddings according to strategy
        match &self.combination_strategy {
            CombinationStrategy::Concatenate => {
                let matrices: Vec<_> = embeddings.into_iter().map(|(m, _)| m).collect();
                GraphMatrix::concatenate_columns(matrices)
            }

            CombinationStrategy::WeightedAverage => {
                let mut result = embeddings[0].0.clone();
                result = result.scalar_multiply(embeddings[0].1 / total_weight)?;

                for (embedding, weight) in embeddings.into_iter().skip(1) {
                    let weighted = embedding.scalar_multiply(weight / total_weight)?;
                    result = result.add(&weighted)?;
                }

                Ok(result)
            }

            CombinationStrategy::Interleave { chunk_size: _ } => {
                // TODO: Implement interleaving strategy
                todo!("Interleave combination strategy not yet implemented")
            }
        }
    }

    fn supports_incremental(&self) -> bool {
        // Only if all constituent methods support incremental updates
        self.methods.iter().all(|(_method, _)| {
            // This is simplified - would need to check each method properly
            false
        })
    }

    fn supports_streaming(&self) -> bool {
        false
    }

    fn name(&self) -> &str {
        "composite"
    }

    fn default_dimensions(&self) -> usize {
        // Use the max default dimensions of constituent methods
        self.methods.iter().map(|(_, _)| 10).max().unwrap_or(10)
    }
}

/// Extension trait to add embedding methods to Graph
pub trait GraphEmbeddingExt {
    /// Compute spectral embedding of the graph
    fn spectral_embedding(&self, dimensions: usize) -> GraphResult<GraphMatrix>;

    /// Compute energy-based embedding
    fn energy_embedding(&self) -> energy::EnergyEmbeddingBuilder;

    /// Compute random embedding (for testing)
    fn random_embedding(&self, dimensions: usize) -> GraphResult<GraphMatrix>;

    /// Compute embedding using the specified configuration
    fn compute_embedding(&self, config: &EmbeddingConfig) -> GraphResult<GraphMatrix>;
}

impl GraphEmbeddingExt for Graph {
    fn spectral_embedding(&self, dimensions: usize) -> GraphResult<GraphMatrix> {
        let engine = spectral::SpectralEmbedding::new(true, 1e-8);
        engine.compute_embedding(self, dimensions)
    }

    fn energy_embedding(&self) -> energy::EnergyEmbeddingBuilder {
        energy::EnergyEmbeddingBuilder::new()
    }

    fn random_embedding(&self, dimensions: usize) -> GraphResult<GraphMatrix> {
        let engine = random::RandomEmbedding::new(
            RandomDistribution::Gaussian {
                mean: 0.0,
                stddev: 1.0,
            },
            true,
            None,
        );
        engine.compute_embedding(self, dimensions)
    }

    fn compute_embedding(&self, config: &EmbeddingConfig) -> GraphResult<GraphMatrix> {
        let engine = EmbeddingEngineFactory::create_engine(config)?;
        engine.compute_embedding(self, config.dimensions)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn path_graph(n: usize) -> Graph {
        let mut graph = Graph::new();
        let nodes: Vec<_> = (0..n).map(|_| graph.add_node()).collect();
        for i in 0..n - 1 {
            graph.add_edge(nodes[i], nodes[i + 1]).unwrap();
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
    fn test_embedding_factory() {
        let config = EmbeddingConfig::default();
        let engine = EmbeddingEngineFactory::create_engine(&config);
        assert!(engine.is_ok());
        assert_eq!(engine.unwrap().name(), "spectral_normalized");
    }

    #[test]
    fn test_graph_extension_trait() {
        let graph = karate_club();

        // Test spectral embedding
        let embedding = graph.spectral_embedding(5);
        assert!(embedding.is_ok());

        let matrix = embedding.unwrap();
        assert_eq!(matrix.shape(), (graph.space().node_count(), 5));

        // Test random embedding
        let random = graph.random_embedding(8);
        assert!(random.is_ok());

        let random_matrix = random.unwrap();
        assert_eq!(random_matrix.shape(), (graph.space().node_count(), 8));
    }

    #[test]
    fn test_custom_matrix_embedding() {
        let graph = path_graph(5);

        // Create a custom embedding matrix
        let custom_matrix = GraphMatrix::zeros(5, 3);
        let engine = CustomMatrixEmbedding::new(custom_matrix);

        let result = engine.compute_embedding(&graph, 3);
        assert!(result.is_ok());

        let embedding = result.unwrap();
        assert_eq!(embedding.shape(), (5, 3));
    }
}
