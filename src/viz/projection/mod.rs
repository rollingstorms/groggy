//! Phase 2: Multi-dimensional to 2D Honeycomb Projection System
//!
//! This module implements sophisticated projection algorithms that map high-dimensional
//! node embeddings to 2D honeycomb grid coordinates while preserving neighborhood
//! relationships and enabling smooth real-time transitions.

#![allow(clippy::wrong_self_convention)]

use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::matrix::GraphMatrix;
use crate::viz::streaming::data_source::Position;

pub mod algorithms;
pub mod honeycomb;
pub mod interpolation;
pub mod quality;

/// Core trait for projecting high-dimensional embeddings to 2D coordinates
pub trait ProjectionEngine: std::fmt::Debug {
    /// Project a high-dimensional embedding matrix to 2D coordinates
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        graph: &Graph,
    ) -> GraphResult<Vec<Position>>;

    /// Whether this projection supports incremental updates
    fn supports_incremental(&self) -> bool;

    /// Whether this projection supports smooth interpolation
    fn supports_interpolation(&self) -> bool;

    /// Human-readable name for this projection method
    fn name(&self) -> &str;

    /// Validate that the embedding is suitable for this projection method
    fn validate_embedding(&self, embedding: &GraphMatrix) -> GraphResult<()> {
        if embedding.shape().0 == 0 {
            return Err(GraphError::InvalidInput(
                "Cannot project empty embedding matrix".to_string(),
            ));
        }

        if embedding.shape().1 < 2 {
            return Err(GraphError::InvalidInput(format!(
                "Cannot project {}-dimensional embedding to 2D",
                embedding.shape().1
            )));
        }

        Ok(())
    }
}

/// Configuration for projection computation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct ProjectionConfig {
    /// Projection method to use
    pub method: ProjectionMethod,

    /// Target honeycomb grid parameters
    pub honeycomb_config: HoneycombConfig,

    /// Quality preservation settings
    pub quality_config: QualityConfig,

    /// Interpolation settings for smooth transitions
    pub interpolation_config: InterpolationConfig,

    /// Whether to enable debug data collection
    pub debug_enabled: bool,

    /// Random seed for reproducible results
    pub seed: Option<u64>,
}

/// Available projection methods
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum ProjectionMethod {
    /// Principal Component Analysis projection
    PCA {
        /// Whether to center the data
        center: bool,
        /// Whether to standardize components
        standardize: bool,
    },

    /// t-SNE inspired projection with perplexity
    TSNE {
        /// Perplexity parameter for local neighborhood size
        perplexity: f64,
        /// Number of optimization iterations
        iterations: usize,
        /// Learning rate for optimization
        learning_rate: f64,
        /// Early exaggeration factor
        early_exaggeration: f64,
    },

    /// UMAP-inspired projection
    UMAP {
        /// Number of nearest neighbors
        n_neighbors: usize,
        /// Minimum distance between points
        min_dist: f64,
        /// Number of optimization epochs
        n_epochs: usize,
        /// Negative sampling rate
        negative_sample_rate: f64,
    },

    /// Multi-scale projection combining multiple methods
    MultiScale {
        /// Global structure preservation method
        global_method: Box<ProjectionMethod>,
        /// Local structure preservation method
        local_method: Box<ProjectionMethod>,
        /// Balance between global and local (0.0 = local, 1.0 = global)
        global_weight: f64,
    },

    /// Custom projection matrix
    CustomMatrix {
        /// Pre-computed 2D projection matrix
        #[serde(skip)]
        projection_matrix: Box<GraphMatrix>,
    },

    /// Energy-based projection with custom forces
    EnergyBased {
        /// Attraction function for similar nodes
        attraction_strength: f64,
        /// Repulsion function for dissimilar nodes
        repulsion_strength: f64,
        /// Number of optimization iterations
        iterations: usize,
        /// Learning rate
        learning_rate: f64,
    },
}

/// Honeycomb grid configuration
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct HoneycombConfig {
    /// Size of each hexagonal cell in pixels
    pub cell_size: f64,

    /// Grid layout strategy
    pub layout_strategy: HoneycombLayoutStrategy,

    /// Whether to snap nodes to exact hex centers
    pub snap_to_centers: bool,

    /// Padding around the grid
    pub grid_padding: f64,

    /// Maximum grid dimensions (auto-computed if None)
    pub max_grid_size: Option<(usize, usize)>,

    /// Enable automatic sizing of hexagonal cells based on embedding extents
    pub auto_cell_size: bool,

    /// Target grid resolution when auto sizing
    pub target_cols: usize,
    pub target_rows: usize,

    /// Multiplier applied after auto sizing (1.0 = exact fit)
    pub scale_multiplier: f64,

    /// Target average number of nodes per hex cell (legacy fallback for auto scaling)
    pub target_avg_occupancy: f64,

    /// Minimum allowed cell size to avoid degenerate grids
    pub min_cell_size: f64,
}

/// Strategy for laying out nodes on the honeycomb grid
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum HoneycombLayoutStrategy {
    /// Spiral outward from center
    Spiral,
    /// Fill by density (denser areas get priority)
    DensityBased,
    /// Preserve relative distances as much as possible
    DistancePreserving,
    /// Energy-based optimization with unique hex assignment
    EnergyBased,
    /// Custom ordering function
    Custom { ordering_fn: String },
}

/// Quality metrics and preservation settings
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityConfig {
    /// Whether to compute neighborhood preservation metrics
    pub compute_neighborhood_preservation: bool,

    /// Whether to compute distance preservation metrics
    pub compute_distance_preservation: bool,

    /// Whether to compute clustering preservation metrics
    pub compute_clustering_preservation: bool,

    /// Number of nearest neighbors to consider for quality metrics
    pub k_neighbors: usize,

    /// Whether to enable quality-based optimization
    pub optimize_for_quality: bool,

    /// Target quality thresholds
    pub quality_thresholds: QualityThresholds,
}

/// Quality metric thresholds
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct QualityThresholds {
    /// Minimum neighborhood preservation (0.0 - 1.0)
    pub min_neighborhood_preservation: f64,
    /// Minimum distance correlation
    pub min_distance_correlation: f64,
    /// Maximum stress metric
    pub max_stress: f64,
}

/// Interpolation configuration for smooth transitions
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct InterpolationConfig {
    /// Whether to enable smooth interpolation between projections
    pub enable_interpolation: bool,

    /// Interpolation method
    pub method: InterpolationMethod,

    /// Number of interpolation steps
    pub steps: usize,

    /// Easing function for interpolation
    pub easing: EasingFunction,

    /// Whether to preserve honeycomb constraints during interpolation
    pub preserve_honeycomb: bool,
}

/// Interpolation methods
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum InterpolationMethod {
    /// Linear interpolation
    Linear,
    /// Bezier curve interpolation
    Bezier { control_points: Vec<Position> },
    /// Spline interpolation
    Spline,
    /// Physics-based spring interpolation
    SpringPhysics { damping: f64, stiffness: f64 },
}

/// Easing functions for smooth animations
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub enum EasingFunction {
    Linear,
    EaseIn,
    EaseOut,
    EaseInOut,
    Bounce,
    Elastic,
    Custom { function: String },
}

impl Default for ProjectionConfig {
    fn default() -> Self {
        Self {
            method: ProjectionMethod::PCA {
                center: true,
                standardize: true,
            },
            honeycomb_config: HoneycombConfig::default(),
            quality_config: QualityConfig::default(),
            interpolation_config: InterpolationConfig::default(),
            debug_enabled: false,
            seed: None,
        }
    }
}

impl Default for HoneycombConfig {
    fn default() -> Self {
        Self {
            cell_size: 40.0,
            layout_strategy: HoneycombLayoutStrategy::EnergyBased,
            snap_to_centers: true,
            grid_padding: 20.0,
            max_grid_size: None,
            auto_cell_size: true,
            target_cols: 64,
            target_rows: 48,
            scale_multiplier: 1.1,
            target_avg_occupancy: 1.0,
            min_cell_size: 6.0,
        }
    }
}

impl Default for QualityConfig {
    fn default() -> Self {
        Self {
            compute_neighborhood_preservation: true,
            compute_distance_preservation: true,
            compute_clustering_preservation: false,
            k_neighbors: 10,
            optimize_for_quality: false,
            quality_thresholds: QualityThresholds::default(),
        }
    }
}

impl Default for QualityThresholds {
    fn default() -> Self {
        Self {
            min_neighborhood_preservation: 0.7,
            min_distance_correlation: 0.6,
            max_stress: 0.3,
        }
    }
}

impl Default for InterpolationConfig {
    fn default() -> Self {
        Self {
            enable_interpolation: true,
            method: InterpolationMethod::Linear,
            steps: 30,
            easing: EasingFunction::EaseInOut,
            preserve_honeycomb: true,
        }
    }
}

/// Factory for creating projection engines
pub struct ProjectionEngineFactory;

impl ProjectionEngineFactory {
    /// Create a projection engine from configuration
    pub fn create_engine(config: &ProjectionConfig) -> GraphResult<Box<dyn ProjectionEngine>> {
        match &config.method {
            ProjectionMethod::PCA {
                center,
                standardize,
            } => Ok(Box::new(algorithms::PCAProjection::new(
                *center,
                *standardize,
            ))),

            ProjectionMethod::TSNE {
                perplexity,
                iterations,
                learning_rate,
                early_exaggeration,
            } => Ok(Box::new(algorithms::TSNEProjection::new(
                *perplexity,
                *iterations,
                *learning_rate,
                *early_exaggeration,
            ))),

            ProjectionMethod::UMAP {
                n_neighbors,
                min_dist,
                n_epochs,
                negative_sample_rate,
            } => Ok(Box::new(algorithms::UMAPProjection::new(
                *n_neighbors,
                *min_dist,
                *n_epochs,
                *negative_sample_rate,
            ))),

            ProjectionMethod::MultiScale {
                global_method,
                local_method,
                global_weight,
            } => {
                let global_engine = Self::create_engine(&ProjectionConfig {
                    method: (**global_method).clone(),
                    ..config.clone()
                })?;
                let local_engine = Self::create_engine(&ProjectionConfig {
                    method: (**local_method).clone(),
                    ..config.clone()
                })?;
                Ok(Box::new(algorithms::MultiScaleProjection::new(
                    global_engine,
                    local_engine,
                    *global_weight,
                )))
            }

            ProjectionMethod::CustomMatrix { projection_matrix } => Ok(Box::new(
                algorithms::CustomMatrixProjection::new((**projection_matrix).clone()),
            )),

            ProjectionMethod::EnergyBased {
                attraction_strength,
                repulsion_strength,
                iterations,
                learning_rate,
            } => Ok(Box::new(algorithms::EnergyBasedProjection::new(
                *attraction_strength,
                *repulsion_strength,
                *iterations,
                *learning_rate,
            ))),
        }
    }

    /// Get list of available projection methods
    pub fn available_methods() -> Vec<&'static str> {
        vec![
            "pca",
            "tsne",
            "umap",
            "multi_scale",
            "custom_matrix",
            "energy_based",
        ]
    }
}

/// Extension trait to add projection methods to Graph
pub trait GraphProjectionExt {
    /// Project high-dimensional embedding to 2D honeycomb coordinates
    fn project_to_honeycomb(
        &self,
        embedding: &GraphMatrix,
        config: &ProjectionConfig,
    ) -> GraphResult<Vec<Position>>;

    /// Project using PCA (quick and simple)
    fn project_pca(&self, embedding: &GraphMatrix) -> GraphResult<Vec<Position>>;

    /// Project using t-SNE inspired method (good for clustering)
    fn project_tsne(&self, embedding: &GraphMatrix, perplexity: f64) -> GraphResult<Vec<Position>>;

    /// Project using UMAP inspired method (balanced global/local)
    fn project_umap(
        &self,
        embedding: &GraphMatrix,
        n_neighbors: usize,
    ) -> GraphResult<Vec<Position>>;
}

impl GraphProjectionExt for Graph {
    fn project_to_honeycomb(
        &self,
        embedding: &GraphMatrix,
        config: &ProjectionConfig,
    ) -> GraphResult<Vec<Position>> {
        let engine = ProjectionEngineFactory::create_engine(config)?;
        engine.project_embedding(embedding, self)
    }

    fn project_pca(&self, embedding: &GraphMatrix) -> GraphResult<Vec<Position>> {
        let config = ProjectionConfig {
            method: ProjectionMethod::PCA {
                center: true,
                standardize: true,
            },
            ..Default::default()
        };
        self.project_to_honeycomb(embedding, &config)
    }

    fn project_tsne(&self, embedding: &GraphMatrix, perplexity: f64) -> GraphResult<Vec<Position>> {
        let config = ProjectionConfig {
            method: ProjectionMethod::TSNE {
                perplexity,
                iterations: 1000,
                learning_rate: 200.0,
                early_exaggeration: 12.0,
            },
            ..Default::default()
        };
        self.project_to_honeycomb(embedding, &config)
    }

    fn project_umap(
        &self,
        embedding: &GraphMatrix,
        n_neighbors: usize,
    ) -> GraphResult<Vec<Position>> {
        let config = ProjectionConfig {
            method: ProjectionMethod::UMAP {
                n_neighbors,
                min_dist: 0.1,
                n_epochs: 500,
                negative_sample_rate: 5.0,
            },
            ..Default::default()
        };
        self.project_to_honeycomb(embedding, &config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_projection_config_defaults() {
        let config = ProjectionConfig::default();
        assert!(matches!(config.method, ProjectionMethod::PCA { .. }));
        assert_eq!(config.honeycomb_config.cell_size, 40.0);
        assert!(config.quality_config.compute_neighborhood_preservation);
        assert!(config.interpolation_config.enable_interpolation);
    }

    #[test]
    fn test_projection_factory() {
        let config = ProjectionConfig::default();
        let engine = ProjectionEngineFactory::create_engine(&config);
        assert!(engine.is_ok());
        assert_eq!(engine.unwrap().name(), "pca");
    }

    #[test]
    fn test_available_methods() {
        let methods = ProjectionEngineFactory::available_methods();
        assert!(methods.contains(&"pca"));
        assert!(methods.contains(&"tsne"));
        assert!(methods.contains(&"umap"));
        assert_eq!(methods.len(), 6);
    }
}
