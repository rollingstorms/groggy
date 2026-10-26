//! Projection algorithms for mapping high-dimensional embeddings to 2D coordinates

use super::ProjectionEngine;
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::matrix::GraphMatrix;
use crate::viz::streaming::data_source::Position;

/// Two-component PCA projection using a scaled feature-space scatter matrix.
#[derive(Debug)]
pub struct PCAProjection {
    center: bool,
    standardize: bool,
}

impl PCAProjection {
    pub fn new(center: bool, standardize: bool) -> Self {
        Self {
            center,
            standardize,
        }
    }

    /// Return owned two-component scores in input row order, retaining row labels.
    /// Standardization uses population deviation; constant features use scale one.
    pub fn project_matrix(&self, embedding: &GraphMatrix) -> GraphResult<GraphMatrix> {
        self.validate_embedding(embedding)?;
        let (rows, cols) = embedding.shape();
        let mut prepared = GraphMatrix::zeros(rows, cols);
        for col in 0..cols {
            let mut scale = 0.0_f64;
            for row in 0..rows {
                let value = embedding.get_checked(row, col)?;
                if !value.is_finite() {
                    return Err(GraphError::InvalidInput(
                        "PCA requires finite input values".into(),
                    ));
                }
                scale = scale.max(value.abs());
            }
            if scale == 0.0 {
                continue;
            }
            // Normalized columns keep mean/deviation arithmetic bounded even
            // when the original values would overflow a sum or square.
            let mut mean = 0.0;
            for row in 0..rows {
                let value = embedding.get_checked(row, col)? / scale;
                mean += (value - mean) / (row + 1) as f64;
            }
            let mut variance = 0.0;
            if self.standardize {
                for row in 0..rows {
                    let difference = embedding.get_checked(row, col)? / scale - mean;
                    variance += difference * difference;
                }
                variance /= rows as f64;
            }
            let deviation = variance.sqrt();
            for row in 0..rows {
                let normalized = embedding.get_checked(row, col)? / scale;
                let value = if self.center {
                    normalized - mean
                } else {
                    normalized
                };
                let prepared_value = if self.standardize && deviation > 0.0 {
                    value / deviation
                } else {
                    // A constant feature is unchanged by scaling alone.
                    value * scale
                };
                if !prepared_value.is_finite() {
                    return Err(GraphError::InvalidInput(
                        "PCA preprocessing exceeds finite f64 range".into(),
                    ));
                }
                prepared.set(row, col, prepared_value)?;
            }
        }
        let mut scaling = 0.0_f64;
        for row in 0..rows {
            for col in 0..cols {
                scaling = scaling.max(prepared.get_checked(row, col)?.abs());
            }
        }
        if scaling == 0.0 {
            scaling = 1.0;
        }
        for row in 0..rows {
            for col in 0..cols {
                prepared.set(row, col, prepared.get_checked(row, col)? / scaling)?;
            }
        }
        // Common positive scaling leaves the principal directions unchanged.
        // Use mean scatter rather than dividing by rows-1: singleton projection
        // is well-defined and scores do not depend on covariance normalization.
        let scatter = prepared
            .transpose()?
            .multiply(&prepared)?
            .scalar_multiply(1.0 / rows as f64)?;
        let (_, vectors) = scatter.eigenvalue_decomposition()?; // descending
        let axes = vectors.select_columns(&[0, 1])?;
        let mut projected = prepared.multiply(&axes)?;
        for row in 0..rows {
            for col in 0..2 {
                let value = projected.get_checked(row, col)? * scaling;
                if !value.is_finite() {
                    return Err(GraphError::InvalidInput(
                        "PCA scores exceed finite f64 range".into(),
                    ));
                }
                projected.set(row, col, value)?;
            }
        }
        projected.set_row_labels(embedding.row_labels().map(<[String]>::to_vec));
        projected.set_column_names(vec!["pc_1".into(), "pc_2".into()]);
        Ok(projected)
    }
}

impl ProjectionEngine for PCAProjection {
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        _graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        let projected = self.project_matrix(embedding)?;

        // Convert to positions
        let (n_samples, _) = projected.shape();
        let mut positions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let x = projected.get_checked(i, 0)?;
            let y = projected.get_checked(i, 1)?;
            positions.push(Position { x, y });
        }

        Ok(positions)
    }

    fn supports_incremental(&self) -> bool {
        false
    }
    fn supports_interpolation(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "pca"
    }
}

pub use super::tsne::TSNEProjection;

pub use super::umap::UMAPProjection;

/// Multi-scale projection combining global and local methods
#[derive(Debug)]
pub struct MultiScaleProjection {
    global_engine: Box<dyn ProjectionEngine>,
    local_engine: Box<dyn ProjectionEngine>,
    global_weight: f64,
}

impl MultiScaleProjection {
    pub fn new(
        global_engine: Box<dyn ProjectionEngine>,
        local_engine: Box<dyn ProjectionEngine>,
        global_weight: f64,
    ) -> Self {
        Self {
            global_engine,
            local_engine,
            global_weight,
        }
    }
}

impl ProjectionEngine for MultiScaleProjection {
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        self.validate_embedding(embedding)?;

        // Get projections from both methods
        let global_positions = self.global_engine.project_embedding(embedding, graph)?;
        let local_positions = self.local_engine.project_embedding(embedding, graph)?;

        // Combine positions with weighted average
        let mut combined_positions = Vec::with_capacity(global_positions.len());
        for (global, local) in global_positions.iter().zip(local_positions.iter()) {
            let x = self.global_weight * global.x + (1.0 - self.global_weight) * local.x;
            let y = self.global_weight * global.y + (1.0 - self.global_weight) * local.y;
            combined_positions.push(Position { x, y });
        }

        Ok(combined_positions)
    }

    fn supports_incremental(&self) -> bool {
        self.global_engine.supports_incremental() && self.local_engine.supports_incremental()
    }

    fn supports_interpolation(&self) -> bool {
        self.global_engine.supports_interpolation() && self.local_engine.supports_interpolation()
    }

    fn name(&self) -> &str {
        "multi_scale"
    }
}

/// Custom matrix projection
#[derive(Debug)]
pub struct CustomMatrixProjection {
    projection_matrix: GraphMatrix,
}

impl CustomMatrixProjection {
    pub fn new(projection_matrix: GraphMatrix) -> Self {
        Self { projection_matrix }
    }
}

impl ProjectionEngine for CustomMatrixProjection {
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        _graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        self.validate_embedding(embedding)?;

        // Apply custom projection matrix
        let projected = embedding.multiply(&self.projection_matrix)?;

        if projected.shape().1 < 2 {
            return Err(GraphError::InvalidInput(
                "Custom projection matrix must produce at least 2 dimensions".to_string(),
            ));
        }

        // Convert to positions
        let (n_samples, _) = projected.shape();
        let mut positions = Vec::with_capacity(n_samples);

        for i in 0..n_samples {
            let x = projected.get_checked(i, 0)?;
            let y = projected.get_checked(i, 1)?;
            positions.push(Position { x, y });
        }

        Ok(positions)
    }

    fn supports_incremental(&self) -> bool {
        true
    }
    fn supports_interpolation(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "custom_matrix"
    }
}

/// Energy-based projection with custom forces
#[derive(Debug)]
pub struct EnergyBasedProjection {
    attraction_strength: f64,
    repulsion_strength: f64,
    iterations: usize,
    learning_rate: f64,
}

impl EnergyBasedProjection {
    pub fn new(
        attraction_strength: f64,
        repulsion_strength: f64,
        iterations: usize,
        learning_rate: f64,
    ) -> Self {
        Self {
            attraction_strength,
            repulsion_strength,
            iterations,
            learning_rate,
        }
    }
}

impl ProjectionEngine for EnergyBasedProjection {
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        _graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        self.validate_embedding(embedding)?;

        let (n_samples, _) = embedding.shape();

        // Initialize random positions
        let mut positions = GraphMatrix::<f64>::zeros(n_samples, 2);
        let mut rng = fastrand::Rng::new();
        for i in 0..n_samples {
            for j in 0..2 {
                positions.set(i, j, rng.f64() * 10.0 - 5.0)?;
            }
        }

        // Compute pairwise distances in high-dimensional space
        let mut hd_distances = GraphMatrix::<f64>::zeros(n_samples, n_samples);
        for i in 0..n_samples {
            for j in i + 1..n_samples {
                let mut dist_sq = 0.0;
                for k in 0..embedding.shape().1 {
                    let diff = embedding.get_checked(i, k)? - embedding.get_checked(j, k)?;
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                hd_distances.set(i, j, dist)?;
                hd_distances.set(j, i, dist)?;
            }
        }

        // Energy minimization
        for _iteration in 0..self.iterations {
            let mut forces = GraphMatrix::<f64>::zeros(n_samples, 2);

            for i in 0..n_samples {
                for j in 0..n_samples {
                    if i != j {
                        let xi = positions.get_checked(i, 0)?;
                        let yi = positions.get_checked(i, 1)?;
                        let xj = positions.get_checked(j, 0)?;
                        let yj = positions.get_checked(j, 1)?;

                        let dx = xj - xi;
                        let dy = yj - yi;
                        let ld_dist = (dx * dx + dy * dy).sqrt().max(1e-12);
                        let hd_dist = hd_distances.get_checked(i, j)?;

                        // Attractive force (spring-like)
                        let spring_force = self.attraction_strength * (ld_dist - hd_dist);
                        let fx_attr = spring_force * dx / ld_dist;
                        let fy_attr = spring_force * dy / ld_dist;

                        // Repulsive force (electrical)
                        let repulsive_force = self.repulsion_strength / (ld_dist * ld_dist);
                        let fx_repel = -repulsive_force * dx / ld_dist;
                        let fy_repel = -repulsive_force * dy / ld_dist;

                        let fx_total = forces.get_checked(i, 0)? + fx_attr + fx_repel;
                        let fy_total = forces.get_checked(i, 1)? + fy_attr + fy_repel;
                        forces.set(i, 0, fx_total)?;
                        forces.set(i, 1, fy_total)?;
                    }
                }
            }

            // Update positions
            for i in 0..n_samples {
                for j in 0..2 {
                    let pos = positions.get_checked(i, j)?;
                    let force = forces.get_checked(i, j)?;
                    positions.set(i, j, pos + self.learning_rate * force)?;
                }
            }
        }

        // Convert to positions
        let mut result_positions = Vec::with_capacity(n_samples);
        for i in 0..n_samples {
            let x = positions.get_checked(i, 0)?;
            let y = positions.get_checked(i, 1)?;
            result_positions.push(Position { x, y });
        }

        Ok(result_positions)
    }

    fn supports_incremental(&self) -> bool {
        false
    }
    fn supports_interpolation(&self) -> bool {
        true
    }
    fn name(&self) -> &str {
        "energy_based"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;

    fn create_test_embedding() -> GraphMatrix {
        let mut embedding = GraphMatrix::<f64>::zeros(5, 3);
        // Simple 3D embedding with some structure
        let data = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
            [0.5, 0.5, 0.5],
        ];

        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                embedding.set(i, j, val).unwrap();
            }
        }
        embedding
    }

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new_manual();
        for _ in 0..5 {
            graph.add_node().unwrap();
        }
        graph
    }

    #[test]
    fn test_pca_projection() {
        let embedding = create_test_embedding();
        let graph = create_test_graph();
        let engine = PCAProjection::new(true, true);

        let positions = engine.project_embedding(&embedding, &graph);
        assert!(positions.is_ok());

        let positions = positions.unwrap();
        assert_eq!(positions.len(), 5);

        // Check that all positions are finite
        for pos in positions {
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
        }
    }

    #[test]
    fn test_tsne_projection() {
        let embedding = create_test_embedding();
        let graph = create_test_graph();
        // The fifth row is equidistant from all four others: perplexity below
        // four cannot be calibrated without inventing tie-breaking distances.
        assert!(TSNEProjection::new(3.0, 100, 100.0, 4.0)
            .project_embedding(&embedding, &graph)
            .is_err());
        let engine = TSNEProjection::new(4.0, 100, 100.0, 4.0);

        let positions = engine.project_embedding(&embedding, &graph);
        assert!(positions.is_ok());

        let positions = positions.unwrap();
        assert_eq!(positions.len(), 5);

        for pos in positions {
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
        }
    }

    #[test]
    fn test_umap_projection() {
        let embedding = create_test_embedding();
        let graph = create_test_graph();
        let engine = UMAPProjection::new(3, 0.1, 100, 5.0);

        let positions = engine.project_embedding(&embedding, &graph);
        assert!(positions.is_ok());

        let positions = positions.unwrap();
        assert_eq!(positions.len(), 5);

        for pos in positions {
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
        }
    }

    #[test]
    fn test_energy_based_projection() {
        let embedding = create_test_embedding();
        let graph = create_test_graph();
        let engine = EnergyBasedProjection::new(1.0, 0.1, 50, 0.01);

        let positions = engine.project_embedding(&embedding, &graph);
        assert!(positions.is_ok());

        let positions = positions.unwrap();
        assert_eq!(positions.len(), 5);

        for pos in positions {
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
        }
    }

    #[test]
    fn test_custom_matrix_projection() {
        let embedding = create_test_embedding();
        let graph = create_test_graph();

        // Create a simple projection matrix (3D -> 2D)
        let mut projection_matrix = GraphMatrix::<f64>::zeros(3, 2);
        projection_matrix.set(0, 0, 1.0).unwrap(); // x component
        projection_matrix.set(1, 1, 1.0).unwrap(); // y component

        let engine = CustomMatrixProjection::new(projection_matrix);

        let positions = engine.project_embedding(&embedding, &graph);
        assert!(positions.is_ok());

        let positions = positions.unwrap();
        assert_eq!(positions.len(), 5);

        for pos in positions {
            assert!(pos.x.is_finite());
            assert!(pos.y.is_finite());
        }
    }
}
