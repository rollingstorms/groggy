//! Projection algorithms for mapping high-dimensional embeddings to 2D coordinates

use super::ProjectionEngine;
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::matrix::GraphMatrix;
use crate::viz::streaming::data_source::Position;

/// PCA-based projection (fastest, linear)
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

    /// Compute PCA projection of embedding matrix
    fn compute_pca(&self, embedding: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let (n_samples, n_features) = embedding.shape();

        // Center the data if requested
        let centered = if self.center {
            let means = embedding.column_means()?;
            let mut centered = embedding.clone();
            for i in 0..n_samples {
                for j in 0..n_features {
                    let val = centered.get_checked(i, j)?;
                    let mean = means[j];
                    centered.set(i, j, val - mean)?;
                }
            }
            centered
        } else {
            embedding.clone()
        };

        // Standardize if requested
        let standardized = if self.standardize {
            let variances = centered.column_variances()?;
            let mut standardized = centered;
            for i in 0..n_samples {
                for j in 0..n_features {
                    let val = standardized.get_checked(i, j)?;
                    let std_dev = (variances[j]).sqrt().max(1e-12);
                    standardized.set(i, j, val / std_dev)?;
                }
            }
            standardized
        } else {
            centered
        };

        // Compute covariance matrix
        let transposed = standardized.transpose()?;
        let covariance = transposed.multiply(&standardized)?;
        let scale = 1.0 / (n_samples - 1) as f64;
        let covariance = covariance.scalar_multiply(scale)?;

        // Compute eigendecomposition
        let (eigenvalues, eigenvectors) = covariance.eigenvalue_decomposition()?;

        // Sort eigenvalues in descending order and get top 2 components
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        eigen_pairs.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());

        // Select top 2 principal components
        let pc_indices = vec![eigen_pairs[0].1, eigen_pairs[1].1];
        let principal_components = eigenvectors.select_columns(&pc_indices)?;

        // Project data onto principal components
        let projected = standardized.multiply(&principal_components)?;

        Ok(projected)
    }
}

impl ProjectionEngine for PCAProjection {
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        _graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        self.validate_embedding(embedding)?;

        // Compute PCA projection
        let projected = self.compute_pca(embedding)?;

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

/// t-SNE inspired projection (good for clustering)
#[derive(Debug)]
pub struct TSNEProjection {
    perplexity: f64,
    iterations: usize,
    learning_rate: f64,
    early_exaggeration: f64,
}

impl TSNEProjection {
    pub fn new(
        perplexity: f64,
        iterations: usize,
        learning_rate: f64,
        early_exaggeration: f64,
    ) -> Self {
        Self {
            perplexity,
            iterations,
            learning_rate,
            early_exaggeration,
        }
    }

    /// Compute pairwise distances in high-dimensional space
    fn compute_distances(&self, embedding: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let (n, _) = embedding.shape();
        let mut distances = GraphMatrix::<f64>::zeros(n, n);

        for i in 0..n {
            for j in i + 1..n {
                let mut dist_sq = 0.0;
                for k in 0..embedding.shape().1 {
                    let diff = embedding.get_checked(i, k)? - embedding.get_checked(j, k)?;
                    dist_sq += diff * diff;
                }
                let dist = dist_sq.sqrt();
                distances.set(i, j, dist)?;
                distances.set(j, i, dist)?;
            }
        }

        Ok(distances)
    }

    /// Compute probability matrix from distances
    fn compute_probabilities(&self, distances: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let (n, _) = distances.shape();
        let mut probabilities = GraphMatrix::<f64>::zeros(n, n);

        // For simplicity, use a fixed variance based on perplexity
        // In a full t-SNE implementation, this would be computed per-point
        let variance = self.perplexity / 3.0;

        for i in 0..n {
            let mut row_sum = 0.0;
            for j in 0..n {
                if i != j {
                    let dist = distances.get_checked(i, j)?;
                    let prob = (-dist * dist / (2.0 * variance)).exp();
                    probabilities.set(i, j, prob)?;
                    row_sum += prob;
                }
            }

            // Normalize row
            if row_sum > 1e-12 {
                for j in 0..n {
                    if i != j {
                        let prob = probabilities.get_checked(i, j)?;
                        probabilities.set(i, j, prob / row_sum)?;
                    }
                }
            }
        }

        Ok(probabilities)
    }

    /// Optimize 2D embedding using gradient descent
    fn optimize_embedding(&self, probabilities: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let (n, _) = probabilities.shape();

        // Initialize random 2D positions
        let mut positions = GraphMatrix::<f64>::zeros(n, 2);
        let mut rng = fastrand::Rng::new();
        for i in 0..n {
            for j in 0..2 {
                positions.set(i, j, rng.f64() * 2.0 - 1.0)?;
            }
        }

        // Gradient descent optimization (simplified)
        for iteration in 0..self.iterations {
            let mut gradients = GraphMatrix::<f64>::zeros(n, 2);

            // Compute gradients
            for i in 0..n {
                for j in 0..n {
                    if i != j {
                        let pi = positions.get_checked(i, 0)?;
                        let pj = positions.get_checked(j, 0)?;
                        let qi = positions.get_checked(i, 1)?;
                        let qj = positions.get_checked(j, 1)?;

                        let dx = pi - pj;
                        let dy = qi - qj;
                        let dist_sq = dx * dx + dy * dy + 1e-12;

                        let q_ij = 1.0 / (1.0 + dist_sq);
                        let p_ij = probabilities.get_checked(i, j)?;

                        let factor = 4.0 * (p_ij - q_ij) * q_ij;

                        let grad_x = gradients.get_checked(i, 0)? + factor * dx;
                        let grad_y = gradients.get_checked(i, 1)? + factor * dy;
                        gradients.set(i, 0, grad_x)?;
                        gradients.set(i, 1, grad_y)?;
                    }
                }
            }

            // Update positions
            let current_lr = if iteration < 250 {
                self.learning_rate * self.early_exaggeration
            } else {
                self.learning_rate
            };

            for i in 0..n {
                for j in 0..2 {
                    let pos = positions.get_checked(i, j)?;
                    let grad = gradients.get_checked(i, j)?;
                    positions.set(i, j, pos - current_lr * grad)?;
                }
            }
        }

        Ok(positions)
    }
}

impl ProjectionEngine for TSNEProjection {
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        _graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        self.validate_embedding(embedding)?;

        // Compute pairwise distances
        let distances = self.compute_distances(embedding)?;

        // Compute probability matrix
        let probabilities = self.compute_probabilities(&distances)?;

        // Optimize 2D embedding
        let projected = self.optimize_embedding(&probabilities)?;

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
        "tsne"
    }
}

/// UMAP-inspired projection (balanced global/local preservation)
#[derive(Debug)]
pub struct UMAPProjection {
    n_neighbors: usize,
    min_dist: f64,
    n_epochs: usize,
    negative_sample_rate: f64,
}

impl UMAPProjection {
    pub fn new(
        n_neighbors: usize,
        min_dist: f64,
        n_epochs: usize,
        negative_sample_rate: f64,
    ) -> Self {
        Self {
            n_neighbors,
            min_dist,
            n_epochs,
            negative_sample_rate,
        }
    }

    /// Find k-nearest neighbors for each point
    fn compute_knn_graph(&self, embedding: &GraphMatrix) -> GraphResult<Vec<Vec<(usize, f64)>>> {
        let (n, _) = embedding.shape();
        let mut knn_graph = Vec::with_capacity(n);

        for i in 0..n {
            let mut distances = Vec::new();

            for j in 0..n {
                if i != j {
                    let mut dist_sq = 0.0;
                    for k in 0..embedding.shape().1 {
                        let diff = embedding.get_checked(i, k)? - embedding.get_checked(j, k)?;
                        dist_sq += diff * diff;
                    }
                    distances.push((j, dist_sq.sqrt()));
                }
            }

            // Sort by distance and take k nearest
            distances.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            distances.truncate(self.n_neighbors.min(distances.len()));

            knn_graph.push(distances);
        }

        Ok(knn_graph)
    }

    /// Optimize embedding using UMAP-like forces
    fn optimize_umap(&self, knn_graph: &[Vec<(usize, f64)>]) -> GraphResult<GraphMatrix> {
        let n = knn_graph.len();

        // Initialize random positions
        let mut positions = GraphMatrix::<f64>::zeros(n, 2);
        let mut rng = fastrand::Rng::new();
        for i in 0..n {
            for j in 0..2 {
                positions.set(i, j, rng.f64() * 20.0 - 10.0)?;
            }
        }

        // Optimization loop
        for _epoch in 0..self.n_epochs {
            // Attractive forces (to neighbors)
            for i in 0..n {
                for &(j, _weight) in &knn_graph[i] {
                    let xi = positions.get_checked(i, 0)?;
                    let yi = positions.get_checked(i, 1)?;
                    let xj = positions.get_checked(j, 0)?;
                    let yj = positions.get_checked(j, 1)?;

                    let dx = xj - xi;
                    let dy = yj - yi;
                    let dist = (dx * dx + dy * dy).sqrt().max(1e-12);

                    if dist > self.min_dist {
                        let force = 0.01 * (dist - self.min_dist) / dist;
                        positions.set(i, 0, xi + force * dx)?;
                        positions.set(i, 1, yi + force * dy)?;
                    }
                }
            }

            // Repulsive forces (sample random pairs)
            for _ in 0..(n as f64 * self.negative_sample_rate) as usize {
                let i = rng.usize(0..n);
                let j = rng.usize(0..n);

                if i != j {
                    let xi = positions.get_checked(i, 0)?;
                    let yi = positions.get_checked(i, 1)?;
                    let xj = positions.get_checked(j, 0)?;
                    let yj = positions.get_checked(j, 1)?;

                    let dx = xj - xi;
                    let dy = yj - yi;
                    let dist_sq = dx * dx + dy * dy;

                    if dist_sq > 0.0 {
                        let force = 0.001 / (1.0 + dist_sq);
                        positions.set(i, 0, xi - force * dx)?;
                        positions.set(i, 1, yi - force * dy)?;
                    }
                }
            }
        }

        Ok(positions)
    }
}

impl ProjectionEngine for UMAPProjection {
    fn project_embedding(
        &self,
        embedding: &GraphMatrix,
        _graph: &Graph,
    ) -> GraphResult<Vec<Position>> {
        self.validate_embedding(embedding)?;

        // Compute k-nearest neighbor graph
        let knn_graph = self.compute_knn_graph(embedding)?;

        // Optimize embedding
        let projected = self.optimize_umap(&knn_graph)?;

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
        "umap"
    }
}

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
        let mut graph = Graph::new();
        for _ in 0..5 {
            graph.add_node();
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
        let engine = TSNEProjection::new(3.0, 100, 100.0, 4.0);

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
