//! Quality metrics and validation for projection systems

use super::QualityConfig;
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::matrix::GraphMatrix;
use crate::viz::streaming::data_source::Position;

/// Quality metrics for evaluating projection quality
#[derive(Debug, Clone)]
pub struct QualityMetrics {
    /// Neighborhood preservation ratio (0.0 to 1.0)
    pub neighborhood_preservation: f64,

    /// Distance correlation between high-D and 2D spaces
    pub distance_correlation: f64,

    /// Stress metric (lower is better)
    pub stress: f64,

    /// Clustering preservation metric
    pub clustering_preservation: Option<f64>,

    /// Local continuity metric
    pub local_continuity: f64,

    /// Global structure preservation
    pub global_structure: f64,

    /// Overall quality score (0.0 to 1.0)
    pub overall_score: f64,
}

/// Quality evaluator for projection systems
#[derive(Debug)]
pub struct QualityEvaluator {
    config: QualityConfig,
}

impl QualityEvaluator {
    /// Create a new quality evaluator
    pub fn new(config: QualityConfig) -> Self {
        Self { config }
    }

    /// Evaluate the quality of a projection
    pub fn evaluate_projection(
        &self,
        original_embedding: &GraphMatrix,
        projected_positions: &[Position],
        graph: &Graph,
    ) -> GraphResult<QualityMetrics> {
        let n = original_embedding.shape().0;
        if projected_positions.len() != n {
            return Err(GraphError::InvalidInput(
                "Number of projected positions must match embedding rows".to_string(),
            ));
        }

        let mut metrics = QualityMetrics {
            neighborhood_preservation: 0.0,
            distance_correlation: 0.0,
            stress: 0.0,
            clustering_preservation: None,
            local_continuity: 0.0,
            global_structure: 0.0,
            overall_score: 0.0,
        };

        // Compute high-dimensional distances
        let hd_distances = self.compute_pairwise_distances(original_embedding)?;

        // Compute 2D distances
        let ld_distances = self.compute_2d_distances(projected_positions)?;

        // Compute individual metrics
        if self.config.compute_neighborhood_preservation {
            metrics.neighborhood_preservation =
                self.compute_neighborhood_preservation(&hd_distances, &ld_distances)?;
        }

        if self.config.compute_distance_preservation {
            metrics.distance_correlation =
                self.compute_distance_correlation(&hd_distances, &ld_distances)?;
            metrics.stress = self.compute_stress(&hd_distances, &ld_distances)?;
        }

        if self.config.compute_clustering_preservation {
            metrics.clustering_preservation = Some(self.compute_clustering_preservation(
                original_embedding,
                projected_positions,
                graph,
            )?);
        }

        // Compute local and global structure metrics
        metrics.local_continuity = self.compute_local_continuity(&hd_distances, &ld_distances)?;
        metrics.global_structure = self.compute_global_structure(&hd_distances, &ld_distances)?;

        // Compute overall quality score
        metrics.overall_score = self.compute_overall_score(&metrics);

        Ok(metrics)
    }

    /// Check if projection meets quality thresholds
    pub fn meets_quality_thresholds(&self, metrics: &QualityMetrics) -> bool {
        let thresholds = &self.config.quality_thresholds;

        metrics.neighborhood_preservation >= thresholds.min_neighborhood_preservation
            && metrics.distance_correlation >= thresholds.min_distance_correlation
            && metrics.stress <= thresholds.max_stress
    }

    /// Generate quality improvement suggestions
    pub fn suggest_improvements(&self, metrics: &QualityMetrics) -> Vec<String> {
        let mut suggestions = Vec::new();
        let thresholds = &self.config.quality_thresholds;

        if metrics.neighborhood_preservation < thresholds.min_neighborhood_preservation {
            suggestions.push(format!(
                "Neighborhood preservation ({:.3}) is below threshold ({:.3}). Consider using UMAP or t-SNE for better local structure preservation.",
                metrics.neighborhood_preservation, thresholds.min_neighborhood_preservation
            ));
        }

        if metrics.distance_correlation < thresholds.min_distance_correlation {
            suggestions.push(format!(
                "Distance correlation ({:.3}) is below threshold ({:.3}). Consider using PCA or MDS for better distance preservation.",
                metrics.distance_correlation, thresholds.min_distance_correlation
            ));
        }

        if metrics.stress > thresholds.max_stress {
            suggestions.push(format!(
                "Stress metric ({:.3}) is above threshold ({:.3}). Consider increasing iterations or using a different projection method.",
                metrics.stress, thresholds.max_stress
            ));
        }

        if metrics.local_continuity < 0.7 {
            suggestions.push(
                "Local continuity is low. Consider increasing the number of nearest neighbors or using smoother interpolation.".to_string()
            );
        }

        if metrics.global_structure < 0.6 {
            suggestions.push(
                "Global structure preservation is low. Consider using PCA or multi-scale projection methods.".to_string()
            );
        }

        if suggestions.is_empty() {
            suggestions
                .push("Projection quality is good! No specific improvements needed.".to_string());
        }

        suggestions
    }

    /// Compute pairwise distances in high-dimensional space
    fn compute_pairwise_distances(&self, embedding: &GraphMatrix) -> GraphResult<GraphMatrix> {
        let (n, d) = embedding.shape();
        let mut distances = GraphMatrix::<f64>::zeros(n, n);

        for i in 0..n {
            for j in i + 1..n {
                let mut dist_sq = 0.0;
                for k in 0..d {
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

    /// Compute pairwise distances in 2D space
    fn compute_2d_distances(&self, positions: &[Position]) -> GraphResult<GraphMatrix> {
        let n = positions.len();
        let mut distances = GraphMatrix::<f64>::zeros(n, n);

        for i in 0..n {
            for j in i + 1..n {
                let dx = positions[i].x - positions[j].x;
                let dy = positions[i].y - positions[j].y;
                let dist = (dx * dx + dy * dy).sqrt();
                distances.set(i, j, dist)?;
                distances.set(j, i, dist)?;
            }
        }

        Ok(distances)
    }

    /// Compute neighborhood preservation metric
    fn compute_neighborhood_preservation(
        &self,
        hd_distances: &GraphMatrix,
        ld_distances: &GraphMatrix,
    ) -> GraphResult<f64> {
        let n = hd_distances.shape().0;
        let k = self.config.k_neighbors.min(n - 1);

        let mut total_preserved = 0;
        let mut total_neighborhoods = 0;

        for i in 0..n {
            // Find k nearest neighbors in high-dimensional space
            let mut hd_neighbors = Vec::new();
            for j in 0..n {
                if i != j {
                    hd_neighbors.push((j, hd_distances.get_checked(i, j)?));
                }
            }
            hd_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            hd_neighbors.truncate(k);

            // Find k nearest neighbors in 2D space
            let mut ld_neighbors = Vec::new();
            for j in 0..n {
                if i != j {
                    ld_neighbors.push((j, ld_distances.get_checked(i, j)?));
                }
            }
            ld_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            ld_neighbors.truncate(k);

            // Count preserved neighbors
            let hd_neighbor_set: std::collections::HashSet<_> =
                hd_neighbors.iter().map(|(idx, _)| *idx).collect();
            let ld_neighbor_set: std::collections::HashSet<_> =
                ld_neighbors.iter().map(|(idx, _)| *idx).collect();

            let preserved = hd_neighbor_set.intersection(&ld_neighbor_set).count();
            total_preserved += preserved;
            total_neighborhoods += k;
        }

        Ok(total_preserved as f64 / total_neighborhoods as f64)
    }

    /// Compute distance correlation between high-D and 2D spaces
    fn compute_distance_correlation(
        &self,
        hd_distances: &GraphMatrix,
        ld_distances: &GraphMatrix,
    ) -> GraphResult<f64> {
        let n = hd_distances.shape().0;
        let mut hd_values = Vec::new();
        let mut ld_values = Vec::new();

        // Collect upper triangular distance values
        for i in 0..n {
            for j in i + 1..n {
                hd_values.push(hd_distances.get_checked(i, j)?);
                ld_values.push(ld_distances.get_checked(i, j)?);
            }
        }

        self.compute_correlation(&hd_values, &ld_values)
    }

    /// Compute stress metric (normalized)
    fn compute_stress(
        &self,
        hd_distances: &GraphMatrix,
        ld_distances: &GraphMatrix,
    ) -> GraphResult<f64> {
        let n = hd_distances.shape().0;
        let mut numerator = 0.0;
        let mut denominator = 0.0;

        for i in 0..n {
            for j in i + 1..n {
                let hd_dist = hd_distances.get_checked(i, j)?;
                let ld_dist = ld_distances.get_checked(i, j)?;
                let diff = hd_dist - ld_dist;
                numerator += diff * diff;
                denominator += hd_dist * hd_dist;
            }
        }

        if denominator > 1e-12 {
            Ok((numerator / denominator).sqrt())
        } else {
            Ok(0.0)
        }
    }

    /// Compute clustering preservation metric
    fn compute_clustering_preservation(
        &self,
        original_embedding: &GraphMatrix,
        projected_positions: &[Position],
        _graph: &Graph,
    ) -> GraphResult<f64> {
        // Simplified clustering preservation using k-means like approach
        let n = original_embedding.shape().0;
        if n < 4 {
            return Ok(1.0); // Too few points for meaningful clustering
        }

        let k = 3.min(n / 2); // Number of clusters

        // Perform simple k-means clustering in high-D space
        let hd_clusters = self.simple_kmeans(original_embedding, k)?;

        // Perform k-means clustering in 2D space
        let ld_embedding = self.positions_to_matrix(projected_positions)?;
        let ld_clusters = self.simple_kmeans(&ld_embedding, k)?;

        // Compute cluster agreement (simplified)
        let mut agreement = 0;
        for i in 0..n {
            for j in i + 1..n {
                let same_hd_cluster = hd_clusters[i] == hd_clusters[j];
                let same_ld_cluster = ld_clusters[i] == ld_clusters[j];
                if same_hd_cluster == same_ld_cluster {
                    agreement += 1;
                }
            }
        }

        let total_pairs = n * (n - 1) / 2;
        Ok(agreement as f64 / total_pairs as f64)
    }

    /// Compute local continuity metric
    fn compute_local_continuity(
        &self,
        hd_distances: &GraphMatrix,
        ld_distances: &GraphMatrix,
    ) -> GraphResult<f64> {
        let n = hd_distances.shape().0;
        let k = self.config.k_neighbors.min(n - 1);

        let mut continuity_sum = 0.0;

        for i in 0..n {
            // Find k nearest neighbors in 2D
            let mut ld_neighbors = Vec::new();
            for j in 0..n {
                if i != j {
                    ld_neighbors.push((j, ld_distances.get_checked(i, j)?));
                }
            }
            ld_neighbors.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            ld_neighbors.truncate(k);

            // Compute average high-D distance to these neighbors
            let mut avg_hd_dist = 0.0;
            for &(j, _) in &ld_neighbors {
                avg_hd_dist += hd_distances.get_checked(i, j)?;
            }
            avg_hd_dist /= k as f64;

            // Find minimum high-D distance among k-nearest 2D neighbors
            let mut min_hd_dist = f64::INFINITY;
            for &(j, _) in &ld_neighbors {
                let hd_dist = hd_distances.get_checked(i, j)?;
                if hd_dist < min_hd_dist {
                    min_hd_dist = hd_dist;
                }
            }

            // Continuity contribution (higher is better)
            if avg_hd_dist > 1e-12 {
                continuity_sum += min_hd_dist / avg_hd_dist;
            }
        }

        Ok(continuity_sum / n as f64)
    }

    /// Compute global structure preservation
    fn compute_global_structure(
        &self,
        hd_distances: &GraphMatrix,
        ld_distances: &GraphMatrix,
    ) -> GraphResult<f64> {
        // Use distance correlation as a proxy for global structure preservation
        self.compute_distance_correlation(hd_distances, ld_distances)
    }

    /// Compute overall quality score
    fn compute_overall_score(&self, metrics: &QualityMetrics) -> f64 {
        let mut score = 0.0;
        let mut weight_sum = 0.0;

        // Weighted combination of metrics
        if self.config.compute_neighborhood_preservation {
            score += 0.3 * metrics.neighborhood_preservation;
            weight_sum += 0.3;
        }

        if self.config.compute_distance_preservation {
            score += 0.2 * metrics.distance_correlation;
            score += 0.2 * (1.0 - metrics.stress).max(0.0); // Convert stress to positive metric
            weight_sum += 0.4;
        }

        if let Some(clustering) = metrics.clustering_preservation {
            score += 0.15 * clustering;
            weight_sum += 0.15;
        }

        score += 0.15 * metrics.local_continuity;
        score += 0.2 * metrics.global_structure;
        weight_sum += 0.35;

        if weight_sum > 1e-12 {
            score / weight_sum
        } else {
            0.0
        }
    }

    /// Compute Pearson correlation coefficient
    fn compute_correlation(&self, x: &[f64], y: &[f64]) -> GraphResult<f64> {
        if x.len() != y.len() || x.is_empty() {
            return Ok(0.0);
        }

        let n = x.len() as f64;
        let mean_x = x.iter().sum::<f64>() / n;
        let mean_y = y.iter().sum::<f64>() / n;

        let mut numerator = 0.0;
        let mut sum_sq_x = 0.0;
        let mut sum_sq_y = 0.0;

        for (xi, yi) in x.iter().zip(y.iter()) {
            let dx = xi - mean_x;
            let dy = yi - mean_y;
            numerator += dx * dy;
            sum_sq_x += dx * dx;
            sum_sq_y += dy * dy;
        }

        let denominator = (sum_sq_x * sum_sq_y).sqrt();
        if denominator > 1e-12 {
            Ok(numerator / denominator)
        } else {
            Ok(0.0)
        }
    }

    /// Simple k-means clustering (for clustering preservation metric)
    fn simple_kmeans(&self, embedding: &GraphMatrix, k: usize) -> GraphResult<Vec<usize>> {
        let (n, d) = embedding.shape();
        if k >= n {
            return Ok((0..n).collect());
        }

        // Initialize centroids randomly
        let mut centroids = Vec::with_capacity(k);
        for i in 0..k {
            let idx = (i * n / k).min(n - 1);
            let mut centroid = vec![0.0; d];
            for j in 0..d {
                centroid[j] = embedding.get_checked(idx, j)?;
            }
            centroids.push(centroid);
        }

        let mut assignments = vec![0; n];

        // Run k-means for a few iterations
        for _ in 0..10 {
            // Assign points to nearest centroid
            for i in 0..n {
                let mut min_dist = f64::INFINITY;
                let mut best_cluster = 0;

                for (cluster_idx, centroid) in centroids.iter().enumerate() {
                    let mut dist_sq = 0.0;
                    for j in 0..d {
                        let diff = embedding.get_checked(i, j)? - centroid[j];
                        dist_sq += diff * diff;
                    }

                    if dist_sq < min_dist {
                        min_dist = dist_sq;
                        best_cluster = cluster_idx;
                    }
                }

                assignments[i] = best_cluster;
            }

            // Update centroids
            for cluster_idx in 0..k {
                let mut new_centroid = vec![0.0; d];
                let mut count = 0;

                for i in 0..n {
                    if assignments[i] == cluster_idx {
                        for j in 0..d {
                            new_centroid[j] += embedding.get_checked(i, j)?;
                        }
                        count += 1;
                    }
                }

                if count > 0 {
                    for j in 0..d {
                        new_centroid[j] /= count as f64;
                    }
                    centroids[cluster_idx] = new_centroid;
                }
            }
        }

        Ok(assignments)
    }

    /// Convert positions to matrix format
    fn positions_to_matrix(&self, positions: &[Position]) -> GraphResult<GraphMatrix> {
        let n = positions.len();
        let mut matrix = GraphMatrix::<f64>::zeros(n, 2);

        for (i, pos) in positions.iter().enumerate() {
            matrix.set(i, 0, pos.x)?;
            matrix.set(i, 1, pos.y)?;
        }

        Ok(matrix)
    }
}

/// Quality improvement optimizer
#[derive(Debug)]
pub struct QualityOptimizer {
    evaluator: QualityEvaluator,
}

impl QualityOptimizer {
    /// Create a new quality optimizer
    pub fn new(config: QualityConfig) -> Self {
        Self {
            evaluator: QualityEvaluator::new(config),
        }
    }

    /// Optimize projection for better quality metrics
    pub fn optimize_projection(
        &self,
        original_embedding: &GraphMatrix,
        initial_positions: &[Position],
        graph: &Graph,
        max_iterations: usize,
    ) -> GraphResult<(Vec<Position>, QualityMetrics)> {
        let mut current_positions = initial_positions.to_vec();
        let mut best_positions = current_positions.clone();
        let mut best_quality =
            self.evaluator
                .evaluate_projection(original_embedding, &current_positions, graph)?;

        for iteration in 0..max_iterations {
            // Try small perturbations to improve quality
            let mut improved = false;
            let step_size = 1.0 / (1.0 + iteration as f64 * 0.1);

            for i in 0..current_positions.len() {
                // Try moving node in different directions
                let original_pos = current_positions[i];

                for &(dx, dy) in &[
                    (step_size, 0.0),
                    (-step_size, 0.0),
                    (0.0, step_size),
                    (0.0, -step_size),
                ] {
                    current_positions[i] = Position {
                        x: original_pos.x + dx,
                        y: original_pos.y + dy,
                    };

                    let quality = self.evaluator.evaluate_projection(
                        original_embedding,
                        &current_positions,
                        graph,
                    )?;

                    if quality.overall_score > best_quality.overall_score {
                        best_positions = current_positions.clone();
                        best_quality = quality;
                        improved = true;
                    }
                }

                // Restore position if no improvement
                current_positions[i] = best_positions[i];
            }

            // Early termination if no improvement
            if !improved {
                break;
            }

            // Check if quality thresholds are met
            if self.evaluator.meets_quality_thresholds(&best_quality) {
                break;
            }
        }

        Ok((best_positions, best_quality))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;

    fn create_test_embedding() -> GraphMatrix {
        let mut embedding = GraphMatrix::<f64>::zeros(4, 3);
        let data = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 1.0, 0.0],
        ];

        for (i, row) in data.iter().enumerate() {
            for (j, &val) in row.iter().enumerate() {
                embedding.set(i, j, val).unwrap();
            }
        }
        embedding
    }

    fn create_test_positions() -> Vec<Position> {
        vec![
            Position { x: 1.0, y: 0.0 },
            Position { x: 0.0, y: 1.0 },
            Position { x: 0.0, y: 0.0 },
            Position { x: 1.0, y: 1.0 },
        ]
    }

    fn create_test_graph() -> Graph {
        let mut graph = Graph::new();
        for _ in 0..4 {
            graph.add_node();
        }
        graph
    }

    #[test]
    fn test_quality_evaluation() {
        let config = QualityConfig::default();
        let evaluator = QualityEvaluator::new(config);

        let embedding = create_test_embedding();
        let positions = create_test_positions();
        let graph = create_test_graph();

        let metrics = evaluator.evaluate_projection(&embedding, &positions, &graph);
        assert!(metrics.is_ok());

        let metrics = metrics.unwrap();
        assert!(metrics.neighborhood_preservation >= 0.0);
        assert!(metrics.neighborhood_preservation <= 1.0);
        assert!(metrics.distance_correlation >= -1.0);
        assert!(metrics.distance_correlation <= 1.0);
        assert!(metrics.stress >= 0.0);
        assert!(metrics.overall_score >= 0.0);
        assert!(metrics.overall_score <= 1.0);
    }

    #[test]
    fn test_distance_computation() {
        let config = QualityConfig::default();
        let evaluator = QualityEvaluator::new(config);

        let embedding = create_test_embedding();
        let distances = evaluator.compute_pairwise_distances(&embedding);
        assert!(distances.is_ok());

        let distances = distances.unwrap();
        assert_eq!(distances.shape(), (4, 4));

        // Check symmetry
        for i in 0..4 {
            for j in 0..4 {
                let dist_ij = distances.get_checked(i, j).unwrap();
                let dist_ji = distances.get_checked(j, i).unwrap();
                assert!((dist_ij - dist_ji).abs() < 1e-10);
            }
        }

        // Check diagonal is zero
        for i in 0..4 {
            assert!(distances.get_checked(i, i).unwrap() < 1e-10);
        }
    }

    #[test]
    fn test_correlation_computation() {
        let config = QualityConfig::default();
        let evaluator = QualityEvaluator::new(config);

        let x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
        let y = vec![2.0, 4.0, 6.0, 8.0, 10.0]; // Perfect correlation

        let correlation = evaluator.compute_correlation(&x, &y).unwrap();
        assert!((correlation - 1.0).abs() < 1e-10);

        let z = vec![5.0, 4.0, 3.0, 2.0, 1.0]; // Perfect negative correlation
        let neg_correlation = evaluator.compute_correlation(&x, &z).unwrap();
        assert!((neg_correlation + 1.0).abs() < 1e-10);
    }

    #[test]
    fn test_quality_thresholds() {
        let mut config = QualityConfig::default();
        config.quality_thresholds.min_neighborhood_preservation = 0.8;
        config.quality_thresholds.min_distance_correlation = 0.7;
        config.quality_thresholds.max_stress = 0.2;

        let evaluator = QualityEvaluator::new(config);

        let good_metrics = QualityMetrics {
            neighborhood_preservation: 0.9,
            distance_correlation: 0.8,
            stress: 0.1,
            clustering_preservation: None,
            local_continuity: 0.8,
            global_structure: 0.8,
            overall_score: 0.8,
        };

        assert!(evaluator.meets_quality_thresholds(&good_metrics));

        let bad_metrics = QualityMetrics {
            neighborhood_preservation: 0.6, // Below threshold
            distance_correlation: 0.8,
            stress: 0.1,
            clustering_preservation: None,
            local_continuity: 0.8,
            global_structure: 0.8,
            overall_score: 0.7,
        };

        assert!(!evaluator.meets_quality_thresholds(&bad_metrics));
    }

    #[test]
    fn test_improvement_suggestions() {
        let config = QualityConfig::default();
        let evaluator = QualityEvaluator::new(config);

        let poor_metrics = QualityMetrics {
            neighborhood_preservation: 0.5, // Below default threshold
            distance_correlation: 0.4,      // Below default threshold
            stress: 0.5,                    // Above default threshold
            clustering_preservation: None,
            local_continuity: 0.6,
            global_structure: 0.5,
            overall_score: 0.5,
        };

        let suggestions = evaluator.suggest_improvements(&poor_metrics);
        assert!(!suggestions.is_empty());
        assert!(suggestions.len() >= 3); // Should have suggestions for all poor metrics
    }
}
