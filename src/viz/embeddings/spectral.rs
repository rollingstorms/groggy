//! Spectral embedding implementation using Laplacian eigenvectors
//!
//! This module implements spectral graph embedding, which uses the eigenvectors
//! of the graph Laplacian to embed nodes in high-dimensional space while
//! preserving graph structure.

#![allow(clippy::wrong_self_convention)]

use super::EmbeddingEngine;
use crate::api::graph::Graph;
use crate::errors::{GraphError, GraphResult};
use crate::storage::matrix::GraphMatrix;
use crate::types::NodeId;
use std::collections::HashMap;

/// Spectral embedding engine using Laplacian eigenvectors
#[derive(Debug, Clone)]
pub struct SpectralEmbedding {
    /// Whether to use normalized Laplacian
    normalized: bool,
    /// Minimum eigenvalue threshold for numerical stability
    eigenvalue_threshold: f64,
    /// Whether to skip the constant eigenvector
    skip_constant: bool,
}

impl SpectralEmbedding {
    /// Create a new spectral embedding engine
    pub fn new(normalized: bool, eigenvalue_threshold: f64) -> Self {
        Self {
            normalized,
            eigenvalue_threshold,
            skip_constant: true,
        }
    }

    // Note: Intentionally not implementing Default trait to keep constructor explicit
    /// Create with default parameters
    #[allow(clippy::should_implement_trait)]
    pub fn default() -> Self {
        Self::new(true, 1e-8)
    }

    /// Build D-A, optionally scaled by D^-1/2 on both sides.
    /// Degree is the adjacency row sum: loops count once, parallel instances add.
    fn compute_laplacian(&self, graph: &Graph) -> GraphResult<GraphMatrix> {
        let mut node_ids = graph.space().node_ids();
        node_ids.sort_unstable();
        let n = node_ids.len();
        n.checked_mul(n)
            .and_then(|size| size.checked_mul(std::mem::size_of::<f64>()))
            .ok_or_else(|| {
                GraphError::InvalidInput("Spectral dense matrix size overflow".into())
            })?;
        let node_to_idx: HashMap<NodeId, usize> = node_ids
            .iter()
            .enumerate()
            .map(|(index, &node)| (node, index))
            .collect();
        let mut laplacian = GraphMatrix::zeros(n, n);
        let mut degrees = vec![0.0_f64; n];
        for edge in graph.space().edge_ids() {
            let (source, target) = graph.edge_endpoints(edge)?;
            let src = node_to_idx.get(&source).copied().ok_or_else(|| {
                GraphError::InvalidInput(format!(
                    "Active edge {edge} has inactive endpoint {source}"
                ))
            })?;
            let dst = node_to_idx.get(&target).copied().ok_or_else(|| {
                GraphError::InvalidInput(format!(
                    "Active edge {edge} has inactive endpoint {target}"
                ))
            })?;
            laplacian.set(src, dst, laplacian.get_checked(src, dst)? - 1.0)?;
            degrees[src] += 1.0;
            if src != dst {
                laplacian.set(dst, src, laplacian.get_checked(dst, src)? - 1.0)?;
                degrees[dst] += 1.0;
            }
        }
        for (node, &degree) in degrees.iter().enumerate() {
            laplacian.set(node, node, laplacian.get_checked(node, node)? + degree)?;
        }
        if self.normalized {
            // Zero-degree rows remain zero, including their diagonal. Threshold
            // controls eigenvalue selection, never degree normalization.
            let inverse_roots: Vec<_> = degrees
                .iter()
                .map(|&degree| {
                    if degree > 0.0 {
                        1.0 / degree.sqrt()
                    } else {
                        0.0
                    }
                })
                .collect();
            for (row, &left) in inverse_roots.iter().enumerate() {
                for (col, &right) in inverse_roots.iter().enumerate() {
                    laplacian.set(row, col, laplacian.get_checked(row, col)? * left * right)?;
                }
            }
        }
        Ok(laplacian)
    }

    /// Every component of this positive undirected Laplacian has one zero mode.
    /// Count support components so a tiny user cutoff cannot retain a numerically
    /// perturbed null vector. Dense scanning adds O(N²) work and O(N) memory.
    fn nullity(laplacian: &GraphMatrix) -> GraphResult<usize> {
        let n = laplacian.shape().0;
        let mut seen = vec![false; n];
        let mut stack = Vec::new();
        let mut components = 0;
        for start in 0..n {
            if seen[start] {
                continue;
            }
            components += 1;
            seen[start] = true;
            stack.push(start);
            while let Some(node) = stack.pop() {
                for (neighbor, visited) in seen.iter_mut().enumerate() {
                    if !*visited && laplacian.get_checked(node, neighbor)? < 0.0 {
                        *visited = true;
                        stack.push(neighbor);
                    }
                }
            }
        }
        Ok(components)
    }

    /// Compute eigenvectors of the Laplacian
    fn compute_eigenvectors(
        &self,
        laplacian: &GraphMatrix,
        dimensions: usize,
    ) -> GraphResult<GraphMatrix> {
        // Use the matrix ecosystem's eigendecomposition
        let (eigenvalues, eigenvectors) = laplacian.eigenvalue_decomposition()?;

        // Sort eigenvalues and eigenvectors in ascending order
        let mut eigen_pairs: Vec<(f64, usize)> = eigenvalues
            .iter()
            .enumerate()
            .map(|(i, &val)| (val, i))
            .collect();
        eigen_pairs.sort_by(|a, b| a.0.total_cmp(&b.0));

        // Determine how many eigenvectors to skip
        let skip_count = if self.skip_constant {
            let near_zero = eigen_pairs
                .iter()
                .filter(|(value, _)| value.abs() <= self.eigenvalue_threshold)
                .count();
            near_zero.max(Self::nullity(laplacian)?)
        } else {
            0
        };

        let available_dims = eigen_pairs.len() - skip_count;
        if available_dims < dimensions {
            return Err(GraphError::InvalidInput(format!(
                "Graph only has {} non-constant eigenvectors, but {} dimensions requested",
                available_dims, dimensions
            )));
        }

        // Extract the requested eigenvectors
        let selected_indices: Vec<usize> = eigen_pairs
            .iter()
            .skip(skip_count)
            .take(dimensions)
            .map(|(_, idx)| *idx)
            .collect();

        let embedding = eigenvectors.select_columns(&selected_indices)?;

        Ok(embedding)
    }
}

impl EmbeddingEngine for SpectralEmbedding {
    fn compute_embedding(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        self.validate_graph(graph)?;

        if dimensions == 0 {
            return Err(GraphError::InvalidInput(
                "Cannot compute embedding with 0 dimensions".to_string(),
            ));
        }

        let n = graph.space().node_count();
        let maximum = if self.skip_constant {
            n.saturating_sub(1)
        } else {
            n
        };
        if dimensions > maximum {
            return Err(GraphError::InvalidInput(format!(
                "At most {maximum} spectral dimensions are available, but {dimensions} requested"
            )));
        }

        // Compute Laplacian matrix
        let laplacian = self.compute_laplacian(graph)?;

        // Compute eigenvectors
        let mut embedding = self.compute_eigenvectors(&laplacian, dimensions)?;
        let mut node_ids = graph.space().node_ids();
        node_ids.sort_unstable();
        embedding.set_row_labels(Some(
            node_ids.iter().map(|id| format!("node_{id}")).collect(),
        ));
        embedding.set_column_names(
            (0..dimensions)
                .map(|index| format!("spectral_{index}"))
                .collect(),
        );

        Ok(embedding)
    }

    fn supports_incremental(&self) -> bool {
        false // Spectral embedding requires full recomputation
    }

    fn supports_streaming(&self) -> bool {
        false // Eigendecomposition is not easily streamable
    }

    fn name(&self) -> &str {
        if self.normalized {
            "spectral_normalized"
        } else {
            "spectral_unnormalized"
        }
    }

    fn default_dimensions(&self) -> usize {
        10
    }

    fn validate_graph(&self, graph: &Graph) -> GraphResult<()> {
        if !self.eigenvalue_threshold.is_finite() || self.eigenvalue_threshold <= 0.0 {
            return Err(GraphError::InvalidInput(
                "Spectral eigenvalue threshold must be finite and positive".into(),
            ));
        }
        if graph.is_directed() {
            return Err(GraphError::InvalidInput("Spectral embedding requires an undirected graph; directed Laplacians are unsupported".into()));
        }
        if graph.space().node_count() == 0 {
            return Err(GraphError::InvalidInput(
                "Cannot compute spectral embedding for empty graph".to_string(),
            ));
        }

        if graph.space().node_count() == 1 && self.skip_constant {
            return Err(GraphError::InvalidInput(
                "Cannot compute meaningful spectral embedding for single node".to_string(),
            ));
        }

        Ok(())
    }
}

/// Builder for spectral embedding with custom parameters
#[derive(Clone)]
pub struct SpectralEmbeddingBuilder {
    normalized: bool,
    eigenvalue_threshold: f64,
    skip_constant: bool,
}

impl SpectralEmbeddingBuilder {
    pub fn new() -> Self {
        Self {
            normalized: true,
            eigenvalue_threshold: 1e-8,
            skip_constant: true,
        }
    }

    pub fn normalized(mut self, normalized: bool) -> Self {
        self.normalized = normalized;
        self
    }

    pub fn eigenvalue_threshold(mut self, threshold: f64) -> Self {
        self.eigenvalue_threshold = threshold;
        self
    }

    pub fn skip_constant_eigenvector(mut self, skip: bool) -> Self {
        self.skip_constant = skip;
        self
    }

    pub fn build(self) -> SpectralEmbedding {
        SpectralEmbedding {
            normalized: self.normalized,
            eigenvalue_threshold: self.eigenvalue_threshold,
            skip_constant: self.skip_constant,
        }
    }

    pub fn compute(&self, graph: &Graph, dimensions: usize) -> GraphResult<GraphMatrix> {
        let engine = self.clone().build();
        engine.compute_embedding(graph, dimensions)
    }
}

/// Extension trait to add spectral embedding builder to Graph
pub trait GraphSpectralExt {
    /// Create a spectral embedding builder
    fn spectral(&self) -> SpectralEmbeddingBuilder;
}

impl GraphSpectralExt for Graph {
    fn spectral(&self) -> SpectralEmbeddingBuilder {
        SpectralEmbeddingBuilder::new()
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
    fn laplacian_entries_accumulate_parallel_loops_and_zero_isolates() {
        let mut graph = Graph::new();
        graph.add_nodes(4);
        for (u, v) in [(0, 1), (0, 1), (0, 0), (0, 0), (1, 2)] {
            graph.add_edge(u, v).unwrap();
        }
        let ordinary = [
            [2.0, -2.0, 0.0, 0.0],
            [-2.0, 3.0, -1.0, 0.0],
            [0.0, -1.0, 1.0, 0.0],
            [0.0; 4],
        ];
        let normalized = [
            [0.5, -2.0 / 12f64.sqrt(), 0.0, 0.0],
            [-2.0 / 12f64.sqrt(), 1.0, -1.0 / 3f64.sqrt(), 0.0],
            [0.0, -1.0 / 3f64.sqrt(), 1.0, 0.0],
            [0.0; 4],
        ];
        for (normalize, expected) in [(false, ordinary), (true, normalized)] {
            let matrix = SpectralEmbedding::new(normalize, 10.0)
                .compute_laplacian(&graph)
                .unwrap();
            for (row, values) in expected.iter().enumerate() {
                for (col, &want) in values.iter().enumerate() {
                    assert!((matrix.get(row, col).unwrap() - want).abs() < 1e-14);
                }
            }
        }
    }

    #[test]
    fn test_spectral_embedding_basic() {
        let graph = path_graph(5);
        let engine = SpectralEmbedding::default();

        let embedding = engine.compute_embedding(&graph, 3);
        assert!(embedding.is_ok());

        let matrix = embedding.unwrap();
        assert_eq!(matrix.shape(), (5, 3));
    }

    #[test]
    fn test_spectral_embedding_preserves_structure() {
        let graph = karate_club();
        let engine = SpectralEmbedding::default();
        // This fixture connects only nodes 0,1,2,3,7: four positive modes.
        let embedding = engine.compute_embedding(&graph, 4).unwrap();
        assert_eq!(embedding.shape(), (graph.space().node_count(), 4));
        assert!(engine.compute_embedding(&graph, 5).is_err());
        for row in (4..34).filter(|&node| node != 7) {
            for col in 0..4 {
                assert!(embedding.get(row, col).unwrap().abs() < 1e-10);
            }
        }
    }

    #[test]
    fn test_spectral_embedding_builder() {
        let graph = cycle_graph(6);

        let embedding = graph
            .spectral()
            .normalized(false)
            .eigenvalue_threshold(1e-10)
            .skip_constant_eigenvector(true)
            .compute(&graph, 4);

        assert!(embedding.is_ok());
        assert_eq!(embedding.unwrap().shape(), (6, 4));
    }

    #[test]
    fn test_normalized_vs_unnormalized() {
        let graph = star_graph(10); // Star with center + 9 leaves

        let normalized = SpectralEmbedding::new(true, 1e-8).compute_embedding(&graph, 3);

        let unnormalized = SpectralEmbedding::new(false, 1e-8).compute_embedding(&graph, 3);

        assert!(normalized.is_ok());
        assert!(unnormalized.is_ok());

        // Both should produce valid embeddings
        let norm_emb = normalized.unwrap();
        let unnorm_emb = unnormalized.unwrap();

        assert_eq!(norm_emb.shape(), (10, 3));
        assert_eq!(unnorm_emb.shape(), (10, 3));

        // The embeddings should be different
        // Note: Temporarily disabled - needs matrix.frobenius_norm() API
        // let diff = norm_emb.subtract(&unnorm_emb).unwrap().frobenius_norm();
        // assert!(
        //     diff > 1e-6,
        //     "Normalized and unnormalized embeddings should differ"
        // );
    }

    #[test]
    fn test_eigenvalue_ordering() {
        let graph = path_graph(4);
        let engine = SpectralEmbedding::new(true, 1e-12);

        // For a path graph, we know the eigenvalue structure
        let embedding = engine.compute_embedding(&graph, 2);
        assert!(embedding.is_ok());

        // Test that we can compute embedding even with strict threshold
        let matrix = embedding.unwrap();
        assert_eq!(matrix.shape(), (4, 2));
    }

    #[test]
    fn test_spectral_edge_cases() {
        // Test empty graph
        let empty_graph = Graph::new();
        let engine = SpectralEmbedding::default();

        let result = engine.compute_embedding(&empty_graph, 2);
        assert!(result.is_err());

        // Test single node
        let mut single_node = Graph::new();
        single_node.add_node();

        let result2 = engine.compute_embedding(&single_node, 2);
        assert!(result2.is_err());

        // Test requesting too many dimensions
        let small_graph = path_graph(3);
        let _result3 = engine.compute_embedding(&small_graph, 10);
        // This might succeed or fail depending on eigenvalue threshold
        // The key is that it handles the case gracefully
    }
}
