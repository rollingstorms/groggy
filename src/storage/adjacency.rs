//! Adjacency Matrix Operations - Unified with GraphMatrix
//!
//! This module provides adjacency matrix operations as specialized GraphMatrix instances.
//! All adjacency matrices are now GraphMatrix objects with specialized constructors and methods.
//!
//! # Unified Architecture
//! - All adjacency matrices are GraphMatrix instances (no separate type)
//! - Inherits all statistical operations from GraphMatrix/NumArray
//! - Consistent API with the rest of the matrix system
//! - Eliminates code duplication
//!
//! # Migration Guide
//! ```rust,ignore
//! // OLD: Separate AdjacencyMatrix type
//! let adj = AdjacencyMatrix::new(nodes, edges);
//!
//! // NEW: GraphMatrix with adjacency constructor
//! let adj = GraphMatrix::adjacency_from_edges(nodes, edges);
//! let laplacian = adj.to_laplacian();
//! let stats = adj.sum_axis(Axis::Rows);
//! ```

use crate::errors::GraphResult;
use crate::storage::matrix::GraphMatrix;
use crate::types::NodeId;

/// Matrix storage format options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixFormat {
    Dense,
    Sparse,
}

/// Matrix type for specialized operations
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixType {
    Adjacency,
    Laplacian { normalized: bool },
    Transition,
}

/// Type alias for adjacency matrices - they are just GraphMatrix instances
pub type AdjacencyMatrix = GraphMatrix;

/// Builder for creating adjacency matrices using the unified GraphMatrix system
pub struct AdjacencyMatrixBuilder;

impl AdjacencyMatrixBuilder {
    /// Create unweighted adjacency matrix from edge list
    pub fn from_edges(nodes: &[NodeId], edges: &[(NodeId, NodeId)]) -> GraphResult<GraphMatrix> {
        GraphMatrix::adjacency_from_edges(nodes, edges)
    }

    /// Create weighted adjacency matrix from weighted edge list
    pub fn from_weighted_edges(
        nodes: &[NodeId],
        weighted_edges: &[(NodeId, NodeId, f64)],
    ) -> GraphResult<GraphMatrix> {
        GraphMatrix::weighted_adjacency_from_edges(nodes, weighted_edges)
    }

    /// Create Laplacian matrix directly from edges
    pub fn laplacian_from_edges(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId)],
    ) -> GraphResult<GraphMatrix> {
        let adjacency = Self::from_edges(nodes, edges)?;
        adjacency.to_laplacian()
    }

    /// Create normalized Laplacian matrix directly from edges
    pub fn normalized_laplacian_from_edges(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId)],
    ) -> GraphResult<GraphMatrix> {
        let adjacency = Self::from_edges(nodes, edges)?;
        adjacency.to_normalized_laplacian(0.5, 1)
    }
}

/// Convenience functions for adjacency matrix operations
impl GraphMatrix {
    /// Check if this matrix represents a valid adjacency matrix
    pub fn is_valid_adjacency(&self) -> bool {
        self.is_adjacency_matrix()
    }

    /// Get node degrees (row sums for adjacency matrix)
    pub fn node_degrees(&self) -> GraphResult<crate::storage::array::NumArray<f64>> {
        Ok(crate::storage::array::NumArray::new(
            self.sum_axis(crate::storage::matrix::Axis::Rows)?,
        ))
    }

    /// Get in-degrees (column sums for directed adjacency matrix)
    pub fn in_degrees(&self) -> GraphResult<crate::storage::array::NumArray<f64>> {
        Ok(crate::storage::array::NumArray::new(
            self.sum_axis(crate::storage::matrix::Axis::Columns)?,
        ))
    }

    /// Get out-degrees (same as node_degrees for adjacency matrix)
    pub fn out_degrees(&self) -> GraphResult<crate::storage::array::NumArray<f64>> {
        self.node_degrees()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_adjacency_matrix_unified() {
        let nodes = vec![1, 2, 3];
        let edges = vec![(1, 2), (2, 3), (1, 3)];

        // Create adjacency matrix using unified system
        let adj = AdjacencyMatrixBuilder::from_edges(&nodes, &edges).unwrap();

        // Verify it's a GraphMatrix
        assert_eq!(adj.shape(), (3, 3));
        assert!(adj.is_square());
        assert!(adj.is_valid_adjacency());

        // Test statistical operations work
        let degrees = adj.node_degrees().unwrap();
        assert_eq!(degrees.len(), 3);

        // Test Laplacian conversion
        let laplacian = adj.to_laplacian().unwrap();
        assert_eq!(laplacian.shape(), (3, 3));
    }

    #[test]
    fn test_weighted_adjacency_matrix() {
        let nodes = vec![1, 2, 3];
        let weighted_edges = vec![(1, 2, 0.5), (2, 3, 1.5), (1, 3, 2.0)];

        let adj = AdjacencyMatrixBuilder::from_weighted_edges(&nodes, &weighted_edges).unwrap();

        // Verify weighted values
        assert_eq!(adj.get(0, 1), Some(0.5)); // node 1 -> node 2
        assert_eq!(adj.get(1, 2), Some(1.5)); // node 2 -> node 3
        assert_eq!(adj.get(0, 2), Some(2.0)); // node 1 -> node 3

        // Test normalization
        let normalized = adj.to_normalized_laplacian(0.5, 1).unwrap();
        assert_eq!(normalized.shape(), (3, 3));
    }
}
