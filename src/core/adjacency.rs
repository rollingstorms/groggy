//! AdjacencyMatrix Operations - High-Performance Matrix Generation using GraphArray
//!
//! This module provides efficient adjacency matrix generation for graphs and subgraphs
//! with native Rust performance using our existing GraphArray infrastructure.
//!
//! # Features
//! - AdjacencyMatrix built on GraphArray for consistency and statistical operations
//! - Dense and sparse matrix representations
//! - Compact indexing for subgraphs  
//! - Weighted and unweighted matrices
//! - Laplacian matrix generation
//! - Leverages existing statistical operations from GraphArray
//! - Memory-efficient storage formats

use crate::core::array::GraphArray; // Our enhanced array type
use crate::core::pool::GraphPool;
use crate::core::space::GraphSpace;
use crate::errors::GraphResult;
use crate::types::{AttrValue, EdgeId, NodeId};
use std::collections::HashMap;
use std::sync::Arc;

/// AdjacencyMatrix - dense matrix representation using GraphArray for efficiency  
#[derive(Debug, Clone)]
pub struct AdjacencyMatrix {
    /// Matrix data stored as GraphArray for native statistical operations
    pub data: GraphArray,
    /// Number of rows (and columns, since square)
    pub size: usize,
    /// Storage format: row-major order
    pub row_major: bool,
    /// Optional row/column labels (node IDs)
    pub labels: Option<Vec<NodeId>>,
}

impl AdjacencyMatrix {
    /// Get matrix element at (row, col) position
    pub fn get(&self, row: usize, col: usize) -> Option<&AttrValue> {
        if row < self.size && col < self.size {
            let index = if self.row_major {
                row * self.size + col
            } else {
                col * self.size + row // Column-major
            };
            self.data.get(index)
        } else {
            None
        }
    }

    /// Get row as a slice of the underlying GraphArray
    pub fn get_row(&self, row: usize) -> Option<Vec<AttrValue>> {
        if row < self.size {
            let start = row * self.size;
            let end = start + self.size;
            Some(self.data.to_list()[start..end].to_vec())
        } else {
            None
        }
    }

    /// Get column values
    pub fn get_column(&self, col: usize) -> Option<Vec<AttrValue>> {
        if col < self.size {
            let mut column = Vec::with_capacity(self.size);
            for row in 0..self.size {
                if let Some(value) = self.get(row, col) {
                    column.push(value.clone());
                }
            }
            Some(column)
        } else {
            None
        }
    }

    /// Get diagonal values
    pub fn diagonal(&self) -> Vec<AttrValue> {
        let mut diag = Vec::with_capacity(self.size);
        for i in 0..self.size {
            if let Some(value) = self.get(i, i) {
                diag.push(value.clone());
            }
        }
        diag
    }

    /// Calculate trace (sum of diagonal elements) using GraphArray's sum method
    pub fn trace(&self) -> Option<f64> {
        let diag_array = GraphArray::from_vec(self.diagonal());
        diag_array.sum()
    }
}

/// Sparse adjacency matrix representation using COO format with GraphArray values
#[derive(Debug, Clone)]
pub struct SparseAdjacencyMatrix {
    /// Row indices
    pub rows: Vec<usize>,
    /// Column indices  
    pub cols: Vec<usize>,
    /// Values at (row, col) positions as GraphArray for statistical operations
    pub values: GraphArray,
    /// Matrix dimensions (rows, cols)
    pub shape: (usize, usize),
    /// Optional row/column labels (node IDs)
    pub labels: Option<Vec<NodeId>>,
}

impl SparseAdjacencyMatrix {
    /// Get matrix element at (row, col) position
    pub fn get(&self, row: usize, col: usize) -> Option<&AttrValue> {
        if row >= self.shape.0 || col >= self.shape.1 {
            return None;
        }

        // Find the entry in sparse format
        for i in 0..self.rows.len() {
            if self.rows[i] == row && self.cols[i] == col {
                return self.values.get(i);
            }
        }

        None // Entry not found, implicitly zero
    }

    /// Get number of non-zero entries
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Get density (fraction of non-zero entries)
    pub fn density(&self) -> f64 {
        let total_entries = self.shape.0 * self.shape.1;
        if total_entries > 0 {
            self.nnz() as f64 / total_entries as f64
        } else {
            0.0
        }
    }

    /// Get diagonal values (if they exist in sparse representation)
    pub fn diagonal(&self) -> Vec<AttrValue> {
        let mut diag = Vec::new();

        // Find diagonal entries
        for i in 0..self.rows.len() {
            if self.rows[i] == self.cols[i] {
                if let Some(value) = self.values.get(i) {
                    diag.push(value.clone());
                }
            }
        }

        diag
    }

    /// Calculate trace (sum of diagonal elements) using GraphArray's sum method
    pub fn trace(&self) -> Option<f64> {
        let diag_array = GraphArray::from_vec(self.diagonal());
        diag_array.sum()
    }

    /// Convert to dense representation if needed for certain operations
    pub fn to_dense(&self) -> AdjacencyMatrix {
        let mut data = vec![AttrValue::Float(0.0); self.shape.0 * self.shape.1];

        // Fill in non-zero values
        for i in 0..self.rows.len() {
            let row = self.rows[i];
            let col = self.cols[i];
            let index = row * self.shape.1 + col;
            if let Some(value) = self.values.get(i) {
                data[index] = value.clone();
            }
        }

        AdjacencyMatrix {
            data: GraphArray::from_vec(data),
            size: self.shape.0,
            row_major: true,
            labels: self.labels.clone(),
        }
    }
}

/// Matrix format options
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum MatrixFormat {
    Dense,
    Sparse,
}

/// Matrix type options
#[derive(Debug, Clone, PartialEq)]
pub enum MatrixType {
    /// Simple 0/1 adjacency matrix
    Unweighted,
    /// Weighted matrix using edge attributes
    Weighted { weight_attr: Option<String> },
    /// Graph Laplacian matrix
    Laplacian { normalized: bool },
}

/// Index mapping for subgraph matrices
#[derive(Debug, Clone)]
pub struct IndexMapping {
    /// Maps compact matrix index to original node ID
    pub index_to_node: Vec<NodeId>,
    /// Maps original node ID to compact matrix index
    pub node_to_index: HashMap<NodeId, usize>,
}

impl IndexMapping {
    /// Create mapping from a list of node IDs
    pub fn from_nodes(nodes: &[NodeId]) -> Self {
        let index_to_node = nodes.to_vec();
        let node_to_index = nodes
            .iter()
            .enumerate()
            .map(|(i, &node_id)| (node_id, i))
            .collect();

        Self {
            index_to_node,
            node_to_index,
        }
    }

    /// Get matrix index for a node ID
    pub fn get_index(&self, node_id: NodeId) -> Option<usize> {
        self.node_to_index.get(&node_id).copied()
    }

    /// Get node ID for a matrix index
    pub fn get_node(&self, index: usize) -> Option<NodeId> {
        self.index_to_node.get(index).copied()
    }
}

/// High-performance adjacency matrix generator
pub struct AdjacencyMatrixBuilder {
    format: MatrixFormat,
    matrix_type: MatrixType,
    use_compact_indexing: bool,
}

impl AdjacencyMatrixBuilder {
    /// Create new builder with default settings
    pub fn new() -> Self {
        Self {
            format: MatrixFormat::Sparse,
            matrix_type: MatrixType::Unweighted,
            use_compact_indexing: true,
        }
    }

    /// Set matrix format (dense or sparse)
    pub fn format(mut self, format: MatrixFormat) -> Self {
        self.format = format;
        self
    }

    /// Set matrix type (unweighted, weighted, or Laplacian)
    pub fn matrix_type(mut self, matrix_type: MatrixType) -> Self {
        self.matrix_type = matrix_type;
        self
    }

    /// Enable/disable compact indexing for subgraphs
    pub fn compact_indexing(mut self, compact: bool) -> Self {
        self.use_compact_indexing = compact;
        self
    }

    /// Build adjacency matrix for full graph
    pub fn build_full_graph(
        &self,
        pool: &GraphPool,
        space: &mut GraphSpace,
    ) -> GraphResult<AdjacencyMatrix> {
        let node_ids = space.node_ids();
        let mapping = if self.use_compact_indexing {
            Some(IndexMapping::from_nodes(&node_ids))
        } else {
            None
        };

        self.build_matrix(pool, space, &node_ids, mapping)
    }

    /// Build adjacency matrix for subgraph with specific nodes
    pub fn build_subgraph(
        &self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        subgraph_nodes: &[NodeId],
    ) -> GraphResult<AdjacencyMatrix> {
        let mapping = if self.use_compact_indexing {
            Some(IndexMapping::from_nodes(subgraph_nodes))
        } else {
            None
        };

        self.build_matrix(pool, space, subgraph_nodes, mapping)
    }

    /// Core matrix building logic
    fn build_matrix(
        &self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        nodes: &[NodeId],
        mapping: Option<IndexMapping>,
    ) -> GraphResult<AdjacencyMatrix> {
        let size = nodes.len();

        // Get graph topology (separate from matrix building to avoid borrow conflicts)
        let (edge_ids_ref, sources_ref, targets_ref, neighbors_ref) = space.snapshot(pool);
        // Clone the vectors to avoid borrow conflicts (but keep neighbors as Arc for efficiency)
        let edge_ids: Vec<EdgeId> = edge_ids_ref.as_ref().clone();
        let sources: Vec<NodeId> = sources_ref.as_ref().clone();
        let targets: Vec<NodeId> = targets_ref.as_ref().clone();
        let neighbors = neighbors_ref; // Keep as Arc - no need to clone the whole HashMap

        match self.format {
            MatrixFormat::Dense => {
                self.build_dense_matrix(pool, space, nodes, &neighbors, size, &mapping)
            }
            MatrixFormat::Sparse => {
                let sparse = self.build_sparse_matrix(
                    pool, space, nodes, &edge_ids, &sources, &targets, size, &mapping,
                )?;
                Ok(sparse.to_dense())
            }
        }
    }

    /// Build dense adjacency matrix using neighbors cache for O(V²) instead of O(E) performance
    fn build_dense_matrix(
        &self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        nodes: &[NodeId],
        neighbors: &Arc<HashMap<NodeId, Vec<(NodeId, EdgeId)>>>,
        size: usize,
        mapping: &Option<IndexMapping>,
    ) -> GraphResult<AdjacencyMatrix> {
        let mut data = vec![AttrValue::Float(0.0); size * size];
        let node_set: HashMap<NodeId, usize> = if let Some(ref map) = mapping {
            map.node_to_index.clone()
        } else {
            nodes
                .iter()
                .enumerate()
                .map(|(i, &node)| (node, i))
                .collect()
        };

        // Fill matrix using neighbors cache for much faster performance
        // PERFORMANCE: O(degree(v)) per node instead of O(E) edge iteration
        for (node_id, &node_idx) in &node_set {
            if let Some(node_neighbors) = neighbors.get(node_id) {
                for &(neighbor_id, edge_id) in node_neighbors {
                    // Only include edges where neighbor is also in our subgraph
                    if let Some(&neighbor_idx) = node_set.get(&neighbor_id) {
                        let value = self.get_edge_value(pool, space, edge_id)?;

                        // Add the edge (node → neighbor)
                        data[node_idx * size + neighbor_idx] = AttrValue::Float(value as f32);
                    }
                }
            }
        }

        // Apply matrix type transformations
        self.apply_matrix_type_dense(&mut data, size)?;

        Ok(AdjacencyMatrix {
            data: GraphArray::from_vec(data),
            size,
            row_major: true,
            labels: mapping.as_ref().map(|m| m.index_to_node.clone()),
        })
    }

    /// Build sparse adjacency matrix
    #[allow(clippy::too_many_arguments)]
    fn build_sparse_matrix(
        &self,
        pool: &GraphPool,
        space: &mut GraphSpace,
        nodes: &[NodeId],
        edge_ids: &[EdgeId],
        sources: &[NodeId],
        targets: &[NodeId],
        size: usize,
        mapping: &Option<IndexMapping>,
    ) -> GraphResult<SparseAdjacencyMatrix> {
        let mut rows = Vec::new();
        let mut cols = Vec::new();
        let mut values = Vec::new();

        let node_set: HashMap<NodeId, usize> = if let Some(ref map) = mapping {
            map.node_to_index.clone()
        } else {
            nodes
                .iter()
                .enumerate()
                .map(|(i, &node)| (node, i))
                .collect()
        };

        // Collect edges within subgraph - respect graph directionality
        let is_directed = pool.graph_type() == crate::types::GraphType::Directed;

        for i in 0..edge_ids.len() {
            let source = sources[i];
            let target = targets[i];
            let edge_id = edge_ids[i];

            if let (Some(&src_idx), Some(&tgt_idx)) = (node_set.get(&source), node_set.get(&target))
            {
                let value = self.get_edge_value(pool, space, edge_id)?;

                // Always add the primary direction (source → target)
                rows.push(src_idx);
                cols.push(tgt_idx);
                values.push(AttrValue::Float(value as f32));

                // For undirected graphs, also add the reverse direction (target → source)
                if !is_directed && src_idx != tgt_idx {
                    // Avoid duplicate self-loops
                    rows.push(tgt_idx);
                    cols.push(src_idx);
                    values.push(AttrValue::Float(value as f32));
                }
            }
        }

        // Apply matrix type transformations
        self.apply_matrix_type_sparse(&mut rows, &mut cols, &mut values, size)?;

        Ok(SparseAdjacencyMatrix {
            rows,
            cols,
            values: GraphArray::from_vec(values),
            shape: (size, size),
            labels: mapping.as_ref().map(|m| m.index_to_node.clone()),
        })
    }

    /// Get edge value based on matrix type
    fn get_edge_value(
        &self,
        pool: &GraphPool,
        space: &GraphSpace,
        edge_id: EdgeId,
    ) -> GraphResult<f64> {
        match &self.matrix_type {
            MatrixType::Unweighted => Ok(1.0),
            MatrixType::Weighted { weight_attr } => {
                if let Some(attr_name) = weight_attr {
                    // Get the current index for this attribute from space
                    if let Some(index) = space.get_attr_index(edge_id, attr_name, false) {
                        // Get the value from pool using the index
                        if let Some(attr_value) = pool.get_attr_by_index(attr_name, index, false) {
                            match attr_value {
                                AttrValue::Float(f) => Ok(*f as f64),
                                AttrValue::Int(i) => Ok(*i as f64),
                                AttrValue::SmallInt(i) => Ok(*i as f64),
                                _ => Ok(1.0), // Default for non-numeric weights
                            }
                        } else {
                            Ok(1.0) // Default if attribute missing
                        }
                    } else {
                        Ok(1.0) // Default if attribute missing
                    }
                } else {
                    Ok(1.0) // Default unweighted
                }
            }
            MatrixType::Laplacian { .. } => Ok(1.0), // Will be transformed later
        }
    }

    /// Apply matrix type transformations to dense matrix
    fn apply_matrix_type_dense(&self, data: &mut [AttrValue], size: usize) -> GraphResult<()> {
        if let MatrixType::Laplacian { normalized } = self.matrix_type {
            // Convert adjacency to Laplacian: L = D - A
            let mut degrees = vec![0.0; size];

            // Calculate degrees - extract float values
            for i in 0..size {
                for j in 0..size {
                    if let AttrValue::Float(val) = data[i * size + j] {
                        degrees[i] += val as f64;
                    }
                }
            }

            // Create Laplacian matrix
            for i in 0..size {
                for j in 0..size {
                    if let AttrValue::Float(val) = data[i * size + j] {
                        let new_val = if i == j {
                            degrees[i] - (val as f64)
                        } else {
                            -(val as f64)
                        };
                        data[i * size + j] = AttrValue::Float(new_val as f32);
                    }
                }
            }

            // Normalize if requested
            if normalized {
                for i in 0..size {
                    for j in 0..size {
                        if let AttrValue::Float(val) = data[i * size + j] {
                            if degrees[i] > 0.0 && degrees[j] > 0.0 {
                                let normalized_val =
                                    (val as f64) / (degrees[i] * degrees[j]).sqrt();
                                data[i * size + j] = AttrValue::Float(normalized_val as f32);
                            }
                        }
                    }
                }
            }
        }
        // No transformation needed for other matrix types
        Ok(())
    }

    /// Apply matrix type transformations to sparse matrix
    fn apply_matrix_type_sparse(
        &self,
        rows: &mut Vec<usize>,
        cols: &mut Vec<usize>,
        values: &mut Vec<AttrValue>,
        size: usize,
    ) -> GraphResult<()> {
        if let MatrixType::Laplacian { normalized } = self.matrix_type {
            // Calculate degrees - extract float values
            let mut degrees = vec![0.0; size];
            for i in 0..rows.len() {
                if let AttrValue::Float(val) = values[i] {
                    degrees[rows[i]] += val as f64;
                }
            }

            // Convert to Laplacian format
            for i in 0..values.len() {
                let row = rows[i];
                let col = cols[i];

                if let AttrValue::Float(val) = values[i] {
                    let new_val = if row == col {
                        degrees[row] - (val as f64)
                    } else {
                        -(val as f64)
                    };
                    values[i] = AttrValue::Float(new_val as f32);
                }
            }

            // Add diagonal entries if missing
            for i in 0..size {
                let has_diagonal = rows
                    .iter()
                    .zip(cols.iter())
                    .any(|(&r, &c)| r == i && c == i);

                if !has_diagonal && degrees[i] > 0.0 {
                    rows.push(i);
                    cols.push(i);
                    values.push(AttrValue::Float(degrees[i] as f32));
                }
            }

            // Normalize if requested
            if normalized {
                for i in 0..values.len() {
                    let row = rows[i];
                    let col = cols[i];
                    if let AttrValue::Float(val) = values[i] {
                        if degrees[row] > 0.0 && degrees[col] > 0.0 {
                            let normalized_val =
                                (val as f64) / (degrees[row] * degrees[col]).sqrt();
                            values[i] = AttrValue::Float(normalized_val as f32);
                        }
                    }
                }
            }
        }
        // No transformation needed for other matrix types
        Ok(())
    }
}

impl Default for AdjacencyMatrixBuilder {
    fn default() -> Self {
        Self::new()
    }
}

/// Adjacency matrix result using our GraphArray infrastructure
#[derive(Debug, Clone)]
pub enum AdjacencyMatrixResult {
    Dense(AdjacencyMatrix),
    Sparse(SparseAdjacencyMatrix),
}

impl AdjacencyMatrixResult {
    /// Get matrix dimensions
    pub fn shape(&self) -> (usize, usize) {
        match self {
            AdjacencyMatrixResult::Dense(m) => (m.size, m.size),
            AdjacencyMatrixResult::Sparse(m) => m.shape,
        }
    }

    /// Get node labels if available
    pub fn labels(&self) -> Option<&[NodeId]> {
        match self {
            AdjacencyMatrixResult::Dense(m) => m.labels.as_deref(),
            AdjacencyMatrixResult::Sparse(m) => m.labels.as_deref(),
        }
    }

    /// Check if matrix is sparse
    pub fn is_sparse(&self) -> bool {
        matches!(self, AdjacencyMatrixResult::Sparse(_))
    }

    /// Get memory usage in bytes (approximate)
    pub fn memory_usage(&self) -> usize {
        match self {
            AdjacencyMatrixResult::Dense(m) => {
                m.data.len() * std::mem::size_of::<AttrValue>()
                    + m.labels
                        .as_ref()
                        .map_or(0, |l| l.len() * std::mem::size_of::<NodeId>())
            }
            AdjacencyMatrixResult::Sparse(m) => {
                (m.rows.len() + m.cols.len()) * std::mem::size_of::<usize>()
                    + m.values.len() * std::mem::size_of::<AttrValue>()
                    + m.labels
                        .as_ref()
                        .map_or(0, |l| l.len() * std::mem::size_of::<NodeId>())
            }
        }
    }
}
