//! Checked graph operators: one contribution per edge instance, canonical CSR.
use super::*;
use crate::storage::matrix::checked_kernels::Product;
use std::collections::HashMap;

impl<T: NumericType> GraphMatrix<T> {
    /// Sum edge instances into an undirected adjacency operator.
    pub fn adjacency_from_edges(nodes: &[NodeId], edges: &[(NodeId, NodeId)]) -> GraphResult<Self> {
        Self::adjacency_from_edges_with_direction(nodes, edges, false)
    }
    /// Sum edge instances, preserving orientation for directed graphs.
    pub fn adjacency_from_edges_with_direction(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId)],
        directed: bool,
    ) -> GraphResult<Self> {
        Self::adjacency_from_edge_iter(
            nodes,
            edges.iter().map(|&(s, t)| Ok((s, t, T::one()))),
            directed,
            false,
        )
    }
    /// Sum weighted edge instances into an undirected adjacency operator.
    pub fn weighted_adjacency_from_edges(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId, T)],
    ) -> GraphResult<Self> {
        Self::weighted_adjacency_from_edges_with_direction(nodes, edges, false)
    }
    /// Sum weights at each coordinate. Undirected self-loops contribute once.
    pub fn weighted_adjacency_from_edges_with_direction(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId, T)],
        directed: bool,
    ) -> GraphResult<Self> {
        Self::adjacency_from_edge_iter(nodes, edges.iter().copied().map(Ok), directed, false)
    }
    /// Build a canonical CSR operator without allocating a dense intermediate.
    pub fn sparse_weighted_adjacency_from_edges(
        nodes: &[NodeId],
        edges: &[(NodeId, NodeId, T)],
        directed: bool,
    ) -> GraphResult<Self> {
        Self::adjacency_from_edge_iter(nodes, edges.iter().copied().map(Ok), directed, true)
    }
    pub(crate) fn adjacency_from_edge_iter(
        nodes: &[NodeId],
        edges: impl IntoIterator<Item = GraphResult<(NodeId, NodeId, T)>>,
        directed: bool,
        sparse: bool,
    ) -> GraphResult<Self> {
        let error = |message: String| GraphError::InvalidInput(message);
        let mut positions = HashMap::new();
        positions
            .try_reserve(nodes.len())
            .map_err(|e| error(e.to_string()))?;
        for (i, &node) in nodes.iter().enumerate() {
            if positions.insert(node, i).is_some() {
                return Err(error(format!("Duplicate adjacency node ID {node}")));
            }
        }
        let mut entries = Vec::new();
        for edge in edges {
            let (source, target, weight) = edge?;
            let row = *positions
                .get(&source)
                .ok_or_else(|| error(format!("Unknown adjacency endpoint {source}")))?;
            let col = *positions
                .get(&target)
                .ok_or_else(|| error(format!("Unknown adjacency endpoint {target}")))?;
            entries
                .try_reserve(if directed || row == col { 1 } else { 2 })
                .map_err(|e| error(e.to_string()))?;
            entries.push((row, col, weight));
            if !directed && row != col {
                entries.push((col, row, weight));
            }
        }
        entries.sort_unstable_by_key(|&(r, c, _)| (r, c));
        let mut values = Vec::new();
        let mut indices = Vec::new();
        values
            .try_reserve(entries.len())
            .map_err(|e| error(e.to_string()))?;
        indices
            .try_reserve(entries.len())
            .map_err(|e| error(e.to_string()))?;
        let pointer_len = nodes
            .len()
            .checked_add(1)
            .ok_or_else(|| error("Adjacency shape overflow".into()))?;
        let mut pointers = Vec::new();
        pointers
            .try_reserve_exact(pointer_len)
            .map_err(|e| error(e.to_string()))?;
        pointers.resize(pointer_len, 0);
        let mut start = 0;
        while start < entries.len() {
            let (row, col, _) = entries[start];
            let mut sum = Product::<T>::new();
            let mut end = start;
            while end < entries.len() && (entries[end].0, entries[end].1) == (row, col) {
                sum.add(entries[end].2, T::one())
                    .map_err(|e| error(e.to_string()))?;
                end += 1;
            }
            let value = sum.total().map_err(|e| error(e.to_string()))?;
            if value != T::zero() {
                values.push(value);
                indices.push(col);
                pointers[row + 1] += 1;
            }
            start = end;
        }
        for row in 0..nodes.len() {
            pointers[row + 1] += pointers[row];
        }
        let mut result = Self::from_csr(nodes.len(), nodes.len(), values, indices, pointers)?;
        if !sparse {
            result = result.to_dense()?;
        }
        result.column_names = nodes.iter().map(|id| format!("node_{id}")).collect();
        result.row_labels = Some(result.column_names.clone());
        Ok(result)
    }
}
