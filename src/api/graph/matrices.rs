//! Graph-scoped numerical operators. GraphSpace owns membership and value indices.
use super::*;
use crate::storage::{
    adjacency::{MatrixFormat, MatrixType},
    matrix::NumericType,
    matrix::{FromAttrValue, GraphMatrix},
};
impl Graph {
    /// Export numeric scalar feature columns from this graph's value references.
    /// Missing, null and nonnumeric cells retain legacy zero imputation.
    pub fn to_matrix<T: NumericType + FromAttrValue<T>>(&self) -> GraphResult<GraphMatrix<T>> {
        let mut nodes = self.space.node_ids();
        nodes.sort_unstable();
        let pool = self.pool.borrow();
        let (names, _) = pool.attribute_names();
        crate::storage::matrix::graph_projection::features(
            &nodes,
            names.into_iter().collect(),
            |node, name| {
                self.space
                    .get_node_attr_index(node, name)
                    .map(|index| {
                        pool.get_attr_by_index(name, index, true).ok_or_else(|| {
                            GraphError::InvalidInput(format!(
                                "Invalid feature reference for node {node}, attribute {name}"
                            ))
                        })
                    })
                    .transpose()
            },
        )
    }

    /// Build an owned operator from exactly these entity IDs, retaining node order.
    /// Missing weights are one; explicit values use the attribute conversion policy.
    pub fn adjacency_from_entities<T: NumericType + FromAttrValue<T>>(
        &self,
        nodes: &[NodeId],
        edges: &[EdgeId],
        sparse: bool,
        weight_attr: Option<&str>,
    ) -> GraphResult<GraphMatrix<T>> {
        for &node in nodes {
            if !self.space.contains_node(node) {
                return Err(GraphError::InvalidInput(format!(
                    "Inactive adjacency node {node}"
                )));
            }
        }
        let pool = self.pool.borrow();
        let attr = weight_attr.map(str::to_owned);
        crate::storage::matrix::graph_projection::adjacency(
            nodes,
            edges,
            sparse,
            attr.as_ref(),
            self.is_directed(),
            |edge| self.edge_endpoints(edge),
            |edge, name| {
                self.space
                    .get_edge_attr_index(edge, name)
                    .map(|index| {
                        pool.get_attr_by_index(name, index, false).ok_or_else(|| {
                            GraphError::InvalidInput(format!(
                                "Invalid weight reference for edge {edge}"
                            ))
                        })
                    })
                    .transpose()
            },
        )
    }
    fn adjacency_for_nodes<T: NumericType>(
        &self,
        nodes: Option<&[NodeId]>,
        sparse: bool,
    ) -> GraphResult<GraphMatrix<T>> {
        let owned;
        let nodes = match nodes {
            Some(nodes) => nodes,
            None => {
                owned = {
                    let mut ids = self.space.node_ids();
                    ids.sort_unstable();
                    ids
                };
                &owned
            }
        };
        for &node in nodes {
            if !self.space.contains_node(node) {
                return Err(GraphError::InvalidInput(format!(
                    "Inactive adjacency node {node}"
                )));
            }
        }
        let selected: HashSet<_> = nodes.iter().copied().collect();
        let mut edges: Vec<_> = self.space.get_active_edges().iter().copied().collect();
        edges.sort_unstable();
        let mut endpoints = Vec::new();
        for edge in edges {
            let (source, target) = self.edge_endpoints(edge)?;
            if selected.contains(&source) && selected.contains(&target) {
                endpoints.push(Ok((source, target, T::one())));
            } else if nodes.len() == self.space.get_active_nodes().len() {
                return Err(GraphError::InvalidInput(format!(
                    "Edge {edge} has an inactive endpoint"
                )));
            }
        }
        GraphMatrix::adjacency_from_edge_iter(nodes, endpoints, self.is_directed(), sparse)
    }
    /// Dense adjacency, with ascending node ID coordinates and summed parallel edges.
    pub fn to_adjacency_matrix<T: NumericType>(&self) -> GraphResult<GraphMatrix<T>> {
        self.adjacency_for_nodes(None, false)
    }
    /// Weighted dense adjacency using graph-scoped value references.
    pub fn to_weighted_adjacency_matrix<T: NumericType + FromAttrValue<T>>(
        &self,
        weight_attr: &str,
    ) -> GraphResult<GraphMatrix<T>> {
        let mut nodes = self.space.node_ids();
        nodes.sort_unstable();
        let mut edges: Vec<_> = self.space.get_active_edges().iter().copied().collect();
        edges.sort_unstable();
        self.adjacency_from_entities(&nodes, &edges, false, Some(weight_attr))
    }
    /// Weighted dense adjacency, propagating invalid weight errors.
    pub fn weighted_adjacency_matrix(&mut self, weight_attr: &str) -> GraphResult<AdjacencyMatrix> {
        self.to_weighted_adjacency_matrix(weight_attr)
    }
    /// Dense adjacency in canonical node order.
    pub fn dense_adjacency_matrix(&mut self) -> GraphResult<AdjacencyMatrix> {
        self.to_adjacency_matrix()
    }
    /// Canonical CSR adjacency without a dense intermediate.
    pub fn sparse_adjacency_matrix(&mut self) -> GraphResult<AdjacencyMatrix> {
        self.adjacency_for_nodes(None, true)
    }
    /// Generate Laplacian matrix
    pub fn laplacian_matrix(&mut self, normalized: bool) -> GraphResult<AdjacencyMatrix> {
        let adj = self.adjacency_matrix()?;
        if normalized {
            adj.normalized_laplacian()
        } else {
            adj.to_laplacian()
        }
    }

    /// Generate a row-stochastic transition matrix; dangling rows remain zero.
    pub fn transition_matrix(&mut self) -> GraphResult<AdjacencyMatrix> {
        self.adjacency_matrix()?.transition_matrix()
    }

    /// Induced adjacency on explicit node IDs, retaining caller order.
    pub fn subgraph_adjacency_matrix(&mut self, nodes: &[NodeId]) -> GraphResult<AdjacencyMatrix> {
        self.adjacency_for_nodes(Some(nodes), false)
    }
    /// Select storage and operator; compact coordinates retain original IDs as labels.
    pub fn custom_adjacency_matrix(
        &mut self,
        format: MatrixFormat,
        matrix_type: MatrixType,
        compact_indexing: bool,
        nodes: Option<&[NodeId]>,
    ) -> GraphResult<AdjacencyMatrix> {
        if !compact_indexing {
            return Err(GraphError::InvalidInput(
                "Raw-ID matrix coordinates are unsupported; use compact indexing and node labels"
                    .into(),
            ));
        }
        let base = self.adjacency_for_nodes(nodes, format == MatrixFormat::Sparse)?;
        match matrix_type {
            MatrixType::Adjacency => Ok(base),
            MatrixType::Laplacian { normalized: true } => base.normalized_laplacian(),
            MatrixType::Laplacian { normalized: false } => base.to_laplacian(),
            MatrixType::Transition => base.transition_matrix(),
        }
    }
}
