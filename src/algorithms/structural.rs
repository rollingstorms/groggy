//! Unweighted simple-projection summaries and the shared checked triangle kernel.
use crate::{state::topology::Csr, GraphError, GraphResult, NodeId};
use std::{collections::HashSet, sync::Arc};

/// Immutable topology-only input for undirected structural summaries.
/// Shared CSR/IDs pin no property payload, history, writable owner or graph view.
#[derive(Debug, Clone)]
pub struct UndirectedStructure {
    nodes: Arc<[NodeId]>,
    csr: Arc<Csr>,
}
impl UndirectedStructure {
    pub(crate) fn new(nodes: Arc<[NodeId]>, csr: Arc<Csr>) -> Self {
        Self { nodes, csr }
    }
    /// Per-node clustering, ordered by node ID, on the simple loop-free projection.
    pub fn clustering_coefficients(&self) -> GraphResult<Vec<(NodeId, f64)>> {
        let (degrees, counts) = statistics(&self.csr)?;
        Ok(self
            .nodes
            .iter()
            .copied()
            .zip(
                degrees
                    .into_iter()
                    .zip(counts)
                    .map(|(degree, count)| coefficient(degree, count)),
            )
            .collect())
    }
    /// Local coefficient or average including zero-degree/one-degree nodes.
    pub fn clustering_coefficient(&self, node: Option<NodeId>) -> GraphResult<f64> {
        if let Some(node) = node {
            let row = self.nodes.binary_search(&node).map_err(|_| {
                GraphError::InvalidInput(format!("Node {node} is outside the structural selection"))
            })?;
            return local_coefficient(&self.csr, row);
        }
        let values = self.clustering_coefficients()?;
        Ok(if values.is_empty() {
            0.
        } else {
            values.iter().map(|(_, value)| value).sum::<f64>() / values.len() as f64
        })
    }
    /// Closed connected triples / connected triples on the simple projection.
    pub fn transitivity(&self) -> GraphResult<f64> {
        let (degrees, counts) = statistics(&self.csr)?;
        let triples: f64 = degrees.into_iter().map(pairs).sum();
        Ok(if triples == 0. {
            0.
        } else {
            counts.into_iter().map(|v| v as f64).sum::<f64>() / triples
        })
    }
}
fn pairs(degree: usize) -> f64 {
    if degree < 2 {
        0.
    } else {
        degree as f64 * (degree - 1) as f64 / 2.
    }
}
fn coefficient(degree: usize, count: i64) -> f64 {
    let denominator = pairs(degree);
    if denominator == 0. {
        0.
    } else {
        count as f64 / denominator
    }
}
fn statistics(csr: &Csr) -> GraphResult<(Vec<usize>, Vec<i64>)> {
    let mut rows = Vec::with_capacity(csr.node_count());
    let mut degrees = Vec::with_capacity(csr.node_count());
    for node in 0..csr.node_count() {
        let mut row: Vec<_> = csr
            .neighbors(node)
            .iter()
            .copied()
            .filter(|&other| other != node)
            .collect();
        row.sort_unstable();
        row.dedup();
        degrees.push(row.len());
        rows.push(row);
    }
    Ok((degrees, triangle_counts(rows, || false)?))
}
fn local_coefficient(csr: &Csr, node: usize) -> GraphResult<f64> {
    let neighbors: HashSet<_> = csr
        .neighbors(node)
        .iter()
        .copied()
        .filter(|&other| other != node)
        .collect();
    if neighbors.len() < 2 {
        return Ok(0.);
    }
    let mut triangles = 0usize;
    for &source in &neighbors {
        if source >= csr.node_count() {
            return Err(GraphError::InvalidInput(
                "Structural neighbor is outside CSR".into(),
            ));
        }
        let mut targets: Vec<_> = csr
            .neighbors(source)
            .iter()
            .copied()
            .filter(|target| source < *target && neighbors.contains(target))
            .collect();
        targets.sort_unstable();
        targets.dedup();
        triangles = triangles
            .checked_add(targets.len())
            .ok_or_else(|| GraphError::InvalidInput("Local triangle count exceeds usize".into()))?;
    }
    Ok(triangles as f64 / pairs(neighbors.len()))
}
/// Orient the simple projection by row ID; count every triangle once with checked i64 counts.
/// Consumes existing rows so builder callers do not need another full adjacency clone.
pub(crate) fn triangle_counts(
    mut rows: Vec<Vec<usize>>,
    cancelled: impl Fn() -> bool,
) -> GraphResult<Vec<i64>> {
    if cancelled() {
        return Err(GraphError::InvalidInput("triangle_count cancelled".into()));
    }
    let count = rows.len();
    for (node, row) in rows.iter_mut().enumerate() {
        if cancelled() {
            return Err(GraphError::InvalidInput("triangle_count cancelled".into()));
        }
        if row.iter().any(|&other| other >= count) {
            return Err(GraphError::InvalidInput(
                "Triangle neighbor is outside selected rows".into(),
            ));
        }
        row.sort_unstable();
        row.dedup();
        row.retain(|&other| other > node);
    }
    let mut counts = vec![0i64; count];
    for (u, neighbors) in rows.iter().enumerate() {
        if cancelled() {
            return Err(GraphError::InvalidInput("triangle_count cancelled".into()));
        }
        for (position, &v) in neighbors.iter().enumerate() {
            let mut left = position + 1;
            let mut right = 0;
            let other = &rows[v];
            while left < neighbors.len() && right < other.len() {
                if cancelled() {
                    return Err(GraphError::InvalidInput("triangle_count cancelled".into()));
                }
                match neighbors[left].cmp(&other[right]) {
                    std::cmp::Ordering::Less => left += 1,
                    std::cmp::Ordering::Greater => right += 1,
                    std::cmp::Ordering::Equal => {
                        for node in [u, v, neighbors[left]] {
                            counts[node] = counts[node].checked_add(1).ok_or_else(|| {
                                GraphError::InvalidInput("Triangle count exceeds i64".into())
                            })?;
                        }
                        left += 1;
                        right += 1;
                    }
                }
            }
        }
    }
    if cancelled() {
        return Err(GraphError::InvalidInput("triangle_count cancelled".into()));
    }
    Ok(counts)
}
/// Density of selected edge instances; directed/undirected denominators exclude loops.
/// Multiedges and loops still contribute to the numerator, so density may exceed one.
pub(crate) fn density(nodes: usize, edges: usize, directed: bool) -> f64 {
    if nodes < 2 {
        return 0.;
    }
    let denominator = nodes as f64 * (nodes - 1) as f64;
    edges as f64 / denominator * if directed { 1. } else { 2. }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn triangle_kernel_checks_cancellation_empty_input_and_invalid_neighbor_rows() {
        assert!(triangle_counts(Vec::new(), || true).is_err());
        assert!(triangle_counts(vec![vec![1]], || false).is_err());
        assert_eq!(
            triangle_counts(vec![vec![0, 1, 1, 2], vec![0, 2], vec![0, 1]], || false).unwrap(),
            vec![1, 1, 1]
        );
    }
    #[test]
    fn density_never_multiplies_usize_counts_and_keeps_multigraph_semantics() {
        assert!(density(usize::MAX, usize::MAX, false).is_finite());
        assert_eq!(density(2, 3, false), 3.);
        assert_eq!(density(2, 3, true), 1.5);
        assert_eq!(density(1, 1, false), 0.);
    }
}
