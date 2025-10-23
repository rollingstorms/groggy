use std::collections::HashMap;

use crate::types::NodeId;

/// Pre-computed degree and edge statistics for modularity evaluation.
#[derive(Debug, Clone)]
pub struct ModularityData {
    total_edges: f64,
    degrees: HashMap<NodeId, f64>,
}

impl ModularityData {
    /// Build modularity statistics from an undirected edge list. Each edge should appear once.
    pub fn new(edges: &[(NodeId, NodeId)]) -> Self {
        let mut degrees: HashMap<NodeId, f64> = HashMap::new();
        for &(u, v) in edges {
            *degrees.entry(u).or_insert(0.0) += 1.0;
            *degrees.entry(v).or_insert(0.0) += 1.0;
        }
        let total_edges = edges.len() as f64;
        Self {
            total_edges,
            degrees,
        }
    }

    pub fn total_edges(&self) -> f64 {
        self.total_edges
    }

    pub fn degree(&self, node: &NodeId) -> f64 {
        self.degrees.get(node).copied().unwrap_or(0.0)
    }
}

/// Compute Newman-Girvan modularity for an undirected graph and community partition.
///
/// * `partition` maps node -> community identifier.
/// * `edges` should contain each undirected edge once (u <= v).
pub fn modularity(
    partition: &HashMap<NodeId, usize>,
    edges: &[(NodeId, NodeId)],
    data: &ModularityData,
) -> f64 {
    let m = data.total_edges();
    if m == 0.0 {
        return 0.0;
    }

    let mut sum_in: HashMap<usize, f64> = HashMap::new();
    for &(u, v) in edges {
        if let (Some(&cu), Some(&cv)) = (partition.get(&u), partition.get(&v)) {
            if cu == cv {
                *sum_in.entry(cu).or_insert(0.0) += 1.0;
            }
        }
    }

    let mut sum_tot: HashMap<usize, f64> = HashMap::new();
    for (&node, &community) in partition {
        let deg = data.degree(&node);
        *sum_tot.entry(community).or_insert(0.0) += deg;
    }

    let mut q = 0.0;
    for (&community, &tot) in &sum_tot {
        let in_weight = sum_in.get(&community).copied().unwrap_or(0.0);
        let term_in = in_weight / m;
        let term_tot = (tot / (2.0 * m)).powi(2);
        q += term_in - term_tot;
    }

    q
}
