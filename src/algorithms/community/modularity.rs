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

/// Calculate the modularity gain/delta from moving a node to a new community.
///
/// This is the incremental version that avoids cloning the entire partition.
/// Returns the change in modularity if node is moved from old_comm to new_comm.
///
/// Formula based on Louvain paper (Blondel et al. 2008):
/// ΔQ = [Σin + k_i_in] / 2m - [(Σtot + k_i) / 2m]² - [Σin / 2m - (Σtot / 2m)² - (k_i / 2m)²]
///
/// Where:
/// - k_i = degree of node i
/// - k_i_in = sum of weights from i to nodes in the target community
/// - Σin = sum of weights inside the target community
/// - Σtot = sum of degrees in the target community
/// - m = total number of edges
pub fn modularity_delta(
    node: NodeId,
    old_comm: usize,
    new_comm: usize,
    partition: &HashMap<NodeId, usize>,
    adjacency: &HashMap<NodeId, Vec<NodeId>>,
    data: &ModularityData,
    community_degrees: &HashMap<usize, f64>,
    _community_internal: &HashMap<usize, f64>,
) -> f64 {
    if old_comm == new_comm {
        return 0.0;
    }

    let m = data.total_edges();
    if m == 0.0 {
        return 0.0;
    }

    let k_i = data.degree(&node);

    // Calculate k_i_in for new community: edges from node to nodes in new_comm
    let mut k_i_in_new = 0.0;
    if let Some(neighbors) = adjacency.get(&node) {
        for &neighbor in neighbors {
            if let Some(&neighbor_comm) = partition.get(&neighbor) {
                if neighbor_comm == new_comm {
                    k_i_in_new += 1.0;
                }
            }
        }
    }

    // Calculate k_i_in for old community: edges from node to nodes in old_comm
    let mut k_i_in_old = 0.0;
    if let Some(neighbors) = adjacency.get(&node) {
        for &neighbor in neighbors {
            if let Some(&neighbor_comm) = partition.get(&neighbor) {
                if neighbor_comm == old_comm && neighbor != node {
                    k_i_in_old += 1.0;
                }
            }
        }
    }

    let sum_tot_new = community_degrees.get(&new_comm).copied().unwrap_or(0.0);
    let sum_tot_old = community_degrees.get(&old_comm).copied().unwrap_or(0.0);

    // Modularity delta when moving from old to new
    // Remove contribution from old community
    let delta_old = -k_i_in_old / m + (sum_tot_old * k_i) / (2.0 * m * m);

    // Add contribution to new community
    let delta_new = k_i_in_new / m - ((sum_tot_new + k_i) * k_i) / (2.0 * m * m);

    delta_old + delta_new
}
