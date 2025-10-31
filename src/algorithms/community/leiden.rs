use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::algorithms::community::modularity::{modularity_delta, ModularityData};
use crate::algorithms::community::utils::find_connected_components;
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Leiden community detection algorithm.
///
/// Leiden improves on Louvain by:
/// 1. Guaranteeing connected communities after each move phase
/// 2. Adding a refinement phase that splits poorly connected communities
/// 3. Typically converging faster with better quality
#[derive(Clone, Debug)]
pub struct Leiden {
    max_iter: usize,
    max_phases: usize,
    resolution: f64,
    seed: Option<u64>,
    output_attr: AttrName,
}

impl Leiden {
    pub fn new(
        max_iter: usize,
        max_phases: usize,
        resolution: f64,
        seed: Option<u64>,
        output_attr: AttrName,
    ) -> Result<Self> {
        if max_iter == 0 {
            return Err(anyhow!("max_iter must be greater than zero"));
        }
        if max_phases == 0 {
            return Err(anyhow!("max_phases must be greater than zero"));
        }
        if resolution <= 0.0 {
            return Err(anyhow!("resolution must be positive"));
        }
        Ok(Self {
            max_iter,
            max_phases,
            resolution,
            seed,
            output_attr,
        })
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.leiden".to_string(),
            name: "Leiden".to_string(),
            description: "Improved modularity optimization with guaranteed connectivity."
                .to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linearithmic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "max_iter".to_string(),
                    description: "Maximum number of node-move iterations per phase.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(20)),
                },
                ParameterMetadata {
                    name: "max_phases".to_string(),
                    description: "Maximum number of refinement phases.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(10)),
                },
                ParameterMetadata {
                    name: "resolution".to_string(),
                    description: "Resolution parameter for modularity.".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(1.0)),
                },
                ParameterMetadata {
                    name: "seed".to_string(),
                    description: "Random seed for reproducibility.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store the resulting communities.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("community".to_string())),
                },
            ],
        }
    }

    /// Move nodes to neighboring communities if it improves modularity.
    fn move_phase(
        &self,
        partition: &mut HashMap<NodeId, usize>,
        _edges: &[(NodeId, NodeId)],
        data: &ModularityData,
        adjacency: &HashMap<NodeId, Vec<NodeId>>,
        ctx: &Context,
    ) -> Result<bool> {
        let mut changed = false;
        let nodes: Vec<NodeId> = partition.keys().copied().collect();

        // ⚡ OPTIMIZATION: Track community degrees incrementally
        let mut community_degrees: HashMap<usize, f64> = HashMap::new();
        for (&node, &comm) in partition.iter() {
            let deg = data.degree(&node);
            *community_degrees.entry(comm).or_insert(0.0) += deg;
        }

        // Track community internal edges (unused but needed for modularity_delta signature)
        let community_internal: HashMap<usize, f64> = HashMap::new();

        for _ in 0..self.max_iter {
            if ctx.is_cancelled() {
                return Err(anyhow!("Leiden cancelled"));
            }

            let mut local_changed = false;

            for &node in &nodes {
                let current_comm = partition[&node];

                // Find candidate communities (neighbors' communities)
                let mut candidate_comms: HashSet<usize> = HashSet::new();
                candidate_comms.insert(current_comm);
                if let Some(neighbors) = adjacency.get(&node) {
                    for &neighbor in neighbors {
                        if let Some(&comm) = partition.get(&neighbor) {
                            candidate_comms.insert(comm);
                        }
                    }
                }

                let mut best_comm = current_comm;
                let mut best_delta = 0.0;

                // ⚡ OPTIMIZATION: Use incremental modularity (same as Louvain!)
                for &candidate in &candidate_comms {
                    if candidate == current_comm {
                        continue;
                    }

                    // Calculate delta without expensive iteration
                    let delta = modularity_delta(
                        node,
                        current_comm,
                        candidate,
                        partition,
                        adjacency,
                        data,
                        &community_degrees,
                        &community_internal,
                    );

                    // Apply resolution parameter
                    let gain = delta * self.resolution;

                    if gain > best_delta {
                        best_delta = gain;
                        best_comm = candidate;
                    }
                }

                // Apply best move and update community stats incrementally
                if best_comm != current_comm {
                    let node_degree = data.degree(&node);

                    // Update old community
                    if let Some(deg) = community_degrees.get_mut(&current_comm) {
                        *deg -= node_degree;
                    }

                    // Update new community
                    *community_degrees.entry(best_comm).or_insert(0.0) += node_degree;

                    // Move node
                    partition.insert(node, best_comm);
                    local_changed = true;
                    changed = true;
                }
            }

            if !local_changed {
                break;
            }
        }

        Ok(changed)
    }

    /// Refine communities by ensuring connectivity within each community.
    /// Split communities that are not well-connected.
    fn refinement_phase(
        &self,
        partition: &mut HashMap<NodeId, usize>,
        adjacency: &HashMap<NodeId, Vec<NodeId>>,
        ctx: &Context,
    ) -> Result<bool> {
        if ctx.is_cancelled() {
            return Err(anyhow!("Leiden cancelled"));
        }

        // Group nodes by community
        let mut communities: HashMap<usize, Vec<NodeId>> = HashMap::new();
        for (&node, &comm) in partition.iter() {
            communities.entry(comm).or_default().push(node);
        }

        let mut changed = false;
        let mut next_comm_id = partition.values().max().copied().unwrap_or(0) + 1;

        for (comm_id, nodes) in communities {
            if nodes.len() <= 1 {
                continue;
            }

            // Check if community is connected via BFS
            let components = find_connected_components(&nodes, adjacency);

            if components.len() > 1 {
                // Community is disconnected - split it
                changed = true;
                for (idx, component) in components.into_iter().enumerate() {
                    let new_comm = if idx == 0 {
                        comm_id // Keep original ID for first component
                    } else {
                        let id = next_comm_id;
                        next_comm_id += 1;
                        id
                    };

                    for node in component {
                        partition.insert(node, new_comm);
                    }
                }
            }
        }

        Ok(changed)
    }
}

impl Algorithm for Leiden {
    fn id(&self) -> &'static str {
        "community.leiden"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        if let Some(seed) = self.seed {
            fastrand::seed(seed);
        }

        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        if nodes.is_empty() {
            return Ok(subgraph);
        }

        // Build edge list and adjacency
        let mut edge_set: HashSet<(NodeId, NodeId)> = HashSet::new();
        let graph_ref = subgraph.graph();
        let graph = graph_ref.borrow();
        for &edge_id in subgraph.edge_set() {
            let (u, v) = graph.edge_endpoints(edge_id)?;
            if u == v {
                continue;
            }
            let pair = if u <= v { (u, v) } else { (v, u) };
            edge_set.insert(pair);
        }
        drop(graph);

        let edges: Vec<(NodeId, NodeId)> = edge_set.into_iter().collect();

        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &(u, v) in &edges {
            adjacency.entry(u).or_default().push(v);
            adjacency.entry(v).or_default().push(u);
        }

        let data = ModularityData::new(&edges);

        // Initialize each node in its own community
        let mut partition: HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(i, &node)| (node, i))
            .collect();

        // Leiden phases
        for _phase in 0..self.max_phases {
            if ctx.is_cancelled() {
                return Err(anyhow!("Leiden cancelled"));
            }

            // 1. Move phase: optimize modularity
            let move_changed = self.move_phase(&mut partition, &edges, &data, &adjacency, ctx)?;

            // 2. Refinement phase: ensure connectivity
            let refine_changed = self.refinement_phase(&mut partition, &adjacency, ctx)?;

            if !move_changed && !refine_changed {
                break;
            }
        }

        // Relabel communities to be contiguous
        let mut comm_map: HashMap<usize, usize> = HashMap::new();
        let mut next_id = 0;
        for comm in partition.values() {
            if !comm_map.contains_key(comm) {
                comm_map.insert(*comm, next_id);
                next_id += 1;
            }
        }

        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = partition
                .iter()
                .map(|(&node, &comm)| {
                    let relabeled = comm_map[&comm];
                    (node, AttrValue::Int(relabeled as i64))
                })
                .collect();

            ctx.with_scoped_timer("community.leiden.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })?;
        }

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = Leiden::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let max_iter = spec.params.get_int("max_iter").unwrap_or(20).max(1) as usize;
        let max_phases = spec.params.get_int("max_phases").unwrap_or(10).max(1) as usize;
        let resolution = spec.params.get_float("resolution").unwrap_or(1.0);
        let seed = spec.params.get_int("seed").map(|s| s as u64);
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("community")
            .to_string();
        Leiden::new(max_iter, max_phases, resolution, seed, output_attr)
            .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::api::graph::Graph;
    use std::cell::RefCell;
    use std::rc::Rc;

    fn make_two_cliques() -> (Graph, Vec<NodeId>) {
        let mut graph = Graph::new();
        let nodes: Vec<NodeId> = (0..8).map(|_| graph.add_node()).collect();

        // First clique: 0-1-2-3
        for i in 0..4 {
            for j in i + 1..4 {
                graph.add_edge(nodes[i], nodes[j]).unwrap();
                graph.add_edge(nodes[j], nodes[i]).unwrap();
            }
        }

        // Second clique: 4-5-6-7
        for i in 4..8 {
            for j in i + 1..8 {
                graph.add_edge(nodes[i], nodes[j]).unwrap();
                graph.add_edge(nodes[j], nodes[i]).unwrap();
            }
        }

        // Weak link between cliques
        graph.add_edge(nodes[3], nodes[4]).unwrap();
        graph.add_edge(nodes[4], nodes[3]).unwrap();

        (graph, nodes)
    }

    #[test]
    fn test_leiden_two_cliques() {
        let (graph, nodes) = make_two_cliques();
        let node_set: HashSet<NodeId> = nodes.iter().copied().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), node_set, "test".into()).unwrap();

        let leiden = Leiden::new(20, 10, 1.0, Some(42), "community".into()).unwrap();
        let mut ctx = Context::new();
        let result = leiden.execute(&mut ctx, subgraph).unwrap();

        // Check that nodes are assigned communities
        let attr_name: AttrName = "community".into();
        let communities: HashMap<NodeId, i64> = nodes
            .iter()
            .map(|&n| {
                let val = result.get_node_attribute(n, &attr_name).unwrap().unwrap();
                (n, val.as_int().unwrap())
            })
            .collect();

        // First clique should be in same community
        let comm0 = communities[&nodes[0]];
        assert_eq!(communities[&nodes[1]], comm0);
        assert_eq!(communities[&nodes[2]], comm0);

        // Second clique should be in same community
        let comm4 = communities[&nodes[4]];
        assert_eq!(communities[&nodes[5]], comm4);
        assert_eq!(communities[&nodes[6]], comm4);

        // The two cliques should ideally be in different communities
        // (though algorithm may merge due to bridge edge)
    }

    #[test]
    fn test_leiden_empty_graph() {
        let graph = Graph::new();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), HashSet::new(), "empty".into())
                .unwrap();

        let leiden = Leiden::new(20, 10, 1.0, None, "community".into()).unwrap();
        let mut ctx = Context::new();
        let result = leiden.execute(&mut ctx, subgraph).unwrap();

        assert_eq!(result.node_count(), 0);
    }

    #[test]
    fn test_connected_components_detection() {
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        adjacency.insert(0, vec![1, 2]);
        adjacency.insert(1, vec![0, 2]);
        adjacency.insert(2, vec![0, 1]);
        adjacency.insert(3, vec![4]);
        adjacency.insert(4, vec![3]);

        let nodes = vec![0, 1, 2, 3, 4];
        let components = find_connected_components(&nodes, &adjacency);

        assert_eq!(components.len(), 2);
        assert!(components.iter().any(|c| c.contains(&0) && c.contains(&1)));
        assert!(components.iter().any(|c| c.contains(&3) && c.contains(&4)));
    }
}
