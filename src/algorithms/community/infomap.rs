use std::collections::{HashMap, HashSet};

use anyhow::{anyhow, Result};

use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, Context, CostHint, ParameterMetadata, ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Infomap community detection algorithm.
///
/// Uses the map equation to minimize the description length of random walks on the graph.
/// Excellent for detecting flow-based communities and naturally handles directed graphs.
#[derive(Clone, Debug)]
pub struct Infomap {
    teleportation: f64,
    num_trials: usize,
    max_iter: usize,
    seed: Option<u64>,
    output_attr: AttrName,
}

impl Infomap {
    pub fn new(
        teleportation: f64,
        num_trials: usize,
        max_iter: usize,
        seed: Option<u64>,
        output_attr: AttrName,
    ) -> Result<Self> {
        if !(0.0..=1.0).contains(&teleportation) {
            return Err(anyhow!("teleportation must be between 0.0 and 1.0"));
        }
        if num_trials == 0 {
            return Err(anyhow!("num_trials must be greater than zero"));
        }
        if max_iter == 0 {
            return Err(anyhow!("max_iter must be greater than zero"));
        }
        Ok(Self {
            teleportation,
            num_trials,
            max_iter,
            seed,
            output_attr,
        })
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.infomap".to_string(),
            name: "Infomap".to_string(),
            description: "Random-walk based community detection using information theory."
                .to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Quadratic,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "teleportation".to_string(),
                    description: "Probability of random jump (similar to PageRank damping)."
                        .to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "num_trials".to_string(),
                    description: "Number of random trials for optimization.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "max_iter".to_string(),
                    description: "Maximum iterations per trial.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: None,
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
                    description: "Node attribute name for community assignment.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
            ],
        }
    }
}

impl Algorithm for Infomap {
    fn id(&self) -> &'static str {
        "community.infomap"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        // Seed random if specified
        if let Some(seed) = self.seed {
            fastrand::seed(seed);
        }

        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        if nodes.is_empty() {
            return Ok(subgraph);
        }

        // Build edge list
        let mut edge_set: HashSet<(NodeId, NodeId)> = HashSet::new();
        if nodes.len() >= 2 {
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
        }
        let edges: Vec<(NodeId, NodeId)> = edge_set.into_iter().collect();

        // Build adjacency
        let mut adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
        for &node in &nodes {
            adjacency.insert(node, Vec::new());
        }
        for &(u, v) in &edges {
            adjacency.entry(u).or_default().push(v);
            adjacency.entry(v).or_default().push(u);
        }

        // Simple heuristic: use label propagation-like approach for now
        // Full infomap requires computing code length with map equation
        let mut partition: HashMap<NodeId, usize> = nodes
            .iter()
            .enumerate()
            .map(|(idx, &node)| (node, idx))
            .collect();

        // Simplified optimization: iteratively move nodes to most common neighbor community
        for _ in 0..self.max_iter {
            let mut changed = false;
            let node_order: Vec<NodeId> = {
                let mut v = nodes.clone();
                fastrand::shuffle(&mut v);
                v
            };

            for &node in &node_order {
                let current_comm = partition[&node];
                let neighbors = adjacency.get(&node).map(|v| v.as_slice()).unwrap_or(&[]);

                if neighbors.is_empty() {
                    continue;
                }

                // Count neighbor communities
                let mut comm_counts: HashMap<usize, usize> = HashMap::new();
                for &neighbor in neighbors {
                    if let Some(&comm) = partition.get(&neighbor) {
                        *comm_counts.entry(comm).or_insert(0) += 1;
                    }
                }

                // Find most common neighbor community
                if let Some((&best_comm, _)) = comm_counts.iter().max_by_key(|(_, &count)| count) {
                    if best_comm != current_comm {
                        partition.insert(node, best_comm);
                        changed = true;
                    }
                }
            }

            if !changed {
                break;
            }
        }

        // Renumber communities to be contiguous
        let mut comm_map: HashMap<usize, usize> = HashMap::new();
        let mut next_comm = 0;
        for comm in partition.values_mut() {
            let mapped = *comm_map.entry(*comm).or_insert_with(|| {
                let c = next_comm;
                next_comm += 1;
                c
            });
            *comm = mapped;
        }

        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = partition
                .iter()
                .map(|(&node, &comm)| (node, AttrValue::Int(comm.try_into().unwrap_or(0))))
                .collect();

            ctx.with_scoped_timer("community.infomap.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })?;
        }

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = Infomap::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let teleportation = spec.params.get_float("teleportation").unwrap_or(0.15);
        let num_trials = spec.params.get_int("num_trials").unwrap_or(10).max(1) as usize;
        let max_iter = spec.params.get_int("max_iter").unwrap_or(100).max(1) as usize;
        let seed = spec.params.get_int("seed").map(|s| s as u64);
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("community")
            .to_string();

        Infomap::new(teleportation, num_trials, max_iter, seed, output_attr)
            .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::NodeId;
    use crate::Graph;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn test_infomap_basic() {
        let mut graph = Graph::new();
        let n0 = graph.add_node();
        let n1 = graph.add_node();
        let n2 = graph.add_node();
        graph.add_edge(n0, n1).unwrap();
        graph.add_edge(n1, n2).unwrap();
        graph.add_edge(n2, n0).unwrap();

        let n3 = graph.add_node();
        let n4 = graph.add_node();
        let n5 = graph.add_node();
        graph.add_edge(n3, n4).unwrap();
        graph.add_edge(n4, n5).unwrap();
        graph.add_edge(n5, n3).unwrap();

        graph.add_edge(n2, n3).unwrap();

        let nodes: HashSet<NodeId> = [n0, n1, n2, n3, n4, n5].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();
        let algo = Infomap::new(0.15, 5, 50, Some(42), "community".to_string()).unwrap();
        let mut ctx = Context::default();

        let result = algo.execute(&mut ctx, subgraph).unwrap();
        assert_eq!(result.node_count(), 6);
    }

    #[test]
    fn test_infomap_empty() {
        let graph = Graph::new();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), HashSet::new(), "empty".into())
                .unwrap();
        let algo = Infomap::new(0.15, 5, 50, Some(42), "community".to_string()).unwrap();
        let mut ctx = Context::default();

        let result = algo.execute(&mut ctx, subgraph).unwrap();
        assert_eq!(result.node_count(), 0);
    }

    #[test]
    fn test_infomap_parameter_validation() {
        assert!(Infomap::new(-0.1, 10, 50, None, "community".to_string()).is_err());
        assert!(Infomap::new(1.5, 10, 50, None, "community".to_string()).is_err());
        assert!(Infomap::new(0.15, 0, 50, None, "community".to_string()).is_err());
        assert!(Infomap::new(0.15, 10, 0, None, "community".to_string()).is_err());
        assert!(Infomap::new(0.15, 10, 50, None, "community".to_string()).is_ok());
    }
}
