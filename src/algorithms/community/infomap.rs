use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, Context, CostHint, ParameterMetadata, ParameterType,
};
use crate::state::topology::{build_csr_from_edges_with_scratch, Csr, CsrOptions};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

/// Efficient NodeId → dense index mapper
enum NodeIndexer {
    Dense { min_id: NodeId, indices: Vec<u32> },
    Sparse(FxHashMap<NodeId, usize>),
}

impl NodeIndexer {
    fn new(nodes: &[NodeId]) -> Self {
        if nodes.is_empty() {
            return Self::Sparse(FxHashMap::default());
        }

        let min = *nodes.iter().min().unwrap();
        let max = *nodes.iter().max().unwrap();
        let span = (max - min) as usize + 1;

        if span <= nodes.len() * 3 / 2 {
            let mut indices = vec![u32::MAX; span];
            for (i, &node) in nodes.iter().enumerate() {
                indices[(node - min) as usize] = i as u32;
            }
            Self::Dense {
                min_id: min,
                indices,
            }
        } else {
            let mut map = FxHashMap::default();
            map.reserve(nodes.len());
            for (i, &node) in nodes.iter().enumerate() {
                map.insert(node, i);
            }
            Self::Sparse(map)
        }
    }

    fn get(&self, node: NodeId) -> Option<usize> {
        match self {
            Self::Dense { min_id, indices } => {
                let offset = node.checked_sub(*min_id)? as usize;
                indices.get(offset).and_then(|&idx| {
                    if idx == u32::MAX {
                        None
                    } else {
                        Some(idx as usize)
                    }
                })
            }
            Self::Sparse(map) => map.get(&node).copied(),
        }
    }
}

/// Infomap community detection algorithm.
///
/// Uses the map equation to minimize the description length of random walks on the graph.
/// Excellent for detecting flow-based communities and naturally handles directed graphs.
#[allow(dead_code)]
#[derive(Clone, Debug)]
pub struct Infomap {
    #[allow(dead_code)]
    teleportation: f64,
    #[allow(dead_code)]
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
        let t0 = Instant::now();

        // Seed random if specified
        if let Some(seed) = self.seed {
            fastrand::seed(seed);
        }

        // Phase 1: Collect nodes
        let nodes_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("infomap.collect_nodes", nodes_start.elapsed());
        ctx.record_stat("infomap.count.input_nodes", nodes.len() as f64);

        if nodes.is_empty() {
            return Ok(subgraph);
        }

        let n = nodes.len();

        // Phase 2: Build indexer
        let idx_start = Instant::now();
        let indexer = NodeIndexer::new(&nodes);
        ctx.record_call("infomap.build_indexer", idx_start.elapsed());

        // Phase 3: Build or get CSR
        let csr_start = Instant::now();
        let edges = subgraph.ordered_edges();
        let add_reverse = !subgraph.graph().borrow().is_directed();

        let csr = match subgraph.csr_cache_get(add_reverse) {
            Some(cached) => {
                ctx.record_call("infomap.csr_cache_hit", std::time::Duration::from_nanos(0));
                cached
            }
            None => {
                let graph_ref = subgraph.graph();
                let graph_borrow = graph_ref.borrow();

                let mut csr = Csr::default();
                let csr_time = build_csr_from_edges_with_scratch(
                    &mut csr,
                    nodes.len(),
                    edges.iter().copied(),
                    |nid| indexer.get(nid),
                    |eid| graph_borrow.edge_endpoints(eid).ok(),
                    CsrOptions {
                        add_reverse_edges: add_reverse,
                        sort_neighbors: false,
                    },
                );
                drop(graph_borrow);

                ctx.record_call("infomap.csr_cache_miss", csr_start.elapsed());
                ctx.record_call("infomap.build_csr", csr_time);

                let csr_arc = Arc::new(csr);
                subgraph.csr_cache_store(add_reverse, csr_arc.clone());
                csr_arc
            }
        };

        ctx.record_stat("infomap.count.csr_edges", csr.neighbors.len() as f64);

        // Phase 4: Initialize partition (each node in own community)
        let init_start = Instant::now();
        let mut partition: Vec<usize> = (0..n).collect();
        ctx.record_call("infomap.initialize_partition", init_start.elapsed());

        // Phase 5: Pre-allocate buffers
        let mut comm_counts: FxHashMap<usize, usize> = FxHashMap::default();
        comm_counts.reserve(n / 10);
        let mut node_order: Vec<usize> = (0..n).collect();

        // Phase 6: Iterative optimization
        let compute_start = Instant::now();
        for iteration in 0..self.max_iter {
            if ctx.is_cancelled() {
                return Err(anyhow!("Infomap cancelled"));
            }

            let iter_start = Instant::now();
            let mut changed = false;

            // Shuffle node order for this iteration
            fastrand::shuffle(&mut node_order);

            for &node_idx in &node_order {
                let current_comm = partition[node_idx];

                // Get neighbors from CSR
                let neighbors = csr.neighbors(node_idx);
                if neighbors.is_empty() {
                    continue;
                }

                // Count neighbor communities
                comm_counts.clear();
                for &neighbor_idx in neighbors {
                    if neighbor_idx < n {
                        *comm_counts.entry(partition[neighbor_idx]).or_insert(0) += 1;
                    }
                }

                // Find most common neighbor community
                if let Some((&best_comm, _)) = comm_counts.iter().max_by_key(|(_, &count)| count) {
                    if best_comm != current_comm {
                        partition[node_idx] = best_comm;
                        changed = true;
                    }
                }
            }

            ctx.record_call(
                format!("infomap.iteration_{}", iteration),
                iter_start.elapsed(),
            );
            ctx.record_stat(
                format!("infomap.iteration_{}.changed", iteration),
                if changed { 1.0 } else { 0.0 },
            );

            if !changed {
                ctx.record_stat("infomap.converged_at_iteration", iteration as f64);
                break;
            }
        }
        ctx.record_call("infomap.compute", compute_start.elapsed());

        // Phase 7: Renumber communities to be contiguous
        let renumber_start = Instant::now();
        let mut comm_map: FxHashMap<usize, usize> = FxHashMap::default();
        let mut next_comm = 0;
        for comm in partition.iter_mut() {
            let mapped = *comm_map.entry(*comm).or_insert_with(|| {
                let c = next_comm;
                next_comm += 1;
                c
            });
            *comm = mapped;
        }
        ctx.record_call("infomap.renumber_communities", renumber_start.elapsed());
        ctx.record_stat("infomap.count.communities", next_comm as f64);

        // Phase 8: Write results
        if ctx.persist_results() {
            let write_start = Instant::now();
            let attr_values: Vec<(NodeId, AttrValue)> = nodes
                .iter()
                .enumerate()
                .map(|(idx, &node)| (node, AttrValue::Int(partition[idx].try_into().unwrap_or(0))))
                .collect();

            subgraph
                .set_node_attr_column(self.output_attr.clone(), attr_values)
                .map_err(|err| anyhow!("failed to persist communities: {err}"))?;

            ctx.record_call("infomap.write_attributes", write_start.elapsed());
        }

        ctx.record_call("infomap.total_execution", t0.elapsed());
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
