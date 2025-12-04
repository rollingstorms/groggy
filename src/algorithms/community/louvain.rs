use std::collections::{HashMap, HashSet};
use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

use crate::algorithms::community::modularity::{modularity_delta, ModularityData};
use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
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

#[derive(Clone, Debug)]
pub struct Louvain {
    max_iter: usize,
    max_phases: usize,
    #[allow(dead_code)]
    resolution: f64,
    output_attr: AttrName,
}

impl Louvain {
    pub fn new(
        max_iter: usize,
        max_phases: usize,
        resolution: f64,
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
            output_attr,
        })
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.louvain".to_string(),
            name: "Louvain".to_string(),
            description: "Greedy modularity optimisation for community detection.".to_string(),
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
                    description: "Maximum number of coarse-graining phases.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(1)),
                },
                ParameterMetadata {
                    name: "resolution".to_string(),
                    description: "Resolution parameter (currently informational).".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(1.0)),
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
}

impl Algorithm for Louvain {
    fn id(&self) -> &'static str {
        "community.louvain"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let start_time = Instant::now();

        // === PHASE 1: Collect Nodes ===
        let collect_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("louvain.collect_nodes", collect_start.elapsed());
        ctx.record_stat("louvain.count.input_nodes", nodes.len() as f64);

        if nodes.is_empty() {
            return Ok(subgraph);
        }

        // === PHASE 2: Build Indexer ===
        let idx_start = Instant::now();
        let indexer = NodeIndexer::new(&nodes);
        ctx.record_call("louvain.build_indexer", idx_start.elapsed());

        // === PHASE 3: Collect Edges ===
        let edges_start = Instant::now();
        let edges = subgraph.ordered_edges();
        ctx.record_call("louvain.collect_edges", edges_start.elapsed());
        ctx.record_stat("louvain.count.input_edges", edges.len() as f64);

        if edges.is_empty() {
            return Ok(subgraph);
        }

        // === PHASE 4: Build or Get CSR ===
        let is_directed = subgraph.graph().borrow().is_directed();
        let add_reverse = !is_directed; // Louvain needs bidirectional for undirected

        let csr = match subgraph.csr_cache_get(add_reverse) {
            Some(cached) => {
                ctx.record_call("louvain.csr_cache_hit", std::time::Duration::from_nanos(0));
                cached
            }
            None => {
                let cache_miss_start = Instant::now();

                let graph = subgraph.graph();
                let graph_ref = graph.borrow();
                let pool_ref = graph_ref.pool();

                let mut csr = Csr::default();
                let build_start = Instant::now();
                let endpoint_duration = build_csr_from_edges_with_scratch(
                    &mut csr,
                    nodes.len(),
                    edges.iter().copied(),
                    |nid| indexer.get(nid),
                    |eid| pool_ref.get_edge_endpoints(eid),
                    CsrOptions {
                        add_reverse_edges: add_reverse,
                        sort_neighbors: false,
                    },
                );
                let total_build = build_start.elapsed();
                let core_build = total_build.saturating_sub(endpoint_duration);

                ctx.record_call("louvain.csr_cache_miss", cache_miss_start.elapsed());
                ctx.record_call("louvain.collect_edge_endpoints", endpoint_duration);
                ctx.record_call("louvain.build_csr", core_build);

                let csr_arc = Arc::new(csr);
                subgraph.csr_cache_store(add_reverse, csr_arc.clone());
                csr_arc
            }
        };

        ctx.record_stat("louvain.count.csr_nodes", csr.node_count() as f64);
        ctx.record_stat("louvain.count.csr_edges", csr.neighbors.len() as f64);

        // === PHASE 5: Build Edge List and Modularity Data ===
        let mod_start = Instant::now();

        // Build undirected edge list from edges
        let edge_list: Vec<(NodeId, NodeId)> = {
            let graph = subgraph.graph();
            let graph_ref = graph.borrow();
            let pool_ref = graph_ref.pool();

            let mut list = Vec::new();
            for &edge_id in edges.iter() {
                if let Some((u, v)) = pool_ref.get_edge_endpoints(edge_id) {
                    if u != v {
                        let pair = if u <= v { (u, v) } else { (v, u) };
                        list.push(pair);
                    }
                }
            }
            list
        };

        let modularity_data = ModularityData::new(&edge_list);
        ctx.record_call("louvain.build_modularity_data", mod_start.elapsed());
        ctx.record_stat("louvain.count.unique_edges", edge_list.len() as f64);

        // === PHASE 6: Initialize Partition ===
        let init_start = Instant::now();
        let mut partition: HashMap<NodeId, usize> = HashMap::new();
        partition.reserve(nodes.len());
        for (idx, &node) in nodes.iter().enumerate() {
            partition.insert(node, idx);
        }
        ctx.record_call("louvain.initialize_partition", init_start.elapsed());

        // Track community degrees and internal edges incrementally
        let mut community_degrees: HashMap<usize, f64> = HashMap::new();
        let mut community_internal: HashMap<usize, f64> = HashMap::new();

        // Initialize community stats (each node starts in its own community)
        for (&node, &comm) in &partition {
            let deg = modularity_data.degree(&node);
            *community_degrees.entry(comm).or_insert(0.0) += deg;
            *community_internal.entry(comm).or_insert(0.0) += 0.0;
        }

        let epsilon = 1e-6;

        // === PHASE 7: Louvain Iterations ===
        let compute_start = Instant::now();
        let mut total_moves = 0;
        let mut total_iterations = 0;

        if self.max_phases > 0 {
            let _phase = 0;
            let phase_start = Instant::now();
            let mut improved = false;

            for iteration in 0..self.max_iter {
                if ctx.is_cancelled() {
                    return Err(anyhow!("louvain cancelled"));
                }

                let iter_start = Instant::now();
                let mut changed = false;
                let mut moves_this_iter = 0;

                for (node_idx, &node) in nodes.iter().enumerate() {
                    let current_comm = partition[&node];

                    // Find candidate communities using CSR neighbors
                    let mut candidate_comms: HashSet<usize> = HashSet::new();
                    candidate_comms.insert(current_comm);

                    for &neighbor_idx in csr.neighbors(node_idx) {
                        if neighbor_idx < nodes.len() {
                            let neighbor_node = nodes[neighbor_idx];
                            if let Some(&comm) = partition.get(&neighbor_node) {
                                candidate_comms.insert(comm);
                            }
                        }
                    }

                    let mut best_local_comm = current_comm;
                    let mut best_delta = 0.0;

                    // Build temp adjacency for this node (for modularity_delta)
                    let mut node_adjacency: HashMap<NodeId, Vec<NodeId>> = HashMap::new();
                    let mut neighbors_vec = Vec::new();
                    for &neighbor_idx in csr.neighbors(node_idx) {
                        if neighbor_idx < nodes.len() {
                            neighbors_vec.push(nodes[neighbor_idx]);
                        }
                    }
                    node_adjacency.insert(node, neighbors_vec);

                    // Find best move
                    for &candidate in &candidate_comms {
                        if candidate == current_comm {
                            continue;
                        }

                        let delta = modularity_delta(
                            node,
                            current_comm,
                            candidate,
                            &partition,
                            &node_adjacency,
                            &modularity_data,
                            &community_degrees,
                            &community_internal,
                        );

                        if delta > best_delta + epsilon {
                            best_delta = delta;
                            best_local_comm = candidate;
                        }
                    }

                    // Apply best move
                    if best_local_comm != current_comm {
                        let node_degree = modularity_data.degree(&node);

                        // Update old community
                        if let Some(deg) = community_degrees.get_mut(&current_comm) {
                            *deg -= node_degree;
                        }

                        // Update new community
                        *community_degrees.entry(best_local_comm).or_insert(0.0) += node_degree;

                        // Move node
                        partition.insert(node, best_local_comm);
                        changed = true;
                        moves_this_iter += 1;
                    }
                }

                total_iterations += 1;
                total_moves += moves_this_iter;

                ctx.record_call("louvain.compute.iter", iter_start.elapsed());
                ctx.record_stat("louvain.count.iteration", (iteration + 1) as f64);
                ctx.record_stat("louvain.count.moves", moves_this_iter as f64);

                if !changed {
                    break;
                }
                improved = true;
            }

            ctx.record_call("louvain.compute.phase", phase_start.elapsed());
            ctx.record_stat("louvain.phase.improved", if improved { 1.0 } else { 0.0 });

            if !improved {
                ctx.record_call("louvain.compute", compute_start.elapsed());
                ctx.record_stat("louvain.total_iterations", total_iterations as f64);
                ctx.record_stat("louvain.total_moves", total_moves as f64);
                return self.persist_partition(ctx, subgraph, partition, start_time);
            }
        }

        ctx.record_call("louvain.compute", compute_start.elapsed());
        ctx.record_stat("louvain.total_iterations", total_iterations as f64);
        ctx.record_stat("louvain.total_moves", total_moves as f64);

        self.persist_partition(ctx, subgraph, partition, start_time)
    }
}

impl Louvain {
    fn persist_partition(
        &self,
        ctx: &mut Context,
        subgraph: Subgraph,
        partition: HashMap<NodeId, usize>,
        start_time: Instant,
    ) -> Result<Subgraph> {
        if ctx.persist_results() {
            let write_start = Instant::now();
            let attr_values: Vec<(NodeId, AttrValue)> = partition
                .iter()
                .map(|(&node, &community)| (node, AttrValue::Int(community as i64)))
                .collect();

            subgraph
                .set_node_attr_column(self.output_attr.clone(), attr_values)
                .map_err(|err| anyhow!("failed to persist Louvain communities: {err}"))?;

            ctx.record_call("louvain.write_attributes", write_start.elapsed());
        } else {
            let store_start = Instant::now();
            ctx.record_call("louvain.store_output", store_start.elapsed());
        }

        ctx.record_call("louvain.total_execution", start_time.elapsed());

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = Louvain::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let max_iter = spec.params.get_int("max_iter").unwrap_or(20).max(1) as usize;
        let max_phases = spec.params.get_int("max_phases").unwrap_or(1).max(1) as usize;
        let resolution = spec.params.get_float("resolution").unwrap_or(1.0);
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("community")
            .to_string();
        Louvain::new(max_iter, max_phases, resolution, output_attr)
            .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::types::AttrName;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn louvain_separates_components() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();
        let d = graph.add_node();

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, a).unwrap();
        graph.add_edge(c, d).unwrap();
        graph.add_edge(d, c).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c, d].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = Louvain::new(10, 1, 1.0, "community".into()).unwrap();
        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();
        let attr_a = result.get_node_attribute(a, &attr_name).unwrap().unwrap();
        let attr_b = result.get_node_attribute(b, &attr_name).unwrap().unwrap();
        let attr_c = result.get_node_attribute(c, &attr_name).unwrap().unwrap();
        let attr_d = result.get_node_attribute(d, &attr_name).unwrap().unwrap();

        assert_eq!(attr_a, attr_b);
        assert_eq!(attr_c, attr_d);
        assert_ne!(attr_a, attr_c);
    }
}
