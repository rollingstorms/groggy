use std::sync::Arc;
use std::time::Instant;

use anyhow::{anyhow, Result};
use rustc_hash::FxHashMap;

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
    Dense {
        min_id: NodeId,
        indices: Vec<u32>, // sentinel = u32::MAX for missing nodes
    },
    Sparse(FxHashMap<NodeId, usize>),
}

impl NodeIndexer {
    /// Constructs the optimal indexer based on node ID distribution.
    /// Uses dense array if the ID span is ≤ 1.5x node count.
    fn new(nodes: &[NodeId]) -> Self {
        if nodes.is_empty() {
            return Self::Sparse(FxHashMap::default());
        }

        let min = *nodes.iter().min().unwrap();
        let max = *nodes.iter().max().unwrap();
        let span = (max - min) as usize + 1;

        // Dense indexing threshold: span must be reasonable relative to node count
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
pub struct LabelPropagation {
    max_iter: usize,
    tolerance: f64,
    output_attr: AttrName,
    seed_attr: Option<AttrName>,
}

#[derive(Clone)]
struct LabelState {
    nodes: Vec<NodeId>,
    labels: Vec<AttrValue>,
}

#[allow(dead_code)]
impl LabelState {
    fn new(nodes: Vec<NodeId>, labels: Vec<AttrValue>) -> Self {
        Self { nodes, labels }
    }

    #[cfg(test)]
    fn label_for(&self, node: NodeId, indexer: &NodeIndexer) -> Option<&AttrValue> {
        indexer.get(node).and_then(|idx| self.labels.get(idx))
    }

    fn into_pairs(self) -> Vec<(NodeId, AttrValue)> {
        self.nodes
            .into_iter()
            .zip(self.labels.into_iter())
            .collect()
    }
}

impl LabelPropagation {
    pub fn new(
        max_iter: usize,
        tolerance: f64,
        output_attr: AttrName,
        seed_attr: Option<AttrName>,
    ) -> Result<Self> {
        if max_iter == 0 {
            return Err(anyhow!("max_iter must be greater than zero"));
        }
        if tolerance < 0.0 {
            return Err(anyhow!("tolerance must be non-negative"));
        }
        Ok(Self {
            max_iter,
            tolerance,
            output_attr,
            seed_attr,
        })
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "community.lpa".to_string(),
            name: "Label Propagation".to_string(),
            description: "Asynchronous label propagation for community detection.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linear,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "max_iter".to_string(),
                    description: "Maximum number of propagation iterations.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(20)),
                },
                ParameterMetadata {
                    name: "tolerance".to_string(),
                    description: "Fraction of nodes allowed to change before converging."
                        .to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(0.01)),
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store the resulting labels.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("community".to_string())),
                },
                ParameterMetadata {
                    name: "seed_attr".to_string(),
                    description: "Optional attribute containing initial labels.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
            ],
        }
    }

    fn initialise_labels(&self, nodes: &[NodeId], subgraph: &Subgraph) -> Result<LabelState> {
        let mut labels = Vec::with_capacity(nodes.len());
        if nodes.is_empty() {
            return Ok(LabelState::new(nodes.to_vec(), labels));
        }

        let graph_handle = subgraph.graph();
        let graph = graph_handle.borrow();
        for &node in nodes.iter() {
            let label = if let Some(seed_attr) = &self.seed_attr {
                graph
                    .get_node_attr(node, seed_attr)?
                    .unwrap_or(AttrValue::Int(node as i64))
            } else {
                AttrValue::Int(node as i64)
            };
            labels.push(label);
        }
        Ok(LabelState::new(nodes.to_vec(), labels))
    }

    fn dominant_label(
        &self,
        neighbor_slice: &[usize],
        labels: &[AttrValue],
        current_idx: usize,
        counts: &mut FxHashMap<usize, usize>, // Index → count
    ) -> usize {
        if neighbor_slice.is_empty() {
            return current_idx;
        }

        counts.clear();
        counts.insert(current_idx, 1);

        // Count neighbor labels by index (much faster than AttrValue)
        for &neighbor_idx in neighbor_slice {
            if neighbor_idx < labels.len() {
                *counts.entry(neighbor_idx).or_insert(0) += 1;
            }
        }

        // Find most common label
        let mut best_idx = current_idx;
        let mut best_count = 1;

        for (&idx, &count) in counts.iter() {
            // Break ties by smallest index (deterministic)
            if count > best_count || (count == best_count && idx < best_idx) {
                best_idx = idx;
                best_count = count;
            }
        }

        best_idx
    }

    fn run_iterations(
        &self,
        ctx: &mut Context,
        _subgraph: &Subgraph,
        _indexer: &NodeIndexer,
        csr: &Csr,
        mut state: LabelState,
    ) -> Result<LabelState> {
        let total = state.nodes.len();
        if total == 0 {
            return Ok(state);
        }

        // Reusable buffers
        let mut counts: FxHashMap<usize, usize> = FxHashMap::default();
        counts.reserve(total / 10);
        let mut neighbor_label_indices = Vec::with_capacity(100); // Reusable buffer

        // Track which label index each node currently has
        let mut label_indices: Vec<usize> = (0..total).collect();

        for iteration in 0..self.max_iter {
            if ctx.is_cancelled() {
                return Err(anyhow!("label propagation cancelled"));
            }

            let iter_start = Instant::now();
            let mut updates = 0usize;

            for idx in 0..total {
                let current_label_idx = label_indices[idx];

                // Get neighbors from CSR and build their label indices
                neighbor_label_indices.clear();
                for &neighbor_idx in csr.neighbors(idx) {
                    if neighbor_idx < total {
                        neighbor_label_indices.push(label_indices[neighbor_idx]);
                    }
                }

                let new_label_idx = self.dominant_label(
                    &neighbor_label_indices,
                    &state.labels,
                    current_label_idx,
                    &mut counts,
                );

                if new_label_idx != current_label_idx {
                    updates += 1;
                    label_indices[idx] = new_label_idx;
                    // Asynchronous update: apply immediately so subsequent nodes see it
                    if new_label_idx < state.labels.len() {
                        state.labels[idx] = state.labels[new_label_idx].clone();
                    }
                }
            }

            ctx.record_call("lpa.compute.iter", iter_start.elapsed());
            ctx.record_stat("lpa.count.iteration", (iteration + 1) as f64);
            ctx.record_stat("lpa.count.updates", updates as f64);
            ctx.emit_iteration(iteration, updates);

            let change_ratio = updates as f64 / total as f64;
            if change_ratio <= self.tolerance {
                ctx.record_stat("lpa.converged", 1.0);
                ctx.record_stat("lpa.iterations_to_converge", (iteration + 1) as f64);
                break;
            }
        }

        Ok(state)
    }
}

impl Algorithm for LabelPropagation {
    fn id(&self) -> &'static str {
        "community.lpa"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let start_time = Instant::now();

        // === PHASE 1: Collect Nodes ===
        let collect_start = Instant::now();
        let nodes = subgraph.ordered_nodes();
        ctx.record_call("lpa.collect_nodes", collect_start.elapsed());
        ctx.record_stat("lpa.count.input_nodes", nodes.len() as f64);

        if nodes.is_empty() {
            return Ok(subgraph);
        }

        // === PHASE 2: Build Indexer ===
        let idx_start = Instant::now();
        let indexer = NodeIndexer::new(&nodes);
        ctx.record_call("lpa.build_indexer", idx_start.elapsed());

        // === PHASE 3: Collect Edges ===
        let edges_start = Instant::now();
        let edges = subgraph.ordered_edges();
        ctx.record_call("lpa.collect_edges", edges_start.elapsed());
        ctx.record_stat("lpa.count.input_edges", edges.len() as f64);

        // === PHASE 4: Build or Get CSR ===
        let is_directed = subgraph.graph().borrow().is_directed();
        let add_reverse = !is_directed; // LPA needs bidirectional for undirected graphs

        let csr = match subgraph.csr_cache_get(add_reverse) {
            Some(cached) => {
                ctx.record_call("lpa.csr_cache_hit", std::time::Duration::from_nanos(0));
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

                ctx.record_call("lpa.csr_cache_miss", cache_miss_start.elapsed());
                ctx.record_call("lpa.collect_edge_endpoints", endpoint_duration);
                ctx.record_call("lpa.build_csr", core_build);

                let csr_arc = Arc::new(csr);
                subgraph.csr_cache_store(add_reverse, csr_arc.clone());
                csr_arc
            }
        };

        ctx.record_stat("lpa.count.csr_nodes", csr.node_count() as f64);
        ctx.record_stat("lpa.count.csr_edges", csr.neighbors.len() as f64);

        // === PHASE 5: Initialize Labels ===
        let init_start = Instant::now();
        let state = self.initialise_labels(&nodes, &subgraph)?;
        ctx.record_call("lpa.initialize_labels", init_start.elapsed());

        // === PHASE 6: Run Iterations ===
        let compute_start = Instant::now();
        let state = self.run_iterations(ctx, &subgraph, &indexer, &csr, state)?;
        ctx.record_call("lpa.compute", compute_start.elapsed());

        // === PHASE 7: Emit Results ===
        if ctx.persist_results() {
            let write_start = Instant::now();
            let attr_values = state.into_pairs();
            subgraph
                .set_node_attr_column(self.output_attr.clone(), attr_values)
                .map_err(|err| anyhow!("failed to persist labels: {err}"))?;
            ctx.record_call("lpa.write_attributes", write_start.elapsed());
        } else {
            let store_start = Instant::now();
            // For non-persistent results, could add to context output if needed
            ctx.record_call("lpa.store_output", store_start.elapsed());
        }

        ctx.record_call("lpa.total_execution", start_time.elapsed());

        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = LabelPropagation::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let max_iter = spec.params.get_int("max_iter").unwrap_or(20).max(1) as usize;
        let tolerance = spec.params.get_float("tolerance").unwrap_or(0.01);
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("community")
            .to_string();
        let seed_attr = spec.params.get_text("seed_attr").map(|s| s.to_string());
        LabelPropagation::new(max_iter, tolerance, output_attr, seed_attr)
            .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::traits::SubgraphOperations;
    use crate::types::AttrName;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn label_propagation_clusters_components() {
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

        let algo = LabelPropagation::new(10, 0.0, "community".into(), None).unwrap();

        let mut exec_ctx = Context::new();
        let result = algo.execute(&mut exec_ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();
        let attr_a = result.get_node_attribute(a, &attr_name).unwrap().unwrap();
        let attr_b = result.get_node_attribute(b, &attr_name).unwrap().unwrap();
        let attr_c = result.get_node_attribute(c, &attr_name).unwrap().unwrap();
        let attr_d = result.get_node_attribute(d, &attr_name).unwrap().unwrap();

        // Nodes in same component should have same label
        assert_eq!(attr_a, attr_b);
        assert_eq!(attr_c, attr_d);
        // Nodes in different components should have different labels
        assert_ne!(attr_a, attr_c);
    }
}
