use std::collections::HashMap;
use std::hash::{Hash, Hasher};
use std::time::Instant;

use anyhow::{anyhow, Result};

use crate::algorithms::registry::Registry;
use crate::algorithms::{
    Algorithm, AlgorithmMetadata, AlgorithmParamValue, Context, CostHint, ParameterMetadata,
    ParameterType,
};
use crate::subgraphs::Subgraph;
use crate::traits::SubgraphOperations;
use crate::types::{AttrName, AttrValue, NodeId};

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
    node_index: HashMap<NodeId, usize>,
}

impl LabelState {
    fn new(nodes: Vec<NodeId>, labels: Vec<AttrValue>) -> Self {
        let mut node_index = HashMap::with_capacity(nodes.len());
        for (idx, node) in nodes.iter().copied().enumerate() {
            node_index.insert(node, idx);
        }
        Self {
            nodes,
            labels,
            node_index,
        }
    }

    #[cfg(test)]
    fn label_for(&self, node: NodeId) -> Option<&AttrValue> {
        self.node_index
            .get(&node)
            .and_then(|&idx| self.labels.get(idx))
    }

    #[cfg(test)]
    fn index_of(&self, node: NodeId) -> Option<usize> {
        self.node_index.get(&node).copied()
    }

    #[cfg(test)]
    fn labels(&self) -> &[AttrValue] {
        &self.labels
    }

    fn into_pairs(self) -> Vec<(NodeId, AttrValue)> {
        self.nodes
            .into_iter()
            .zip(self.labels.into_iter())
            .collect()
    }
}

#[derive(Clone, Copy)]
struct LabelStats {
    count: usize,
    order_key: u64,
}

fn label_order_key(value: &AttrValue) -> u64 {
    let mut hasher = std::collections::hash_map::DefaultHasher::new();
    value.hash(&mut hasher);
    hasher.finish()
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

    fn initialise_labels(&self, subgraph: &Subgraph) -> Result<LabelState> {
        let mut nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        nodes.sort_unstable();
        let mut labels = Vec::with_capacity(nodes.len());
        if nodes.is_empty() {
            return Ok(LabelState::new(nodes, labels));
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
        Ok(LabelState::new(nodes, labels))
    }

    fn dominant_label(
        &self,
        neighbor_indices: &[usize],
        labels: &[AttrValue],
        current: &AttrValue,
        counts: &mut HashMap<AttrValue, LabelStats>,
    ) -> AttrValue {
        if neighbor_indices.is_empty() {
            return current.clone();
        }

        counts.clear();
        counts.insert(
            current.clone(),
            LabelStats {
                count: 1,
                order_key: label_order_key(current),
            },
        );

        for &neighbor_idx in neighbor_indices {
            if let Some(label) = labels.get(neighbor_idx) {
                let entry = counts.entry(label.clone()).or_insert_with(|| LabelStats {
                    count: 0,
                    order_key: label_order_key(label),
                });
                entry.count += 1;
            }
        }

        let mut best_label = None;
        for (label, stats) in counts.iter() {
            match best_label {
                None => {
                    best_label = Some((label, stats.count, stats.order_key));
                }
                Some((_, best_count, best_order)) => {
                    if stats.count > best_count
                        || (stats.count == best_count && stats.order_key < best_order)
                    {
                        best_label = Some((label, stats.count, stats.order_key));
                    }
                }
            }
        }

        best_label
            .map(|(label, _, _)| label.clone())
            .unwrap_or_else(|| current.clone())
    }

    fn run_iterations(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
        mut state: LabelState,
    ) -> Result<LabelState> {
        let total = state.nodes.len();
        if total == 0 {
            return Ok(state);
        }

        let mut neighbor_indices: Vec<Vec<usize>> = Vec::with_capacity(total);
        for &node in &state.nodes {
            let neighbors = subgraph.neighbors(node)?;
            let mapped = neighbors
                .into_iter()
                .filter_map(|n| state.node_index.get(&n).copied())
                .collect();
            neighbor_indices.push(mapped);
        }

        let mut counts: HashMap<AttrValue, LabelStats> = HashMap::new();
        for iteration in 0..self.max_iter {
            if ctx.is_cancelled() {
                return Err(anyhow!("label propagation cancelled"));
            }

            let mut updates = 0usize;
            for (idx, neighbor_list) in neighbor_indices.iter().enumerate() {
                let current = state
                    .labels
                    .get(idx)
                    .cloned()
                    .unwrap_or_else(|| AttrValue::Int(state.nodes[idx] as i64));
                let candidate =
                    self.dominant_label(neighbor_list, &state.labels, &current, &mut counts);
                if candidate != current {
                    updates += 1;
                    state.labels[idx] = candidate;
                }
            }

            ctx.emit_iteration(iteration, updates);

            let change_ratio = updates as f64 / total as f64;
            if change_ratio <= self.tolerance {
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
        let state = self.initialise_labels(&subgraph)?;
        let start = Instant::now();
        let state = self.run_iterations(ctx, &subgraph, state)?;
        ctx.record_duration("community.lpa", start.elapsed());

        if ctx.persist_results() {
            let attr_values = state.into_pairs();
            ctx.with_scoped_timer("community.lpa.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist labels: {err}"))?;
        }
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

        let mut neighbor_check = subgraph.neighbors(b).unwrap();
        neighbor_check.sort_unstable();
        assert!(neighbor_check.contains(&a));

        let algo = LabelPropagation::new(10, 0.0, "community".into(), None).unwrap();

        let mut analytic_ctx = Context::new();
        let expected_state = {
            let initial = algo.initialise_labels(&subgraph).unwrap();
            let neighbors = subgraph.neighbors(b).unwrap();
            let neighbor_indices: Vec<usize> = neighbors
                .iter()
                .filter_map(|&node| initial.index_of(node))
                .collect();
            let current = initial.label_for(b).unwrap().clone();
            let mut counts = HashMap::new();
            let candidate =
                algo.dominant_label(&neighbor_indices, initial.labels(), &current, &mut counts);
            assert_ne!(candidate, current);
            let neighbor_label = initial.label_for(a).unwrap().clone();
            assert_eq!(candidate, neighbor_label);
            algo.run_iterations(&mut analytic_ctx, &subgraph, initial)
                .unwrap()
        };

        let mut exec_ctx = Context::new();
        let result = algo.execute(&mut exec_ctx, subgraph).unwrap();

        let attr_name: AttrName = "community".to_string();
        let attr_a = result.get_node_attribute(a, &attr_name).unwrap().unwrap();
        let attr_b = result.get_node_attribute(b, &attr_name).unwrap().unwrap();
        let attr_c = result.get_node_attribute(c, &attr_name).unwrap().unwrap();
        let attr_d = result.get_node_attribute(d, &attr_name).unwrap().unwrap();

        assert_eq!(expected_state.label_for(a), expected_state.label_for(b));
        assert_eq!(expected_state.label_for(c), expected_state.label_for(d));
        assert_ne!(expected_state.label_for(a), expected_state.label_for(c));

        assert_eq!(attr_a, attr_b);
        assert_eq!(attr_c, attr_d);
        assert_ne!(attr_a, attr_c);
    }
}
