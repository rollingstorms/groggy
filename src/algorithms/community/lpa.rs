use std::collections::HashMap;
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

    fn initialise_labels(&self, subgraph: &Subgraph) -> Result<HashMap<NodeId, AttrValue>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let mut labels = HashMap::with_capacity(nodes.len());
        if nodes.is_empty() {
            return Ok(labels);
        }

        let graph_handle = subgraph.graph();
        let graph = graph_handle.borrow();
        for &node in &nodes {
            let label = if let Some(seed_attr) = &self.seed_attr {
                graph
                    .get_node_attr(node, seed_attr)?
                    .unwrap_or(AttrValue::Int(node as i64))
            } else {
                AttrValue::Int(node as i64)
            };
            labels.insert(node, label);
        }
        Ok(labels)
    }

    fn dominant_label(
        &self,
        neighbors: &[NodeId],
        labels: &HashMap<NodeId, AttrValue>,
        current: &AttrValue,
    ) -> AttrValue {
        if neighbors.is_empty() {
            return current.clone();
        }

        let mut counts: HashMap<AttrValue, usize> = HashMap::new();
        counts.insert(current.clone(), 1);
        for neighbor in neighbors {
            let label = labels
                .get(neighbor)
                .cloned()
                .unwrap_or(AttrValue::Int(*neighbor as i64));
            *counts.entry(label).or_insert(0) += 1;
        }

        let mut best_label = None;
        for (label, count) in counts.into_iter() {
            let repr = format!("{:?}", &label);
            match &mut best_label {
                None => best_label = Some((label, count, repr)),
                Some((best, best_count, best_repr)) => {
                    if count > *best_count || (count == *best_count && repr < *best_repr) {
                        *best = label;
                        *best_count = count;
                        *best_repr = repr;
                    }
                }
            }
        }

        best_label
            .map(|(label, _, _)| label)
            .unwrap_or_else(|| current.clone())
    }

    fn run_iterations(
        &self,
        ctx: &mut Context,
        subgraph: &Subgraph,
        mut labels: HashMap<NodeId, AttrValue>,
    ) -> Result<HashMap<NodeId, AttrValue>> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let total = nodes.len();
        if total == 0 {
            return Ok(labels);
        }

        for iteration in 0..self.max_iter {
            if ctx.is_cancelled() {
                return Err(anyhow!("label propagation cancelled"));
            }

            let mut updates = 0usize;
            for &node in &nodes {
                let neighbors = subgraph.neighbors(node)?;
                let current = labels
                    .get(&node)
                    .cloned()
                    .unwrap_or(AttrValue::Int(node as i64));
                let candidate = self.dominant_label(&neighbors, &labels, &current);
                if candidate != current {
                    updates += 1;
                    labels.insert(node, candidate);
                }
            }

            ctx.emit_iteration(iteration, updates);

            let change_ratio = updates as f64 / total as f64;
            if change_ratio <= self.tolerance {
                break;
            }
        }

        Ok(labels)
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
        let labels = self.initialise_labels(&subgraph)?;
        let start = Instant::now();
        let labels = self.run_iterations(ctx, &subgraph, labels)?;
        ctx.record_duration("community.lpa", start.elapsed());

        let mut attrs = HashMap::new();
        attrs.insert(
            self.output_attr.clone(),
            labels.into_iter().collect::<Vec<(NodeId, AttrValue)>>(),
        );
        subgraph
            .set_node_attrs(attrs)
            .map_err(|err| anyhow!("failed to persist labels: {err}"))?;
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
        let expected_labels = {
            let initial = algo.initialise_labels(&subgraph).unwrap();
            let neighbors = subgraph.neighbors(b).unwrap();
            let current = initial.get(&b).unwrap().clone();
            let candidate = algo.dominant_label(&neighbors, &initial, &current);
            assert_ne!(candidate, current);
            let neighbor_label = initial.get(&a).unwrap().clone();
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

        assert_eq!(expected_labels.get(&a), expected_labels.get(&b));
        assert_eq!(expected_labels.get(&c), expected_labels.get(&d));
        assert_ne!(expected_labels.get(&a), expected_labels.get(&c));

        assert_eq!(attr_a, attr_b);
        assert_eq!(attr_c, attr_d);
        assert_ne!(attr_a, attr_c);
    }
}
