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
pub struct PageRank {
    damping: f64,
    max_iter: usize,
    tolerance: f64,
    personalization_attr: Option<AttrName>,
    output_attr: AttrName,
}

impl PageRank {
    pub fn new(
        damping: f64,
        max_iter: usize,
        tolerance: f64,
        personalization_attr: Option<AttrName>,
        output_attr: AttrName,
    ) -> Result<Self> {
        if !(0.0..1.0).contains(&damping) {
            return Err(anyhow!("damping must be in [0,1)"));
        }
        if max_iter == 0 {
            return Err(anyhow!("max_iter must be greater than zero"));
        }
        if tolerance <= 0.0 {
            return Err(anyhow!("tolerance must be positive"));
        }
        Ok(Self {
            damping,
            max_iter,
            tolerance,
            personalization_attr,
            output_attr,
        })
    }

    fn metadata_template() -> AlgorithmMetadata {
        AlgorithmMetadata {
            id: "centrality.pagerank".to_string(),
            name: "PageRank".to_string(),
            description: "Power-iteration PageRank centrality.".to_string(),
            version: "0.1.0".to_string(),
            cost_hint: CostHint::Linear,
            supports_cancellation: true,
            parameters: vec![
                ParameterMetadata {
                    name: "damping".to_string(),
                    description: "Random reset probability (1-damping).".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(0.85)),
                },
                ParameterMetadata {
                    name: "max_iter".to_string(),
                    description: "Maximum number of power iterations.".to_string(),
                    value_type: ParameterType::Int,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Int(50)),
                },
                ParameterMetadata {
                    name: "tolerance".to_string(),
                    description: "Residual threshold for convergence.".to_string(),
                    value_type: ParameterType::Float,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Float(1e-6)),
                },
                ParameterMetadata {
                    name: "personalization_attr".to_string(),
                    description: "Optional node attribute containing personalization weights."
                        .to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: None,
                },
                ParameterMetadata {
                    name: "output_attr".to_string(),
                    description: "Attribute name to store PageRank scores.".to_string(),
                    value_type: ParameterType::Text,
                    required: false,
                    default_value: Some(AlgorithmParamValue::Text("pagerank".to_string())),
                },
            ],
        }
    }
}

impl Algorithm for PageRank {
    fn id(&self) -> &'static str {
        "centrality.pagerank"
    }

    fn metadata(&self) -> AlgorithmMetadata {
        Self::metadata_template()
    }

    fn execute(&self, ctx: &mut Context, subgraph: Subgraph) -> Result<Subgraph> {
        let nodes: Vec<NodeId> = subgraph.nodes().iter().copied().collect();
        let n = nodes.len();
        if n == 0 {
            return Ok(subgraph);
        }

        let damping = self.damping;
        let teleport = 1.0 - damping;

        let personalization = if let Some(attr) = &self.personalization_attr {
            let mut weights = HashMap::new();
            let mut total = 0.0;
            for &node in &nodes {
                let value = subgraph
                    .get_node_attribute(node, attr)?
                    .and_then(|v| match v {
                        AttrValue::Float(f) => Some(f as f64),
                        AttrValue::Int(ix) => Some(ix as f64),
                        _ => None,
                    })
                    .unwrap_or(1.0);
                total += value;
                weights.insert(node, value);
            }
            if total == 0.0 {
                None
            } else {
                for value in weights.values_mut() {
                    *value /= total;
                }
                Some(weights)
            }
        } else {
            None
        };

        let mut rank: HashMap<NodeId, f64> =
            nodes.iter().map(|&node| (node, 1.0 / n as f64)).collect();
        let mut next_rank = rank.clone();

        let start = Instant::now();
        for iteration in 0..self.max_iter {
            if ctx.is_cancelled() {
                return Err(anyhow!("pagerank cancelled"));
            }

            let mut residual = 0.0;
            for &node in &nodes {
                next_rank.insert(node, 0.0);
            }

            for &node in &nodes {
                let outgoing = subgraph.neighbors(node)?;
                let score = rank[&node];
                if outgoing.is_empty() {
                    let distribute = score / n as f64;
                    for &target in &nodes {
                        *next_rank.entry(target).or_default() += damping * distribute;
                    }
                } else {
                    let distribute = score / outgoing.len() as f64;
                    for neighbor in outgoing {
                        *next_rank.entry(neighbor).or_default() += damping * distribute;
                    }
                }
            }

            for &node in &nodes {
                let base = if let Some(weights) = &personalization {
                    weights.get(&node).copied().unwrap_or(0.0) * teleport
                } else {
                    teleport / n as f64
                };
                let updated = base + next_rank[&node];
                residual += (updated - rank[&node]).abs();
                rank.insert(node, updated);
            }

            ctx.emit_iteration(iteration, 0);

            if residual <= self.tolerance {
                break;
            }
        }
        ctx.record_duration("centrality.pagerank", start.elapsed());

        let mut attrs: HashMap<AttrName, Vec<(NodeId, AttrValue)>> = HashMap::new();
        attrs.insert(
            self.output_attr.clone(),
            rank.into_iter()
                .map(|(node, score)| (node, AttrValue::Float(score as f32)))
                .collect(),
        );

        subgraph
            .set_node_attrs(attrs)
            .map_err(|err| anyhow!("failed to persist PageRank scores: {err}"))?;
        Ok(subgraph)
    }
}

pub fn register(registry: &Registry) -> Result<()> {
    let metadata = PageRank::metadata_template();
    let id = metadata.id.clone();
    registry.register_with_metadata(id.as_str(), metadata, |spec| {
        let damping = spec.params.get_float("damping").unwrap_or(0.85);
        let max_iter = spec.params.get_int("max_iter").unwrap_or(50).max(1) as usize;
        let tolerance = spec.params.get_float("tolerance").unwrap_or(1e-6);
        let personalization_attr = spec
            .params
            .get_text("personalization_attr")
            .map(|s| s.to_string());
        let output_attr = spec
            .params
            .get_text("output_attr")
            .unwrap_or("pagerank")
            .to_string();
        PageRank::new(
            damping,
            max_iter,
            tolerance,
            personalization_attr.map(AttrName::from),
            output_attr.into(),
        )
        .map(|algo| Box::new(algo) as Box<dyn Algorithm>)
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algorithms::Context;
    use crate::api::graph::Graph;
    use crate::traits::SubgraphOperations;
    use std::cell::RefCell;
    use std::collections::HashSet;
    use std::rc::Rc;

    #[test]
    fn pagerank_handles_small_graph() {
        let mut graph = Graph::new();
        let a = graph.add_node();
        let b = graph.add_node();
        let c = graph.add_node();

        graph.add_edge(a, b).unwrap();
        graph.add_edge(b, c).unwrap();
        graph.add_edge(c, a).unwrap();

        let nodes: HashSet<NodeId> = [a, b, c].into_iter().collect();
        let subgraph =
            Subgraph::from_nodes(Rc::new(RefCell::new(graph)), nodes, "test".into()).unwrap();

        let algo = PageRank::new(0.85, 30, 1e-6, None, "pagerank".into()).unwrap();
        let mut ctx = Context::new();
        let result = algo.execute(&mut ctx, subgraph).unwrap();

        let attr = result
            .get_node_attribute(a, &"pagerank".into())
            .unwrap()
            .unwrap();
        if let AttrValue::Float(score) = attr {
            assert!(score > 0.0);
        } else {
            panic!("expected float score");
        }
    }
}
