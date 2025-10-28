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

        // Build NodeId -> index mapping for Vec-based storage
        let mut node_to_idx: HashMap<NodeId, usize> = HashMap::with_capacity(n);
        for (idx, &node) in nodes.iter().enumerate() {
            node_to_idx.insert(node, idx);
        }

        let damping = self.damping;
        let teleport = 1.0 - damping;
        let teleport_per_node = teleport / n as f64;

        // Precompute personalization weights as flat Vec
        let personalization: Option<Vec<f64>> = if let Some(attr) = &self.personalization_attr {
            let mut weights = vec![1.0; n];
            let mut total = 0.0;
            for (idx, &node) in nodes.iter().enumerate() {
                let value = subgraph
                    .get_node_attribute(node, attr)?
                    .and_then(|v| match v {
                        AttrValue::Float(f) => Some(f as f64),
                        AttrValue::Int(ix) => Some(ix as f64),
                        _ => None,
                    })
                    .unwrap_or(1.0);
                weights[idx] = value;
                total += value;
            }
            if total == 0.0 {
                None
            } else {
                for w in &mut weights {
                    *w = (*w / total) * teleport;
                }
                Some(weights)
            }
        } else {
            None
        };

        // Precompute adjacency lists and out-degrees
        let mut adjacency: Vec<Vec<usize>> = vec![Vec::new(); n];
        let mut out_degree: Vec<f64> = vec![0.0; n];
        
        for (idx, &node) in nodes.iter().enumerate() {
            let neighbors = subgraph.neighbors(node)?;
            adjacency[idx].reserve(neighbors.len());
            for neighbor in neighbors {
                if let Some(&neighbor_idx) = node_to_idx.get(&neighbor) {
                    adjacency[idx].push(neighbor_idx);
                }
            }
            out_degree[idx] = adjacency[idx].len() as f64;
        }

        // Initialize rank vectors
        let init_rank = 1.0 / n as f64;
        let mut rank: Vec<f64> = vec![init_rank; n];
        let mut next_rank: Vec<f64> = vec![0.0; n];

        let start = Instant::now();
        for iteration in 0..self.max_iter {
            if ctx.is_cancelled() {
                return Err(anyhow!("pagerank cancelled"));
            }

            // Zero out next_rank buffer
            next_rank.fill(0.0);

            // Aggregate sink mass in one pass
            let mut sink_mass = 0.0;
            for idx in 0..n {
                if out_degree[idx] == 0.0 {
                    sink_mass += rank[idx];
                }
            }
            let sink_contribution = damping * sink_mass / n as f64;

            // Distribute rank from non-sink nodes
            for idx in 0..n {
                if out_degree[idx] > 0.0 {
                    let contrib = damping * rank[idx] / out_degree[idx];
                    for &neighbor_idx in &adjacency[idx] {
                        next_rank[neighbor_idx] += contrib;
                    }
                }
            }

            // Add teleport and sink contributions, compute residual
            let mut max_diff = 0.0;
            for idx in 0..n {
                let teleport_contrib = if let Some(weights) = &personalization {
                    weights[idx]
                } else {
                    teleport_per_node
                };
                let new_rank = teleport_contrib + next_rank[idx] + sink_contribution;
                let diff = (new_rank - rank[idx]).abs();
                if diff > max_diff {
                    max_diff = diff;
                }
                next_rank[idx] = new_rank;
            }

            // Swap buffers instead of copying
            std::mem::swap(&mut rank, &mut next_rank);

            ctx.emit_iteration(iteration, 0);

            if max_diff <= self.tolerance {
                break;
            }
        }
        ctx.record_duration("centrality.pagerank", start.elapsed());

        if ctx.persist_results() {
            let attr_values: Vec<(NodeId, AttrValue)> = nodes
                .iter()
                .enumerate()
                .map(|(idx, &node)| (node, AttrValue::Float(rank[idx] as f32)))
                .collect();

            ctx.with_scoped_timer("centrality.pagerank.write_attrs", || {
                subgraph.set_node_attr_column(self.output_attr.clone(), attr_values)
            })
            .map_err(|err| anyhow!("failed to persist PageRank scores: {err}"))?;
        }
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
